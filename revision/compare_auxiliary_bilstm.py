"""Compare residual BiLSTM forecasts with and without auxiliary reconstruction.

Both configurations use the same input, split, seeds, optimizer, and test origins.
The comparison only assesses forecasting the supplied sensor-derived VWC series.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ARCH = Path(__file__).resolve().parent / "architecture_benchmark"
SEEDS = [42, 43, 44, 45, 46]
HORIZONS = ["1h", "6h", "12h", "24h"]
BLOCKS = [48, 96, 192]
REPS = 2000
KEY = ["farm", "horizon", "origin_timestamp", "target_timestamp"]


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def compare(data: np.ndarray) -> float:
    truth = data[:, 0]
    no_aux = data[:, 2:7]
    aux = data[:, 7:12]
    return float(np.mean([rmse(truth, aux[:, j]) - rmse(truth, no_aux[:, j]) for j in range(5)]))


def main() -> None:
    all_rows = []
    for seed in SEEDS:
        no_aux = pd.read_csv(ARCH / f"bilstm_no_aux_seed{seed}" / "architecture_predictions.csv")
        with_aux = pd.read_csv((ARCH if seed == 42 else ARCH / f"seed{seed}") / "architecture_predictions.csv")
        no_aux = no_aux[no_aux.variant == "bilstm_residual"].drop(columns="variant")
        with_aux = with_aux[with_aux.variant == "shared_bilstm_residual"].drop(columns="variant")
        pair = no_aux.merge(with_aux, on=KEY, suffixes=("_no_aux", "_aux"), validate="one_to_one")
        assert len(pair) == 3006 * 4
        assert np.allclose(pair.target_vwc_pct_no_aux, pair.target_vwc_pct_aux, atol=1e-5)
        assert np.allclose(pair.current_vwc_pct_no_aux, pair.current_vwc_pct_aux, atol=1e-5)
        for horizon, part in pair.groupby("horizon"):
            truth = part.target_vwc_pct_aux.to_numpy(float)
            persistence = part.current_vwc_pct_aux.to_numpy(float)
            all_rows.append({
                "seed": seed,
                "horizon": horizon,
                "n_origins": len(part),
                "persistence_rmse": rmse(truth, persistence),
                "no_aux_rmse": rmse(truth, part.model_vwc_pct_no_aux.to_numpy(float)),
                "aux_rmse": rmse(truth, part.model_vwc_pct_aux.to_numpy(float)),
            })
    by_seed = pd.DataFrame(all_rows).sort_values(["horizon", "seed"])
    by_seed["aux_minus_no_aux_rmse"] = by_seed.aux_rmse - by_seed.no_aux_rmse
    by_seed.to_csv(ARCH / "auxiliary_bilstm_by_seed.csv", index=False, float_format="%.10g")

    summary = {}
    for horizon in HORIZONS:
        part = by_seed[by_seed.horizon == horizon]
        summary[horizon] = {
            "n_origins": int(part.n_origins.iloc[0]),
            "persistence_rmse": float(part.persistence_rmse.mean()),
            "no_aux_rmse_mean": float(part.no_aux_rmse.mean()),
            "no_aux_rmse_sample_std": float(part.no_aux_rmse.std(ddof=1)),
            "aux_rmse_mean": float(part.aux_rmse.mean()),
            "aux_rmse_sample_std": float(part.aux_rmse.std(ddof=1)),
            "mean_difference_aux_minus_no_aux": float(part.aux_minus_no_aux_rmse.mean()),
            "seeds_aux_better": int((part.aux_minus_no_aux_rmse < 0).sum()),
            "blocks": {},
        }

        merged = None
        for seed in SEEDS:
            no_aux = pd.read_csv(ARCH / f"bilstm_no_aux_seed{seed}" / "architecture_predictions.csv")
            with_aux = pd.read_csv((ARCH if seed == 42 else ARCH / f"seed{seed}") / "architecture_predictions.csv")
            no_aux = no_aux[(no_aux.variant == "bilstm_residual") & (no_aux.horizon == horizon)]
            with_aux = with_aux[(with_aux.variant == "shared_bilstm_residual") & (with_aux.horizon == horizon)]
            no_aux = no_aux[KEY + ["model_vwc_pct"]].rename(columns={"model_vwc_pct": f"no_aux_{seed}"})
            with_aux = with_aux[KEY + ["target_vwc_pct", "current_vwc_pct", "model_vwc_pct"]].rename(columns={"model_vwc_pct": f"aux_{seed}"})
            seed_pair = with_aux.merge(no_aux, on=KEY, validate="one_to_one")
            if merged is None:
                merged = seed_pair
            else:
                merged = merged.merge(seed_pair[KEY + [f"aux_{seed}", f"no_aux_{seed}"]], on=KEY, validate="one_to_one")
        assert merged is not None and len(merged) == 3006
        cols = ["target_vwc_pct", "current_vwc_pct"] + [f"no_aux_{s}" for s in SEEDS] + [f"aux_{s}" for s in SEEDS]
        farms = [f[cols].to_numpy(float) for _, f in merged.sort_values(KEY).groupby("farm", sort=True)]
        assert len(farms) == 2 and all(len(farm) == 1503 for farm in farms)
        observed = compare(np.concatenate(farms))
        assert abs(observed - summary[horizon]["mean_difference_aux_minus_no_aux"]) < 1e-6
        for block in BLOCKS:
            rng = np.random.default_rng(20261114 + block)
            draws = []
            for _ in range(REPS):
                sampled = []
                for farm in farms:
                    starts = rng.integers(0, len(farm) - block + 1, size=int(np.ceil(len(farm) / block)))
                    ids = np.concatenate([np.arange(start, start + block) for start in starts])[:len(farm)]
                    sampled.append(farm[ids])
                draws.append(compare(np.concatenate(sampled)))
            summary[horizon]["blocks"][str(block)] = {
                "resamples": REPS,
                "percentile_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
    (ARCH / "auxiliary_bilstm_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for horizon, item in summary.items():
        print(f"{horizon}: persistence={item['persistence_rmse']:.3f}, no aux={item['no_aux_rmse_mean']:.3f}, "
              f"aux={item['aux_rmse_mean']:.3f}, aux-minus-noaux={item['mean_difference_aux_minus_no_aux']:+.3f}, "
              f"seeds aux better={item['seeds_aux_better']}/5, 24h CI={item['blocks']['96']['percentile_95_ci']}")


if __name__ == "__main__":
    main()
