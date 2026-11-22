"""Paired site-wise moving-block intervals for mean six-hour BiLSTM skill.

The same sampled origin blocks are used for persistence and all five trained
seeds. The interval is conditional on these five seeds and this observed season.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent / "architecture_benchmark"
SEEDS = [42, 43, 44, 45, 46]
BLOCKS = [48, 96, 192]
REPS = 2000
KEY = ["farm", "horizon", "origin_timestamp", "target_timestamp"]


def aligned_data(variant: str) -> list[np.ndarray]:
    merged = None
    for seed in SEEDS:
        folder = (OUT if seed == 42 else OUT / f"seed{seed}") if variant == "shared_bilstm_residual" else OUT / f"bilstm_no_aux_seed{seed}"
        part = pd.read_csv(folder / "architecture_predictions.csv")
        part = part[(part.variant == variant) & (part.horizon == "6h")].copy()
        part = part[KEY + ["target_vwc_pct", "current_vwc_pct", "model_vwc_pct"]]
        part = part.rename(columns={"model_vwc_pct": f"model_{seed}"})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part[KEY + [f"model_{seed}"]], on=KEY, validate="one_to_one")
    assert merged is not None and len(merged) == 3006
    arrays = []
    for _, farm in merged.sort_values(KEY).groupby("farm", sort=True):
        assert len(farm) == 1503
        arrays.append(farm[["target_vwc_pct", "current_vwc_pct"] + [f"model_{seed}" for seed in SEEDS]].to_numpy(float))
    return arrays


def difference(parts: list[np.ndarray]) -> float:
    combined = np.concatenate(parts)
    truth = combined[:, 0]
    current = combined[:, 1]
    models = combined[:, 2:]
    baseline_rmse = np.sqrt(np.mean((current - truth) ** 2))
    mean_model_rmse = np.sqrt(np.mean((models - truth[:, None]) ** 2, axis=0)).mean()
    return float(mean_model_rmse - baseline_rmse)


def score(variant: str) -> dict:
    farms = aligned_data(variant)
    result = {
        "variant": variant,
        "estimand": "mean of five seed-specific RMSEs minus persistence RMSE on common six-hour origins",
        "scope": "conditional on five trained seeds, two site IDs and the supplied 2025 series",
        "n_origins": sum(len(farm) for farm in farms),
        "observed_difference_vwc_percentage_points": difference(farms),
        "blocks": {},
    }
    for block in BLOCKS:
        rng = np.random.default_rng(20261114 + block)
        draws = []
        for _ in range(REPS):
            samples = []
            for farm in farms:
                starts = rng.integers(0, len(farm) - block + 1, size=int(np.ceil(len(farm) / block)))
                ids = np.concatenate([np.arange(start, start + block) for start in starts])[:len(farm)]
                samples.append(farm[ids])
            draws.append(difference(samples))
        result["blocks"][str(block)] = {
            "resamples": REPS,
            "percentile_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return result


def main() -> None:
    for variant, name in [("shared_bilstm_residual", "multiseed_six_hour_blocks.json"),
                          ("bilstm_residual", "no_aux_six_hour_blocks.json")]:
        result = score(variant)
        (OUT / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
