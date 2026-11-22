"""Summarize five seeds for each shared architecture without test-seed selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

OUT = Path(__file__).resolve().parent / "architecture_benchmark"
VARIANTS = ["shared_gru_residual", "shared_tcn_residual", "shared_bilstm_residual"]
SEEDS = [42, 43, 44, 45, 46]


def rmse(truth, prediction) -> float:
    return float(np.sqrt(mean_squared_error(truth, prediction)))


def main() -> None:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            folder = OUT if seed == 42 else OUT / (f"seed{seed}" if variant == "shared_bilstm_residual" else f"seed{seed}_others")
            ledger = pd.read_csv(folder / "architecture_predictions.csv")
            ledger = ledger[ledger.variant.eq(variant)]
            if len(ledger) != 3006 * 4:
                raise ValueError(f"{variant} seed {seed}: incomplete ledger ({len(ledger)} rows)")
            for horizon, part in ledger.groupby("horizon", sort=True):
                for farm, subset in [("pooled", part), *list(part.groupby("farm", sort=True))]:
                    rows.append({
                        "variant": variant,
                        "seed": seed,
                        "horizon": horizon,
                        "farm": farm,
                        "n": len(subset),
                        "model_rmse": rmse(subset.target_vwc_pct, subset.model_vwc_pct),
                        "persistence_rmse": rmse(subset.target_vwc_pct, subset.current_vwc_pct),
                    })
    table = pd.DataFrame(rows)
    table["difference_model_minus_persistence"] = table.model_rmse - table.persistence_rmse
    table.to_csv(OUT / "shared_architecture_seed_by_site.csv", index=False, float_format="%.10g")
    summary = {}
    for (variant, horizon), part in table[table.farm.eq("pooled")].groupby(["variant", "horizon"], sort=True):
        d = part.difference_model_minus_persistence.to_numpy()
        summary[f"{variant}:{horizon}"] = {
            "seed_count": len(part),
            "test_origins_per_seed": int(part.n.iloc[0]),
            "persistence_rmse_vwc_percentage_points": float(part.persistence_rmse.iloc[0]),
            "model_rmse_mean_vwc_percentage_points": float(part.model_rmse.mean()),
            "model_rmse_sample_std_vwc_percentage_points": float(part.model_rmse.std(ddof=1)),
            "model_rmse_min_vwc_percentage_points": float(part.model_rmse.min()),
            "model_rmse_max_vwc_percentage_points": float(part.model_rmse.max()),
            "mean_difference_model_minus_persistence": float(d.mean()),
            "seeds_beating_persistence": int(np.sum(d < 0)),
        }
    (OUT / "shared_architecture_seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(table[table.farm.eq("pooled")].pivot(index=["seed", "horizon"], columns="variant", values="difference_model_minus_persistence").round(3).to_string())


if __name__ == "__main__":
    main()
