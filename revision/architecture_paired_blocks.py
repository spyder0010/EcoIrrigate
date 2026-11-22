"""Paired block-bootstrap RMSE differences for all common-origin variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "architecture_benchmark"
BLOCKS = [48, 96, 192]
REPS = 2000


def rmse_difference(data: np.ndarray) -> float:
    truth, current, forecast = data.T
    return float(np.sqrt(np.mean((forecast - truth) ** 2)) - np.sqrt(np.mean((current - truth) ** 2)))


def sample(data: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, len(data) - block + 1, size=int(np.ceil(len(data) / block)))
    ids = np.concatenate([np.arange(start, start + block) for start in starts])[:len(data)]
    return data[ids]


def main() -> None:
    ledger = pd.read_csv(OUT / "architecture_predictions.csv")
    expected = 2 * 1503 * 4 * ledger.variant.nunique()
    if len(ledger) != expected:
        raise ValueError(f"Incomplete common-origin ledger: {len(ledger)} vs {expected}")
    results = {}
    for (variant, horizon), part in ledger.groupby(["variant", "horizon"], sort=True):
        farm_arrays = [
            farm_part[["target_vwc_pct", "current_vwc_pct", "model_vwc_pct"]].to_numpy(dtype=float)
            for _, farm_part in part.groupby("farm", sort=True)
        ]
        observed = rmse_difference(np.concatenate(farm_arrays))
        item = {"n": len(part), "model_minus_persistence_rmse_vwc_percentage_points": observed, "blocks": {}}
        for block in BLOCKS:
            rng = np.random.default_rng(20261114 + block)
            replicates = np.array([
                rmse_difference(np.concatenate([sample(farm, block, rng) for farm in farm_arrays]))
                for _ in range(REPS)
            ])
            item["blocks"][str(block)] = {
                "resamples": REPS,
                "percentile_95_ci": np.quantile(replicates, [0.025, 0.975]).tolist(),
            }
        results[f"{variant}:{horizon}"] = item
    (OUT / "architecture_paired_blocks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for key, item in results.items():
        lo, hi = item["blocks"]["96"]["percentile_95_ci"]
        print(f"{key:31} dRMSE={item['model_minus_persistence_rmse_vwc_percentage_points']:+.3f} 24h CI=[{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
