"""Paired moving-block bootstrap for Ridge minus persistence RMSE.

Blocks are resampled independently within each farm. This is exploratory
uncertainty on a short, single-season held-out interval, not external validity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "baseline_audit"
BLOCK_LENGTHS = [48, 96, 192]
REPS = 2000


def rmse_diff(data: np.ndarray) -> float:
    truth, persistence, ridge = data.T
    return float(np.sqrt(np.mean((ridge - truth) ** 2)) - np.sqrt(np.mean((persistence - truth) ** 2)))


def sample_blocks(data: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(data)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    indices = np.concatenate([np.arange(start, start + block) for start in starts])[:n]
    return data[indices]


def main() -> None:
    ledger = pd.read_csv(OUT / "residual_ridge_predictions.csv")
    rng = np.random.default_rng(20261114)
    results = {}
    for horizon, part in ledger.groupby("horizon", sort=True):
        farm_arrays = []
        for _, farm_part in part.groupby("farm", sort=True):
            farm_arrays.append(farm_part[["target_vwc_pct", "persistence_vwc_pct", "residual_ridge_vwc_pct"]].to_numpy(dtype=float))
        observed = rmse_diff(np.concatenate(farm_arrays, axis=0))
        horizon_result = {"ridge_minus_persistence_rmse_vwc_percentage_points": observed, "n": int(len(part)), "blocks": {}}
        for block in BLOCK_LENGTHS:
            replicates = np.empty(REPS)
            for rep in range(REPS):
                sample = np.concatenate([sample_blocks(farm, block, rng) for farm in farm_arrays], axis=0)
                replicates[rep] = rmse_diff(sample)
            lo, hi = np.quantile(replicates, [0.025, 0.975])
            horizon_result["blocks"][str(block)] = {
                "resamples": REPS,
                "percentile_95_ci": [float(lo), float(hi)],
            }
        results[horizon] = horizon_result
    (OUT / "paired_block_bootstrap.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
