"""Describe matched rolling-block forecast errors with and without auxiliary loss."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ARCH = Path(__file__).resolve().parent / "architecture_benchmark"


def main() -> None:
    with_aux = pd.read_csv(ARCH / "rolling_backtest" / "rolling_metrics.csv")
    no_aux = pd.read_csv(ARCH / "rolling_backtest_no_aux" / "rolling_metrics.csv")
    keys = ["fold", "seed", "horizon"]
    pair = with_aux.merge(no_aux, on=keys, suffixes=("_aux", "_no_aux"), validate="one_to_one")
    assert len(pair) == 4 * 5 * 4
    assert (pair.test_origins_aux == pair.test_origins_no_aux).all()
    assert (pair.persistence_rmse_aux - pair.persistence_rmse_no_aux).abs().max() < 1e-6
    pair["aux_minus_no_aux_rmse"] = pair.model_rmse_aux - pair.model_rmse_no_aux
    pair.to_csv(ARCH / "rolling_auxiliary_comparison.csv", index=False, float_format="%.10g")
    summary = {}
    for horizon, part in pair.groupby("horizon"):
        summary[horizon] = {
            "fold_seed_pairs": len(part),
            "mean_aux_minus_no_aux_rmse": float(part.aux_minus_no_aux_rmse.mean()),
            "pairs_aux_better": int((part.aux_minus_no_aux_rmse < 0).sum()),
            "fold_mean_aux_minus_no_aux": {
                str(fold): float(fold_part.aux_minus_no_aux_rmse.mean())
                for fold, fold_part in part.groupby("fold")
            },
        }
    (ARCH / "rolling_auxiliary_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
