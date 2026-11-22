"""Descriptive error decomposition for the frozen 35-station NOAA ledger.

This analysis was designed after inspecting the overall RMSE/MAE tradeoff.
Bins use the *observed future change*, so they explain the scored outcomes;
they are not information available to a real-time forecaster or controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "external_noaa" / "zero_shot"
HORIZONS = (1, 6, 12, 24)
EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, np.inf)
LABELS = ("<0.5", "0.5-<1", "1-<2", "2-<4", ">=4")


def main() -> None:
    report = json.loads((ROOT / "zero_shot_2026.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(ROOT / "zero_shot_2026_ledger.csv.gz")
    result = {
        "status": "post hoc descriptive decomposition after inspecting pooled errors",
        "conditioning": "absolute observed target change from current VWC; unavailable at forecast origin",
        "units": "VWC percentage points",
        "bin_edges": [None if not np.isfinite(x) else x for x in EDGES],
        "bin_labels": LABELS,
        "results": [],
    }
    csv_rows = []
    for h in HORIZONS:
        eligible = {row["station"] for row in report["station_results"]
                    if row["horizon_hours"] == h and row["eligible"]}
        part = ledger.loc[ledger.horizon_hours.eq(h) & ledger.station.isin(eligible)].copy()
        pooled = next(r for r in report["pooled_results"] if r["horizon_hours"] == h)
        if len(part) != pooled["n_origins"]:
            raise ValueError(f"Frozen origin count changed at {h} hours")
        part["abs_change"] = (part.truth - part.persistence_pred).abs()
        part["p_abs_error"] = part.abs_change
        part["r_abs_error"] = (part.truth - part.ridge_pred).abs()
        part["p_sq_error"] = part.p_abs_error ** 2
        part["r_sq_error"] = part.r_abs_error ** 2
        part["change_bin"] = pd.cut(part.abs_change, bins=EDGES, labels=LABELS,
                                    right=False, include_lowest=True)
        if part.change_bin.isna().any():
            raise ValueError(f"Unbinned target change at {h} hours")
        rows = []
        for label in LABELS:
            block = part.loc[part.change_bin.eq(label)]
            n = len(block)
            if not n:
                raise ValueError(f"Empty {label} change bin at {h} hours")
            p_rmse = float(np.sqrt(block.p_sq_error.mean()))
            r_rmse = float(np.sqrt(block.r_sq_error.mean()))
            p_mae = float(block.p_abs_error.mean())
            r_mae = float(block.r_abs_error.mean())
            row = {
                "horizon_hours": h,
                "abs_observed_change_bin": label,
                "n_origins": n,
                "fraction_origins": round(n / len(part), 6),
                "persistence_rmse": round(p_rmse, 5),
                "ridge_rmse": round(r_rmse, 5),
                "ridge_minus_persistence_rmse": round(r_rmse - p_rmse, 5),
                "persistence_mae": round(p_mae, 5),
                "ridge_mae": round(r_mae, 5),
                "ridge_minus_persistence_mae": round(r_mae - p_mae, 5),
                "contribution_to_pooled_mse_difference": round(
                    float((block.r_sq_error - block.p_sq_error).sum() / len(part)), 6
                ),
                "contribution_to_pooled_mae_difference": round(
                    float((block.r_abs_error - block.p_abs_error).sum() / len(part)), 6
                ),
            }
            rows.append(row)
            csv_rows.append(row)
        # Reconstruct pooled squared/absolute loss differences without using
        # rounded bin summaries. All origins must appear in exactly one bin.
        if sum(row["n_origins"] for row in rows) != len(part):
            raise ValueError(f"Bin counts do not reconstruct {h}-hour pool")
        if abs(float(np.sqrt(part.r_sq_error.mean())) - pooled["ridge_rmse"]) > 0.00002:
            raise ValueError(f"Ridge RMSE differs from frozen pooled result at {h} hours")
        if abs(float(np.sqrt(part.p_sq_error.mean())) - pooled["persistence_rmse"]) > 0.00002:
            raise ValueError(f"Persistence RMSE differs from frozen pooled result at {h} hours")
        if abs(float(part.r_abs_error.mean()) - pooled["ridge_mae"]) > 0.00002:
            raise ValueError(f"Ridge MAE differs from frozen pooled result at {h} hours")
        if abs(float(part.p_abs_error.mean()) - pooled["persistence_mae"]) > 0.00002:
            raise ValueError(f"Persistence MAE differs from frozen pooled result at {h} hours")
        result["results"].extend(rows)
    (ROOT / "error_regime_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(ROOT / "error_regime_audit.csv", index=False)
    for row in result["results"]:
        if row["horizon_hours"] == 6:
            print(row)
    print("Saved", ROOT / "error_regime_audit.json")


if __name__ == "__main__":
    main()
