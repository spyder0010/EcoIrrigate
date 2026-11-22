"""Post hoc station-mix check for the seven-station 2026 NOAA blend.

Resample entire stations, preserving paired prediction errors and all observed
hours within each station. This is a descriptive sensitivity calculation with
only seven selected stations, not a population or future-year confidence
interval. The original 72-hour block intervals answer a different, conditional
within-station question.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "external_noaa"
HORIZONS = (1, 6, 12, 24)
SEED = 20261115
REPS = 10_000


def pooled_differences(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = rows[..., 0].sum(axis=-1)
    p_sse, b_sse = rows[..., 1].sum(axis=-1), rows[..., 2].sum(axis=-1)
    p_abs, b_abs = rows[..., 3].sum(axis=-1), rows[..., 4].sum(axis=-1)
    return np.sqrt(b_sse / counts) - np.sqrt(p_sse / counts), (b_abs - p_abs) / counts


def main() -> None:
    report = json.loads((ROOT / "site_blend_2026_expanded_heldout.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(ROOT / "site_blend_2026_expanded_heldout_ledger.csv")
    rng = np.random.default_rng(SEED)
    output = {
        "status": "post hoc; calculated after inspecting the seven-station 2026 results",
        "resampling_unit": "whole station, with paired errors and all available hours retained",
        "interpretation": "selected-station-mix sensitivity, not a population or future-year confidence interval",
        "random_seed": SEED,
        "bootstrap_repetitions": REPS,
        "results": [],
    }
    for horizon in HORIZONS:
        site_rows = [r for r in report["results"] if r["horizon_hours"] == horizon and r["station"] != "ALL"]
        sites = sorted(r["station"] for r in site_rows)
        if len(sites) != 7:
            raise ValueError(f"Expected seven stations at {horizon} h")
        frame = ledger.loc[ledger.horizon_hours.eq(horizon)].copy()
        if sorted(frame.station.unique()) != sites:
            raise ValueError(f"Ledger station mismatch at {horizon} h")
        frame["p_sq"] = (frame.truth - frame.persistence_pred) ** 2
        frame["b_sq"] = (frame.truth - frame.blend_pred) ** 2
        frame["p_abs"] = (frame.truth - frame.persistence_pred).abs()
        frame["b_abs"] = (frame.truth - frame.blend_pred).abs()
        grouped = frame.groupby("station", sort=True).agg(
            n=("truth", "size"), p_sse=("p_sq", "sum"), b_sse=("b_sq", "sum"),
            p_abs=("p_abs", "sum"), b_abs=("b_abs", "sum"),
        )
        rows = grouped.to_numpy(dtype=float)
        pooled = next(r for r in report["results"] if r["horizon_hours"] == horizon and r["station"] == "ALL")
        observed_rmse, observed_mae = pooled_differences(rows)
        if int(rows[:, 0].sum()) != pooled["n_evaluation_origins"]:
            raise ValueError(f"Origin count mismatch at {horizon} h")
        if abs(observed_rmse - (pooled["blend_rmse"] - pooled["persistence_rmse"])) > 0.00002:
            raise ValueError(f"RMSE mismatch at {horizon} h")
        if abs(observed_mae - (pooled["blend_mae"] - pooled["persistence_mae"])) > 0.00002:
            raise ValueError(f"MAE mismatch at {horizon} h")
        draws = rng.integers(0, len(rows), size=(REPS, len(rows)))
        boot_rmse, boot_mae = pooled_differences(rows[draws])
        loo_rmse = np.array([pooled_differences(np.delete(rows, i, axis=0))[0] for i in range(len(rows))])
        loo_mae = np.array([pooled_differences(np.delete(rows, i, axis=0))[1] for i in range(len(rows))])
        output["results"].append({
            "horizon_hours": horizon,
            "n_stations": len(sites),
            "n_origins": int(rows[:, 0].sum()),
            "stations": sites,
            "pooled_blend_minus_persistence_rmse": round(float(observed_rmse), 5),
            "site_wins_rmse": int(sum(r["blend_rmse"] < r["persistence_rmse"] for r in site_rows)),
            "station_bootstrap_rmse_2p5_97p5": [round(float(v), 5) for v in np.quantile(boot_rmse, [0.025, 0.975])],
            "leave_one_station_out_rmse_range": [round(float(loo_rmse.min()), 5), round(float(loo_rmse.max()), 5)],
            "pooled_blend_minus_persistence_mae": round(float(observed_mae), 5),
            "site_wins_mae": int(sum(r["blend_mae"] < r["persistence_mae"] for r in site_rows)),
            "station_bootstrap_mae_2p5_97p5": [round(float(v), 5) for v in np.quantile(boot_mae, [0.025, 0.975])],
            "leave_one_station_out_mae_range": [round(float(loo_mae.min()), 5), round(float(loo_mae.max()), 5)],
        })
    path = ROOT / "seven_station_mix_sensitivity.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for row in output["results"]:
        print(row)


if __name__ == "__main__":
    main()
