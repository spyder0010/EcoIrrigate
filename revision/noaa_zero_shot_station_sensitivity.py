"""Exploratory station-cluster sensitivity for the frozen NOAA transport check.

This analysis was designed after the 35-station outcomes were inspected. Whole
stations are resampled, preserving paired forecast errors and each station's
entire observed time series. The interval describes sensitivity to the chosen
station mix, not uncertainty over all USCRN stations or future years.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "external_noaa" / "zero_shot"
SEED = 42
REPS = 10_000
HORIZONS = (1, 6, 12, 24)
PREVIOUSLY_AUDITED = "AL_Brewton_3_NNE"


def score(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return model-minus-persistence RMSE and MAE from sufficient statistics."""
    counts = rows[..., 0].sum(axis=-1)
    p_sse = rows[..., 1].sum(axis=-1)
    r_sse = rows[..., 2].sum(axis=-1)
    p_abs = rows[..., 3].sum(axis=-1)
    r_abs = rows[..., 4].sum(axis=-1)
    return np.sqrt(r_sse / counts) - np.sqrt(p_sse / counts), (r_abs - p_abs) / counts


def main() -> None:
    report = json.loads((ROOT / "zero_shot_2026.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(ROOT / "zero_shot_2026_ledger.csv.gz")
    rng = np.random.default_rng(SEED)
    output = {
        "status": "exploratory; specified after seeing the 35-station outcomes",
        "resampling_unit": "whole station, with paired errors and all available 2026 hours retained",
        "interpretation": "sensitivity to selected station mix; not a population or future-year confidence interval",
        "random_seed": SEED,
        "bootstrap_repetitions": REPS,
        "results": [],
    }
    for horizon in HORIZONS:
        eligible = sorted(
            row["station"] for row in report["station_results"]
            if row["horizon_hours"] == horizon and row["eligible"]
        )
        frame = ledger.loc[
            ledger.horizon_hours.eq(horizon) & ledger.station.isin(eligible)
        ].copy()
        frame["p_sq"] = (frame.truth - frame.persistence_pred) ** 2
        frame["r_sq"] = (frame.truth - frame.ridge_pred) ** 2
        frame["p_abs"] = (frame.truth - frame.persistence_pred).abs()
        frame["r_abs"] = (frame.truth - frame.ridge_pred).abs()
        grouped = frame.groupby("station", sort=True).agg(
            n=("truth", "size"), p_sse=("p_sq", "sum"), r_sse=("r_sq", "sum"),
            p_abs=("p_abs", "sum"), r_abs=("r_abs", "sum"),
        )
        if list(grouped.index) != eligible or len(grouped) != 35:
            raise ValueError(f"Unexpected eligible cohort at {horizon} h")
        rows = grouped.to_numpy(dtype=float)
        pooled = report["pooled_results"][HORIZONS.index(horizon)]
        if int(rows[:, 0].sum()) != pooled["n_origins"]:
            raise ValueError(f"Origin count mismatch at {horizon} h")
        observed_rmse, observed_mae = score(rows)
        expected_rmse = pooled["ridge_rmse"] - pooled["persistence_rmse"]
        expected_mae = pooled["ridge_mae"] - pooled["persistence_mae"]
        if abs(observed_rmse - expected_rmse) > 0.00002 or abs(observed_mae - expected_mae) > 0.00002:
            raise ValueError(f"Pooled metric mismatch at {horizon} h")
        draws = rng.integers(0, len(rows), size=(REPS, len(rows)))
        boot_rmse, boot_mae = score(rows[draws])
        loo_rmse = np.array([score(np.delete(rows, i, axis=0))[0] for i in range(len(rows))])
        loo_mae = np.array([score(np.delete(rows, i, axis=0))[1] for i in range(len(rows))])
        audited_index = eligible.index(PREVIOUSLY_AUDITED)
        output["results"].append({
            "horizon_hours": horizon,
            "n_stations": len(rows),
            "n_origins": int(rows[:, 0].sum()),
            "pooled_ridge_minus_persistence_rmse": round(float(observed_rmse), 5),
            "station_bootstrap_rmse_2p5_97p5": [round(float(v), 5) for v in np.quantile(boot_rmse, [0.025, 0.975])],
            "leave_one_station_out_rmse_range": [round(float(loo_rmse.min()), 5), round(float(loo_rmse.max()), 5)],
            "leave_one_station_out_rmse_sign_reversals": int(np.sum(np.sign(loo_rmse) != np.sign(observed_rmse))),
            "leave_one_station_out_rmse_sign_reversal_stations": [
                station for station, value in zip(eligible, loo_rmse)
                if np.sign(value) != np.sign(observed_rmse)
            ],
            "pooled_ridge_minus_persistence_mae": round(float(observed_mae), 5),
            "excluding_previously_audited_station": {
                "station": PREVIOUSLY_AUDITED,
                "n_origins": int(rows[:, 0].sum() - rows[audited_index, 0]),
                "ridge_minus_persistence_rmse": round(float(loo_rmse[audited_index]), 5),
                "ridge_minus_persistence_mae": round(float(loo_mae[audited_index]), 5),
            },
            "station_bootstrap_mae_2p5_97p5": [round(float(v), 5) for v in np.quantile(boot_mae, [0.025, 0.975])],
            "leave_one_station_out_mae_range": [round(float(loo_mae.min()), 5), round(float(loo_mae.max()), 5)],
            "leave_one_station_out_mae_sign_reversals": int(np.sum(np.sign(loo_mae) != np.sign(observed_mae))),
        })
    path = ROOT / "station_mix_sensitivity.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for row in output["results"]:
        print(row)
    print("Saved", path)


if __name__ == "__main__":
    main()
