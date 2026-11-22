"""Validation-selected site blend versus persistence on NOAA USCRN.

The 2025 outcomes are development evidence because they motivated this model.
The original three-station 2026 procedure was frozen before downloading those
outcomes. The later seven-station extension followed inspection of the first
three station outcomes and preceded download of the four added stations.
An exact-hour target filter corrects a subsequent row-offset audit finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from noaa_residual_baseline import ALPHAS, HORIZONS, ROOT, STATIONS, make_frame

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPANDED_CANDIDATES = STATIONS + (
    "IA_Des_Moines_17_E",
    "IL_Champaign_9_SW",
    "SC_Blackville_3_W",
    "TX_Austin_33_NW",
)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(mean_squared_error(y, p) ** 0.5)


def fit_model(x: np.ndarray, delta: np.ndarray, alpha: float):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, delta)
    return model


def main(final_year: int, expanded: bool = False) -> None:
    if expanded:
        audit = json.loads((ROOT / "availability_audit.json").read_text(encoding="utf-8"))
        available = {(r["station"], r["year"]): r["vwc5_available"] for r in audit}
        stations = tuple(
            s for s in EXPANDED_CANDIDATES
            if available.get((s, 2023), 0) >= 1000 and available.get((s, 2024), 0) >= 1000
        )
    else:
        stations = STATIONS
    data = pd.concat([make_frame(s) for s in stations], ignore_index=True)
    features = [c for c in data if c not in ("station", "origin_time", "valid_history") and not c.startswith("target_")]
    report = {"method": {
        "description": "A convex blend of persistence and pooled residual Ridge, with one weight per station and horizon.",
        "feature_columns": features,
        "station_encoding": (
            "The original AZ, CA and FL indicators are fixed; the four added stations share the all-zero reference encoding in Ridge and retain separate validation-selected blend weights."
            if expanded else "The three original stations have fixed AZ, CA and FL indicators."
        ),
        "alpha_grid": ALPHAS,
        "weight_grid": WEIGHTS,
        "selection": "Train Ridge on 2023; select pooled alpha and station weights by 2024 RMSE; refit on all years before the evaluation year.",
        "evaluation_year": final_year,
        "stations": stations,
        "expanded": expanded,
        "test_status": "2025 is exploratory development evidence. The original 3-station 2026 check was frozen before that year's download. The 2026 expanded check was specified after inspecting original-station outcomes but before downloading the four added stations.",
    }, "results": []}
    ledgers = []
    for h in HORIZONS:
        target = f"target_{h}"
        target_time = f"target_time_{h}"
        exact_horizon = (data[target_time] - data.origin_time).eq(pd.Timedelta(hours=h))
        # A target in the next year can have an origin before the fitting cutoff.
        # Restrict each annual partition to origins in that same year so no
        # training outcome is later than a validation or evaluation origin.
        same_year = data.origin_time.dt.year.eq(data[target_time].dt.year)
        subset = data.loc[data.valid_history & data[target].notna() & exact_horizon & same_year].copy()
        subset["target_year"] = subset[target_time].dt.year
        train = subset[subset.target_year == 2023]
        valid = subset[subset.target_year == 2024]
        dev = subset[subset.target_year == final_year]
        best_alpha, best_score = None, float("inf")
        for alpha in ALPHAS:
            model = fit_model(train[features].to_numpy(), (train[target] - train.vwc_current).to_numpy(), alpha)
            pred = valid.vwc_current.to_numpy() + model.predict(valid[features].to_numpy())
            score = rmse(valid[target].to_numpy(), pred)
            if score < best_score:
                best_alpha, best_score = alpha, score
        val_model = fit_model(train[features].to_numpy(), (train[target] - train.vwc_current).to_numpy(), best_alpha)
        val_residual = val_model.predict(valid[features].to_numpy())
        site_weights = {}
        site_selection = {}
        for station in stations:
            mask = valid.station.to_numpy() == station
            v_current = valid.vwc_current.to_numpy()[mask]
            v_truth = valid[target].to_numpy()[mask]
            v_residual = val_residual[mask]
            scores = [rmse(v_truth, v_current + w * v_residual) for w in WEIGHTS]
            best_idx = int(np.argmin(scores))
            site_weights[station] = WEIGHTS[best_idx]
            site_selection[station] = {
                "n_validation": int(mask.sum()),
                "weight": WEIGHTS[best_idx],
                "rmse_by_weight": dict(zip(map(str, WEIGHTS), [round(s, 5) for s in scores])),
            }
        fit = subset[subset.target_year.between(2023, final_year - 1)]
        model = fit_model(fit[features].to_numpy(), (fit[target] - fit.vwc_current).to_numpy(), best_alpha)
        residual = model.predict(dev[features].to_numpy())
        weight = dev.station.map(site_weights).to_numpy()
        current = dev.vwc_current.to_numpy()
        truth = dev[target].to_numpy()
        ridge = current + residual
        blend = current + weight * residual
        ledger = dev[["station", "origin_time", target_time, target]].copy()
        ledger["horizon_hours"] = h
        ledger["persistence_pred"] = current
        ledger["ridge_pred"] = ridge
        ledger["blend_pred"] = blend
        ledger["weight"] = weight
        ledgers.append(ledger.rename(columns={target_time: "target_time", target: "truth"}))
        for station in ("ALL",) + stations:
            mask = np.ones(len(dev), dtype=bool) if station == "ALL" else dev.station.to_numpy() == station
            if not mask.any():
                continue
            report["results"].append({
                "horizon_hours": h,
                "station": station,
                "n_evaluation_origins": int(mask.sum()),
                "selected_alpha": best_alpha,
                "site_weight": site_weights if station == "ALL" else site_weights[station],
                "validation_selection": site_selection if station == "ALL" else site_selection[station],
                "persistence_rmse": round(rmse(truth[mask], current[mask]), 5),
                "ridge_rmse": round(rmse(truth[mask], ridge[mask]), 5),
                "blend_rmse": round(rmse(truth[mask], blend[mask]), 5),
                "blend_minus_persistence_rmse": round(rmse(truth[mask], blend[mask]) - rmse(truth[mask], current[mask]), 5),
                "persistence_mae": round(float(mean_absolute_error(truth[mask], current[mask])), 5),
                "ridge_mae": round(float(mean_absolute_error(truth[mask], ridge[mask])), 5),
                "blend_mae": round(float(mean_absolute_error(truth[mask], blend[mask])), 5),
            })
    suffix = "2025_development" if final_year == 2025 else f"{final_year}_heldout"
    if expanded:
        suffix = f"{final_year}_expanded_heldout"
    (ROOT / f"site_blend_{suffix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.concat(ledgers).to_csv(ROOT / f"site_blend_{suffix}_ledger.csv", index=False)
    for row in report["results"]:
        print({k: v for k, v in row.items() if k != "validation_selection"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=(2025, 2026), default=2025)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    main(args.year, args.expanded)
