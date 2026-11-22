"""Predefined external replication: past-only residual Ridge versus persistence.

This uses NOAA 5-cm VWC (m3/m3 converted to percentage points). It is not a
calibration reference for EcoIrrigate. 2023 trains, 2024 selects Ridge alpha,
and 2025 is evaluated once. Missing VWC is never filled; origins require 24
hours of observed VWC history and a measured target at the chosen horizon.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_noaa_external import ROOT, load_file

MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
STATIONS = ("AZ_Tucson_11_W", "CA_Merced_23_WSW", "FL_Sebring_23_SSE")
HORIZONS = (1, 6, 12, 24)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def make_frame(station: str) -> pd.DataFrame:
    entries = [e for e in MANIFEST["files"] if e["station"] == station]
    frame = pd.concat([load_file(ROOT / e["file"]) for e in entries], ignore_index=True)
    return make_frame_from_raw(frame, station)


def make_frame_from_raw(frame: pd.DataFrame, station: str) -> pd.DataFrame:
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    frame["vwc"] = frame.SOIL_MOISTURE_5 * 100
    frame["station"] = station
    frame["year"] = frame.timestamp_utc.dt.year
    # Every feature below is known no later than the forecast origin.
    out = {
        "vwc_current": frame.vwc,
        "hour_sin": np.sin(frame.timestamp_utc.dt.hour * 2 * np.pi / 24),
        "hour_cos": np.cos(frame.timestamp_utc.dt.hour * 2 * np.pi / 24),
        "day_sin": np.sin(frame.timestamp_utc.dt.dayofyear * 2 * np.pi / 365.25),
        "day_cos": np.cos(frame.timestamp_utc.dt.dayofyear * 2 * np.pi / 365.25),
        "temp_current": frame.T_HR_AVG,
        "rh_current": frame.RH_HR_AVG,
        "solar_current": frame.SOLARAD,
    }
    for lag in (1, 3, 6, 12, 24):
        out[f"vwc_lag_{lag}"] = frame.vwc.shift(lag)
        out[f"vwc_change_{lag}"] = frame.vwc - frame.vwc.shift(lag)
    for window in (3, 6, 24):
        out[f"vwc_mean_{window}"] = frame.vwc.rolling(window, min_periods=window).mean()
        out[f"rain_sum_{window}"] = frame.P_CALC.rolling(window, min_periods=window).sum()
        out[f"temp_mean_{window}"] = frame.T_HR_AVG.rolling(window, min_periods=window).mean()
    features = pd.DataFrame(out)
    features["station_AZ"] = float(station == "AZ_Tucson_11_W")
    features["station_CA"] = float(station == "CA_Merced_23_WSW")
    features["station_FL"] = float(station == "FL_Sebring_23_SSE")
    # Disallow origin histories that cross a missing VWC interval or timestamp gap.
    observed_history = frame.vwc.notna().rolling(25, min_periods=25).sum().eq(25)
    continuous_history = frame.timestamp_utc.diff().eq(pd.Timedelta(hours=1)).rolling(24, min_periods=24).sum().eq(24)
    for horizon in HORIZONS:
        features[f"target_{horizon}"] = frame.vwc.shift(-horizon)
        features[f"target_time_{horizon}"] = frame.timestamp_utc.shift(-horizon)
    features["station"] = station
    features["origin_time"] = frame.timestamp_utc
    features["valid_history"] = observed_history & continuous_history
    return features


def main() -> None:
    data = pd.concat([make_frame(s) for s in STATIONS], ignore_index=True)
    feature_columns = [
        c for c in data.columns
        if c not in {"station", "origin_time", "valid_history"}
        and not c.startswith("target_")
    ]
    output = {"design": {
        "train_target_year": 2023,
        "validation_target_year": 2024,
        "test_target_year": 2025,
        "stations": list(STATIONS),
        "horizons_hours": list(HORIZONS),
        "alpha_grid": list(ALPHAS),
        "feature_columns": feature_columns,
        "missing_target_policy": "exclude; never fill target",
        "missing_weather_policy": "training median from SimpleImputer",
        "history_policy": "25 current/past VWC values and 24 contiguous hourly intervals",
        "unit": "VWC percentage points",
    }, "results": []}
    predictions = []
    for horizon in HORIZONS:
        target = f"target_{horizon}"
        target_time = f"target_time_{horizon}"
        exact_horizon = (data[target_time] - data.origin_time).eq(pd.Timedelta(hours=horizon))
        same_year = data.origin_time.dt.year.eq(data[target_time].dt.year)
        subset = data.loc[data.valid_history & data[target].notna() & exact_horizon & same_year].copy()
        subset["target_year"] = subset[target_time].dt.year
        # 2025-12 targets spill into 2026 for the last few origins; omit them.
        subset = subset[subset.target_year.isin((2023, 2024, 2025))]
        train = subset[subset.target_year == 2023]
        valid = subset[subset.target_year == 2024]
        test = subset[subset.target_year == 2025]
        if min(len(train), len(valid), len(test)) == 0:
            raise ValueError("Empty chronological split")
        x_train = train[feature_columns].to_numpy(dtype=float)
        x_valid = valid[feature_columns].to_numpy(dtype=float)
        y_train = (train[target] - train.vwc_current).to_numpy(dtype=float)
        y_valid = valid[target].to_numpy(dtype=float)
        scores = []
        for alpha in ALPHAS:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, y_train)
            predicted_change = model.predict(x_valid)
            scores.append(float(mean_squared_error(y_valid, predicted_change) ** 0.5))
        selected = ALPHAS[int(np.argmin(scores))]
        fit = pd.concat([train, valid], ignore_index=True)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=selected))
        model.fit(fit[feature_columns].to_numpy(dtype=float), (fit[target] - fit.vwc_current).to_numpy(dtype=float))
        pred = test.vwc_current.to_numpy() + model.predict(test[feature_columns].to_numpy(dtype=float))
        truth = test[target].to_numpy(dtype=float)
        persist = test.vwc_current.to_numpy(dtype=float)
        ledger = test[["station", "origin_time", target_time, target, "vwc_current"]].copy()
        ledger["horizon_hours"] = horizon
        ledger["ridge_pred"] = pred
        ledger["persistence_pred"] = persist
        ledger = ledger.rename(columns={target_time: "target_time", target: "truth", "vwc_current": "current_vwc"})
        predictions.append(ledger)
        for station in ("ALL",) + STATIONS:
            mask = np.ones(len(test), dtype=bool) if station == "ALL" else test.station.to_numpy() == station
            if not mask.any():
                continue
            ridge_rmse = float(mean_squared_error(truth[mask], pred[mask]) ** 0.5)
            persist_rmse = float(mean_squared_error(truth[mask], persist[mask]) ** 0.5)
            output["results"].append({
                "horizon_hours": horizon,
                "station": station,
                "n_test_origins": int(mask.sum()),
                "selected_alpha": selected,
                "validation_rmse_by_alpha": dict(zip(map(str, ALPHAS), map(lambda x: round(x, 5), scores))) if station == "ALL" else None,
                "ridge_rmse": round(ridge_rmse, 5),
                "persistence_rmse": round(persist_rmse, 5),
                "ridge_minus_persistence_rmse": round(ridge_rmse - persist_rmse, 5),
                "ridge_mae": round(float(mean_absolute_error(truth[mask], pred[mask])), 5),
                "persistence_mae": round(float(mean_absolute_error(truth[mask], persist[mask])), 5),
            })
    (ROOT / "ridge_persistence_2025.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    pd.concat(predictions).to_csv(ROOT / "ridge_persistence_2025_ledger.csv", index=False)
    for row in output["results"]:
        print(row)


if __name__ == "__main__":
    main()
