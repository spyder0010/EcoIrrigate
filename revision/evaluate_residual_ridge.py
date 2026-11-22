"""Test whether a simple residual model beats persistence on supplied VWC.

Uses only origin-time features, chronological train/validation/test targets, and
the same held-out origin rows as audit_dataset_and_baselines.py. This is a
methodological experiment, not evidence of independent moisture measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis_paths import SOURCE_CSV

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = SOURCE_CSV
OUT = Path(__file__).resolve().parent / "baseline_audit"
HORIZONS = {"1h": 4, "6h": 24, "12h": 48, "24h": 96}
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def main() -> None:
    data = pd.read_csv(SOURCE, parse_dates=["Timestamp"])
    frames = []
    for farm_i, (farm, frame) in enumerate(data.groupby("Farm_ID", sort=True)):
        frame = frame.sort_values("Timestamp").reset_index(drop=True)
        y = frame.Volumetric_Moisture_Pct.to_numpy(dtype=float)
        # All values in the feature vector are observed at or before origin t.
        features = pd.DataFrame({
            "farm_index": farm_i,
            "current_vwc": y,
            "delta_1": y - np.roll(y, 1),
            "delta_4": y - np.roll(y, 4),
            "delta_24": y - np.roll(y, 24),
            "delta_96": y - np.roll(y, 96),
            "mean_prev_4": pd.Series(y).rolling(4).mean(),
            "mean_prev_24": pd.Series(y).rolling(24).mean(),
            "mean_prev_96": pd.Series(y).rolling(96).mean(),
            "board_temp": frame.Sensor_Board_Temperature_C,
            "atm_temp": frame.Atm_Temperature_C,
            "soil_temp_estimated": frame.Soil_Temperature_C,
            "pressure": frame.Atm_Pressure_inHg,
            "hour_sin": frame.Hour_sin,
            "hour_cos": frame.Hour_cos,
        })
        train_end = int(len(frame) * 0.70)
        val_end = train_end + int(len(frame) * 0.15)
        for name, h in HORIZONS.items():
            for split, start, end in [
                ("train", 96, train_end),
                ("validation", train_end, val_end),
                ("test", val_end, len(frame)),
            ]:
                for t in range(start, end - h):
                    row = features.iloc[t].to_dict()
                    row.update({
                        "farm": farm,
                        "horizon": name,
                        "split": split,
                        "origin_timestamp": frame.Timestamp.iloc[t],
                        "target_timestamp": frame.Timestamp.iloc[t + h],
                        "current_vwc_pct": y[t],
                        "target_vwc_pct": y[t + h],
                        "delta_target": y[t + h] - y[t],
                    })
                    frames.append(row)
    table = pd.DataFrame(frames)
    feature_cols = [
        "farm_index", "current_vwc", "delta_1", "delta_4", "delta_24",
        "delta_96", "mean_prev_4", "mean_prev_24", "mean_prev_96",
        "board_temp", "atm_temp", "soil_temp_estimated", "pressure",
        "hour_sin", "hour_cos",
    ]
    assert table[feature_cols].notna().all().all()
    outputs = []
    selected = {}
    for horizon in HORIZONS:
        part = table[table.horizon.eq(horizon)]
        train = part[part.split.eq("train")]
        val = part[part.split.eq("validation")]
        test = part[part.split.eq("test")].copy()
        candidates = []
        for alpha in ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[feature_cols], train.delta_target)
            val_prediction = val.current_vwc_pct.to_numpy() + model.predict(val[feature_cols])
            candidates.append((np.sqrt(mean_squared_error(val.target_vwc_pct, val_prediction)), alpha))
        _, chosen_alpha = min(candidates)
        # Refit with validation examples once alpha is selected; target rows stay
        # inside their own split and no test values enter fitting.
        fitting = part[part.split.isin(["train", "validation"])]
        model = make_pipeline(StandardScaler(), Ridge(alpha=chosen_alpha))
        model.fit(fitting[feature_cols], fitting.delta_target)
        test["residual_ridge_vwc_pct"] = test.current_vwc_pct.to_numpy() + model.predict(test[feature_cols])
        test["persistence_vwc_pct"] = test.current_vwc_pct
        selected[horizon] = {
            "chosen_alpha": chosen_alpha,
            "validation_rmse_vwc_percentage_points": float(min(candidates)[0]),
            "test_ridge_rmse_vwc_percentage_points": float(np.sqrt(mean_squared_error(test.target_vwc_pct, test.residual_ridge_vwc_pct))),
            "test_persistence_rmse_vwc_percentage_points": float(np.sqrt(mean_squared_error(test.target_vwc_pct, test.persistence_vwc_pct))),
            "test_ridge_r2": float(r2_score(test.target_vwc_pct, test.residual_ridge_vwc_pct)),
            "test_persistence_r2": float(r2_score(test.target_vwc_pct, test.persistence_vwc_pct)),
            "n_test": int(len(test)),
        }
        outputs.append(test[["farm", "horizon", "origin_timestamp", "target_timestamp", "target_vwc_pct", "persistence_vwc_pct", "residual_ridge_vwc_pct"]])
    ledger = pd.concat(outputs, ignore_index=True)
    baseline = pd.read_csv(OUT / "aligned_baseline_predictions.csv", parse_dates=["origin_timestamp", "target_timestamp"])
    matched = ledger.merge(baseline[["farm", "horizon", "origin_timestamp", "target_timestamp"]],
                           on=["farm", "horizon", "origin_timestamp", "target_timestamp"], validate="one_to_one")
    assert len(matched) == len(ledger) == len(baseline), "Test origins must match the locked baseline ledger"
    ledger.to_csv(OUT / "residual_ridge_predictions.csv", index=False, float_format="%.10g")
    (OUT / "residual_ridge_metrics.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
