"""Audit the supplied series and score aligned, leakage-free forecast baselines.

This does not validate the origin of VWC or any field/irrigation claim.
Run from any directory: python revision/audit_dataset_and_baselines.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis_paths import SOURCE_CSV

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = SOURCE_CSV
OUT = Path(__file__).resolve().parent / "baseline_audit"
HORIZONS = {"1h": 4, "6h": 24, "12h": 48, "24h": 96}
STEP = pd.Timedelta(minutes=15)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "rmse_vwc_percentage_points": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_vwc_percentage_points": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, parse_dates=["Timestamp"])
    expected_columns = {"Timestamp", "Farm_ID", "Volumetric_Moisture_Pct", "Raw_Capacitive_ADC", "Sensor_Voltage_V"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Required columns missing: {sorted(missing_columns)}")
    if df.duplicated(["Farm_ID", "Timestamp"]).any():
        raise ValueError("Duplicate farm/timestamp pairs make origin alignment ambiguous")
    if df["Volumetric_Moisture_Pct"].isna().any():
        raise ValueError("Missing VWC; no imputation is performed in this audit")

    farm_frames = {}
    farms = []
    for farm, frame in df.groupby("Farm_ID", sort=True):
        frame = frame.sort_values("Timestamp").reset_index(drop=True)
        expected = pd.date_range(frame.Timestamp.iloc[0], frame.Timestamp.iloc[-1], freq="15min")
        absent = expected.difference(frame.Timestamp)
        if len(absent):
            raise ValueError(f"{farm}: {len(absent)} timestamps absent; cannot score an unfilled series")
        n = len(frame)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)
        frame["split"] = np.where(frame.index < train_end, "train", np.where(frame.index < val_end, "validation", "test"))
        farm_frames[farm] = frame
        farms.append({
            "farm": farm,
            "rows": n,
            "first_timestamp": str(frame.Timestamp.iloc[0]),
            "last_timestamp": str(frame.Timestamp.iloc[-1]),
            "expected_15min_rows": len(expected),
            "missing_timestamps_in_supplied_file": len(absent),
            "train_rows": train_end,
            "validation_rows": val_end - train_end,
            "test_rows": n - val_end,
            "test_start": str(frame.Timestamp.iloc[val_end]),
        })

    ledger = []
    for farm, frame in farm_frames.items():
        y = frame.Volumetric_Moisture_Pct.to_numpy(dtype=float)
        n = len(y)
        train_end = int(n * 0.70)
        test_start = train_end + int(n * 0.15)
        train_mean = float(y[:train_end].mean())
        for name, h in HORIZONS.items():
            # Forecast origin t includes an observed y[t]; every target t+h is in
            # the held-out test interval. History may reach into validation.
            for t in range(test_start, n - h):
                assert frame.Timestamp.iloc[t + h] - frame.Timestamp.iloc[t] == h * STEP
                ledger.append({
                    "farm": farm,
                    "horizon": name,
                    "horizon_steps": h,
                    "origin_timestamp": frame.Timestamp.iloc[t],
                    "target_timestamp": frame.Timestamp.iloc[t + h],
                    "origin_vwc_pct": y[t],
                    "target_vwc_pct": y[t + h],
                    "persistence_vwc_pct": y[t],
                    # Extrapolate the observed change over the preceding hour.
                    # This is a fixed, train-free comparator, not a tuned model.
                    "linear_1h_trend_vwc_pct": y[t] + (h / 4) * (y[t] - y[t - 4]),
                    "seasonal_24h_vwc_pct": y[t - 96],
                    "train_mean_vwc_pct": train_mean,
                })
    predictions = pd.DataFrame.from_records(ledger)
    if predictions.empty:
        raise ValueError("No held-out forecast origins")
    predictions.to_csv(OUT / "aligned_baseline_predictions.csv", index=False, float_format="%.10g")

    result_rows = []
    for (farm, horizon), part in predictions.groupby(["farm", "horizon"], sort=True):
        truth = part.target_vwc_pct.to_numpy()
        for model, column in [
            ("persistence", "persistence_vwc_pct"),
            ("linear_1h_trend", "linear_1h_trend_vwc_pct"),
            ("seasonal_24h", "seasonal_24h_vwc_pct"),
            ("train_mean", "train_mean_vwc_pct"),
        ]:
            result_rows.append({"farm": farm, "horizon": horizon, "model": model, **metrics(truth, part[column].to_numpy())})
    for horizon, part in predictions.groupby("horizon", sort=True):
        truth = part.target_vwc_pct.to_numpy()
        for model, column in [
            ("persistence", "persistence_vwc_pct"),
            ("linear_1h_trend", "linear_1h_trend_vwc_pct"),
            ("seasonal_24h", "seasonal_24h_vwc_pct"),
            ("train_mean", "train_mean_vwc_pct"),
        ]:
            result_rows.append({"farm": "pooled", "horizon": horizon, "model": model, **metrics(truth, part[column].to_numpy())})
    result_table = pd.DataFrame(result_rows).sort_values(["farm", "horizon", "model"])
    result_table.to_csv(OUT / "baseline_metrics.csv", index=False, float_format="%.10g")

    adc = df.Raw_Capacitive_ADC.to_numpy(dtype=float)
    volts = df.Sensor_Voltage_V.to_numpy(dtype=float)
    report = {
        "source": str(SOURCE),
        "source_sha256": digest(SOURCE),
        "total_rows": int(len(df)),
        "null_cells_in_supplied_file": int(df.isna().sum().sum()),
        "duplicate_farm_timestamps": int(df.duplicated(["Farm_ID", "Timestamp"]).sum()),
        "farms": farms,
        "adc_voltage_correlation": float(np.corrcoef(adc, volts)[0, 1]),
        "adc_voltage_3v3_conversion_mae_v": float(np.mean(np.abs(volts - adc * 3.3 / 1023))),
        "vwc_target_provenance": "not established by supplied CSV; independent per-record gravimetric target unavailable",
        "claimed_128_historical_gaps": "not verifiable without original unfilled logs or a missingness mask",
        "baseline_definition": "origin t observes VWC through t; target is VWC[t+h]; all targets held out in test segment; linear 1h trend extrapolates y[t]-y[t-4] by h/4; seasonal baseline uses t-96; train mean uses first 70% only",
        "metrics_scope": "descriptive scores on supplied VWC series; not a field or water-savings validation",
    }
    (OUT / "dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(result_table[result_table.farm.eq("pooled")].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
