"""Describe ADC/VWC association without inferring an unavailable reference procedure.

This is a source-data diagnostic, not gravimetric validation. The repeated-ADC
check asks only whether the supplied target is a single-valued function of the
integer ADC field. A chronological linear fit describes association, not origin.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis_paths import SOURCE_CSV

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = SOURCE_CSV
OUTPUT = Path(__file__).resolve().parent / "baseline_audit" / "target_mapping_diagnostic.json"
ADC = "Raw_Capacitive_ADC"
VWC = "Volumetric_Moisture_Pct"


def repeated_adc(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(ADC)[VWC].agg(["size", "nunique", "min", "max"])
    repeated = grouped[grouped["size"] >= 5]
    ranges = repeated["max"] - repeated["min"]
    return {
        "n_rows": len(frame),
        "n_adc_levels": len(grouped),
        "n_adc_levels_with_at_least_5_rows": len(repeated),
        "n_repeated_levels_with_multiple_vwc_values": int(repeated["nunique"].gt(1).sum()),
        "median_vwc_range_within_repeated_adc_level": round(float(ranges.median()), 5),
        "max_vwc_range_within_repeated_adc_level": round(float(ranges.max()), 5),
        "adc_vwc_pearson_correlation": round(float(frame[ADC].corr(frame[VWC])), 8),
    }


def chronological_adc_fit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[[ADC]], train[VWC])
    pred = model.predict(test[[ADC]])
    actual = test[VWC].to_numpy()
    return {
        "n_train": len(train),
        "n_test": len(test),
        "test_rmse_vwc_percentage_points": round(float(np.sqrt(mean_squared_error(actual, pred))), 5),
        "test_mae_vwc_percentage_points": round(float(mean_absolute_error(actual, pred)), 5),
        "test_r2": round(float(r2_score(actual, pred)), 5),
    }


def main() -> None:
    df = pd.read_csv(SOURCE, parse_dates=["Timestamp"])
    if df[["Farm_ID", "Timestamp"]].duplicated().any() or df[[ADC, VWC]].isna().any().any():
        raise ValueError("Dataset key or required fields are missing")
    farms = []
    trains = []
    tests = []
    for farm, part in df.groupby("Farm_ID", sort=True):
        part = part.sort_values("Timestamp").reset_index(drop=True)
        n = len(part)
        train = part.iloc[: int(0.70 * n)]
        test = part.iloc[int(0.85 * n) :]
        trains.append(train)
        tests.append(test)
        farms.append({"farm": farm, **repeated_adc(part), "adc_only_chronological_fit": chronological_adc_fit(train, test)})
    report = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "question": "Is supplied VWC an exact single-valued function of the integer ADC column?",
        "scope": "supplied processed CSV only; no inference about gravimetric reference, lookup code or raw acquisition",
        "overall": repeated_adc(df),
        "farm_results": farms,
        "pooled_adc_only_chronological_fit": chronological_adc_fit(pd.concat(trains), pd.concat(tests)),
        "split": "per farm: first 70% train, final 15% test; middle 15% unused",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
