"""Cross-check every saved neural prediction row against the supplied CSV.

This independent read of the prediction ledgers verifies timestamps, source
targets, current values and split boundaries. It does not re-evaluate model
weights or establish the physical provenance of supplied VWC.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis_paths import SOURCE_CSV

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = SOURCE_CSV
ARCH = ROOT / "architecture_benchmark"
HOURS = {"1h": 1, "6h": 6, "12h": 12, "24h": 24}
TOLERANCE = 5e-6  # Saved float32 predictions and targets.


def verify_rows(path: Path, raw: pd.DataFrame) -> dict:
    rows = pd.read_csv(path, parse_dates=["origin_timestamp", "target_timestamp"])
    if rows.empty:
        raise AssertionError(f"Empty prediction ledger: {path}")
    if rows[["farm", "horizon", "origin_timestamp", "target_timestamp"]].isna().any().any():
        raise AssertionError(f"Missing ledger key: {path}")
    elapsed = (rows.target_timestamp - rows.origin_timestamp).dt.total_seconds() / 3600
    expected = rows.horizon.map(HOURS)
    if expected.isna().any() or not elapsed.eq(expected).all():
        raise AssertionError(f"Incorrect forecast horizon: {path}")
    origin_raw = raw.rename(columns={"Timestamp": "origin_timestamp", "Volumetric_Moisture_Pct": "source_current"})
    target_raw = raw.rename(columns={"Timestamp": "target_timestamp", "Volumetric_Moisture_Pct": "source_target"})
    merged = rows.merge(origin_raw, left_on=["farm", "origin_timestamp"], right_on=["Farm_ID", "origin_timestamp"],
                        how="left", validate="many_to_one").drop(columns="Farm_ID")
    merged = merged.merge(target_raw, left_on=["farm", "target_timestamp"], right_on=["Farm_ID", "target_timestamp"],
                          how="left", validate="many_to_one").drop(columns="Farm_ID")
    if len(merged) != len(rows) or merged[["source_current", "source_target"]].isna().any().any():
        raise AssertionError(f"Unmatched source row: {path}")
    current_error = float((merged.current_vwc_pct - merged.source_current).abs().max())
    target_error = float((merged.target_vwc_pct - merged.source_target).abs().max())
    if current_error > TOLERANCE or target_error > TOLERANCE:
        raise AssertionError(f"VWC mismatch in {path}: current {current_error}, target {target_error}")

    rolling = "fold" in rows.columns
    if rolling:
        metrics_path = path.parent / "rolling_metrics.csv"
        bounds = pd.read_csv(metrics_path, parse_dates=["validation_end", "test_end"])
        bounds = bounds[["fold", "seed", "horizon", "validation_end", "test_end"]]
        merged = merged.merge(bounds, on=["fold", "seed", "horizon"], how="left", validate="many_to_one")
        if merged[["validation_end", "test_end"]].isna().any().any():
            raise AssertionError(f"Missing rolling split metadata: {path}")
        if not (merged.origin_timestamp.gt(merged.validation_end) & merged.target_timestamp.le(merged.test_end)).all():
            raise AssertionError(f"Rolling prediction outside stated test block: {path}")
        group_key = ["fold", "seed", "farm", "horizon", "origin_timestamp"]
    else:
        source_lengths = raw.groupby("Farm_ID", sort=True).size()
        starts = {}
        for farm, n in source_lengths.items():
            test_start_index = int(n * 0.70) + int(n * 0.15)
            starts[farm] = raw.loc[raw.Farm_ID.eq(farm), "Timestamp"].iloc[test_start_index]
        test_start = merged.farm.map(starts)
        if not (merged.origin_timestamp.ge(test_start) & merged.target_timestamp.ge(test_start)).all():
            raise AssertionError(f"Prediction outside final-test segment: {path}")
        group_key = ["variant", "farm", "horizon", "origin_timestamp"]
    if merged.duplicated(group_key).any():
        raise AssertionError(f"Duplicate prediction origin: {path}")
    if merged[["model_vwc_pct", "current_vwc_pct", "target_vwc_pct"]].isna().any().any():
        raise AssertionError(f"Missing saved prediction or target: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows_checked": len(rows),
        "kind": "rolling" if rolling else "final_test",
        "max_abs_current_source_difference": current_error,
        "max_abs_target_source_difference": target_error,
    }


def main() -> None:
    raw = pd.read_csv(SOURCE, usecols=["Timestamp", "Farm_ID", "Volumetric_Moisture_Pct"], parse_dates=["Timestamp"])
    if raw.duplicated(["Farm_ID", "Timestamp"]).any():
        raise AssertionError("Duplicate source farm/timestamp pair")
    paths = sorted(ARCH.rglob("architecture_predictions.csv"))
    paths += [ARCH / "rolling_backtest" / "rolling_predictions.csv",
              ARCH / "rolling_backtest_no_aux" / "rolling_predictions.csv"]
    if len(paths) != 16 or not all(path.is_file() for path in paths):
        raise AssertionError(f"Expected 16 saved neural ledgers, found {len(paths)}")
    checked = [verify_rows(path, raw) for path in paths]
    report = {
        "purpose": "saved neural ledger alignment with supplied source VWC and stated test segments",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "float32_tolerance_vwc_points": TOLERANCE,
        "ledger_count": len(checked),
        "prediction_rows_checked": sum(item["rows_checked"] for item in checked),
        "does_not_check": ["model checkpoint weights", "unfilled sensor logs", "independent gravimetric VWC"],
        "ledgers": checked,
    }
    output = ARCH / "neural_ledger_source_crosscheck.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {report['prediction_rows_checked']:,} saved predictions in {len(checked)} ledgers")
    print(f"Maximum current difference: {max(item['max_abs_current_source_difference'] for item in checked):.9f}")
    print(f"Maximum target difference: {max(item['max_abs_target_source_difference'] for item in checked):.9f}")


if __name__ == "__main__":
    main()
