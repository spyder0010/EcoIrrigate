"""Reconstruct the GN1 result from the workbook XML and forecast ledger."""
from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "external_calibration" / "strypsteen"
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_raw():
    path = ROOT / "Field experiment GN1.xlsx"
    hashes = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    expected = next(e["sha256"] for e in hashes["files"] if e["name"] == path.name)
    assert sha256(path.read_bytes()).hexdigest() == expected
    with ZipFile(path) as workbook:
        root = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    values = {}
    for row in root.findall(".//x:row", NS):
        if int(row.get("r")) < 9:
            continue
        columns = {}
        for cell in row.findall("x:c", NS):
            value = cell.find("x:v", NS)
            if value is not None and cell.get("t", "n") == "n":
                column = re.match(r"[A-Z]+", cell.get("r")).group()
                columns[column] = float(value.text)
        hour = datetime(*(int(columns[f]) for f in ("A", "B", "C", "D")))
        assert hour not in values
        values[hour] = (columns.get("F", np.nan), columns.get("H", np.nan), columns.get("J", np.nan))
    return values


def mse_difference(values):
    ridge, recent, target = values.T
    return ((ridge - target) ** 2 - (recent - target) ** 2).mean()


def main():
    raw = parse_raw()
    report = json.loads((ROOT / "published_calibration_check.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(ROOT / "published_calibration_forecasts.csv", parse_dates=["origin", "target_time"])
    assert len(raw) == report["source_hour_rows"] == 962
    assert sum(np.isfinite(x[-1]) for x in raw.values()) == report["wet_reference_rows"] == 68
    cutoff = sorted(raw)[int(.8 * len(raw))]
    assert str(cutoff) == report["test_cutoff"]
    assert not ledger.duplicated(["probe", "origin", "horizon"]).any()
    assert set(ledger.probe) == set(report["probes"]) == {"probe_2", "probe_4"}
    total_pairs = 0
    for j, probe in enumerate(("probe_2", "probe_4")):
        valid = {stamp: -10 * np.log((values[j] - 170.7) / 296.8)
                 for stamp, values in raw.items() if 185 <= values[j] <= 467.5}
        available = {t for t in valid if all(t - timedelta(hours=lag) in valid for lag in range(13))}
        summary = report["probes"][probe]
        low, high = summary["scored_ucl_range"]
        a = lambda x: -6.2 * np.log((170.7 + 296.8 * np.exp(-x / 10) - 180.2) / 364.3)
        knots = np.linspace(low, high, 9)
        def design(times):
            return np.array([[valid[t], *(valid[t] - valid[t - timedelta(hours=lag)]
                                          for lag in (1, 3, 6, 12))] for t in times])

        for h in (1, 6, 12, 24):
            step = timedelta(hours=h)
            expected = {t for t in available if t >= cutoff and t + step in valid}
            training = sorted(t for t in available if t + step < cutoff and t + step in valid)
            rows = ledger.loc[ledger.probe.eq(probe) & ledger.horizon.eq(h)]
            assert set(rows.origin) == expected
            assert (rows.target_time - rows.origin).eq(step).all()
            assert rows.n_fit.eq(len(training)).all()
            assert len(rows) == summary["by_horizon"][str(h)]["test_origins"]
            np.testing.assert_allclose(rows.persistence, [valid[t] for t in rows.origin], rtol=0, atol=1e-12)
            np.testing.assert_allclose(rows.target, [valid[t] for t in rows.target_time], rtol=0, atol=1e-12)
            x = design(training)
            means, scales = x.mean(axis=0), x.std(axis=0)
            z = (x - means) / np.where(scales > 0, scales, 1)
            change = np.array([valid[t + step] - valid[t] for t in training])
            coef = np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ (change - change.mean()))
            predictions = (design(rows.origin) - means) / np.where(scales > 0, scales, 1) @ coef
            predicted = np.clip(rows.persistence + predictions + change.mean(), 0,
                                -10 * np.log((185 - 170.7) / 296.8))
            np.testing.assert_allclose(rows.ridge, predicted, rtol=0, atol=1e-12)
            scores = rows[["ridge", "persistence", "target"]].to_numpy()
            smooth = a(scores)
            aligned = low + (smooth - a(low)) * (high - low) / (a(high) - a(low))
            interpolated = np.interp(scores, knots, summary["calibration_knots"])
            record = summary["by_horizon"][str(h)]
            keys = ("nominal_ucl_mse_delta", "published_sscl_mse_delta", "normalized_smooth_mse_delta",
                    "normalized_interpolated_mse_delta")
            np.testing.assert_allclose([mse_difference(v) for v in (scores, smooth, aligned, interpolated)],
                                       [record[key] for key in keys], rtol=0, atol=1e-10)
            for bound in record["envelopes"].values():
                assert bound["smooth_curve_eligible"]
                lo, hi = bound["smooth_curve_outer_if_eligible"]
                assert lo <= mse_difference(aligned) <= hi
            total_pairs += len(rows)
    result = report["probes"]["probe_4"]["by_horizon"]["1"]
    assert result["nominal_ucl_mse_delta"] < 0 < result["published_sscl_mse_delta"]
    print(f"Verified GN1 workbook, {total_pairs:,} exact forecast pairs, Ridge refits, curve scores and bounds")


if __name__ == "__main__":
    main()
