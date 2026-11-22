"""Audit frozen NOAA hourly02 bytes against the published field layout.

The fixed-width offsets below are from NOAA hourly02/readme.txt (1-based
columns). This is an independent layout check on the whitespace parser used
by the forecast scripts. It does not validate NOAA's underlying sensors.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from audit_noaa_external import COLUMNS, ROOT, load_file


FIELDS = {
    # name: (zero-based whitespace token, zero-based fixed-width start, stop)
    "UTC_DATE": (1, 6, 14),
    "UTC_TIME": (2, 15, 19),
    "T_HR_AVG": (9, 65, 72),
    "P_CALC": (12, 89, 96),
    "SOLARAD": (13, 97, 103),
    "RH_HR_AVG": (26, 156, 161),
    "SOIL_MOISTURE_5": (28, 164, 171),
}
EXPECTED_COLUMNS = 38
READ_ME = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/readme.txt"


def entries() -> list[tuple[str, dict, Path]]:
    sources = []
    for cohort, directory in (("development", ROOT), ("unseen", ROOT / "zero_shot")):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            sources.append((cohort, entry, directory / entry["file"]))
    return sources


def main() -> None:
    output = {
        "documentation": READ_ME,
        "meaning": "NOAA hourly02 values cover the hour ending at the listed UTC time",
        "audited_fields": list(FIELDS),
        "expected_column_count": EXPECTED_COLUMNS,
        "files": [],
        "overlapping_station_years": [],
    }
    overlap = defaultdict(list)
    for cohort, entry, path in entries():
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise AssertionError(f"Manifest hash mismatch: {path}")
        lines = content.decode("ascii").splitlines()
        if len(lines) != entry["rows"]:
            raise AssertionError(f"Manifest row count mismatch: {path}")
        raw_missing = 0
        nul_padded_records = 0
        year_boundary_final_hour = False
        for number, line in enumerate(lines, start=1):
            # Some frozen NOAA files contain one 244-byte NUL run (the size of
            # an hourly record plus newline) immediately before the next valid
            # record. The pandas C parser ignores this prefix. We check the
            # exact byte pattern and retain it as a source gap, not a value.
            if line.startswith("\x00"):
                prefix = len(line) - len(line.lstrip("\x00"))
                if prefix != 244:
                    raise AssertionError(f"Unexpected NUL prefix: {path} line {number}: {prefix}")
                line = line[prefix:]
                nul_padded_records += 1
            if "\x00" in line:
                raise AssertionError(f"Embedded NUL inside NOAA record: {path} line {number}")
            tokens = line.split()
            if len(tokens) != EXPECTED_COLUMNS or len(line) != 243:
                raise AssertionError(f"Malformed 38-field NOAA row: {path} line {number}")
            for field, (index, start, stop) in FIELDS.items():
                if tokens[index] != line[start:stop].strip():
                    raise AssertionError(f"Fixed-width/whitespace disagreement: {path} line {number} {field}")
            file_year = int(entry.get("year", 2026))
            if int(tokens[1][:4]) != file_year:
                boundary_hour = (
                    number == len(lines)
                    and tokens[1] == f"{file_year + 1}0101"
                    and tokens[2] == "0000"
                )
                if not boundary_hour:
                    raise AssertionError(f"Unexpected source year: {path} line {number}")
                year_boundary_final_hour = True
            if tokens[2][2:] != "00":
                raise AssertionError(f"Non-hourly UTC timestamp: {path} line {number}")
            raw_missing += float(tokens[28]) < -90
        frame = load_file(path)
        if len(frame) != len(lines):
            raise AssertionError(f"Pandas dropped or added source rows: {path}")
        if len(COLUMNS) != EXPECTED_COLUMNS or frame.timestamp_utc.isna().any():
            raise AssertionError(f"Column or UTC parse failure: {path}")
        if not frame.timestamp_utc.is_monotonic_increasing or frame.timestamp_utc.duplicated().any():
            raise AssertionError(f"Unsorted or duplicate UTC timestamps: {path}")
        if int(frame.SOIL_MOISTURE_5.isna().sum()) != raw_missing:
            raise AssertionError(f"5-cm missing-value sentinel mismatch: {path}")
        observed = frame.SOIL_MOISTURE_5.dropna()
        outside_fraction = int(((observed < 0) | (observed > 1)).sum())
        if outside_fraction:
            raise AssertionError(f"5-cm VWC outside 0..1: {path} count {outside_fraction}")
        deltas = frame.timestamp_utc.diff().dropna()
        gap_deltas = deltas[deltas.ne(pd.Timedelta(hours=1))]
        gaps = len(gap_deltas)
        if gaps != nul_padded_records or not gap_deltas.eq(pd.Timedelta(hours=2)).all():
            raise AssertionError(f"NUL padding does not map to one missing UTC hour: {path}")
        row = {
            "cohort": cohort,
            "station": entry["station"],
            "year": int(entry.get("year", 2026)),
            "file": str(path.relative_to(ROOT.parent)),
            "sha256": entry["sha256"],
            "rows": len(lines),
            "first_utc": frame.timestamp_utc.iloc[0].isoformat(),
            "last_utc": frame.timestamp_utc.iloc[-1].isoformat(),
            "timestamp_gaps": gaps,
            "nul_padded_records": nul_padded_records,
            "year_boundary_final_hour": year_boundary_final_hour,
            "vwc5_observed": int(observed.size),
            "vwc5_missing": raw_missing,
        }
        output["files"].append(row)
        overlap[(row["station"], row["year"])].append(row)
    for (station, year), rows in sorted(overlap.items()):
        if len(rows) > 1:
            output["overlapping_station_years"].append({
                "station": station, "year": year,
                "same_bytes": len({r["sha256"] for r in rows}) == 1,
                "cohorts": [r["cohort"] for r in rows],
                "rows": [r["rows"] for r in rows],
            })
    output["summary"] = {
        "n_file_copies": len(output["files"]),
        "n_distinct_station_years": len(overlap),
        "n_rows": sum(row["rows"] for row in output["files"]),
        "n_vwc5_observed": sum(row["vwc5_observed"] for row in output["files"]),
        "files_with_timestamp_gaps": sum(row["timestamp_gaps"] > 0 for row in output["files"]),
        "files_with_nul_padding": sum(row["nul_padded_records"] > 0 for row in output["files"]),
        "nul_padded_record_count": sum(row["nul_padded_records"] for row in output["files"]),
        "files_with_next_year_midnight_final_hour": sum(row["year_boundary_final_hour"] for row in output["files"]),
        "files_with_no_vwc5": sum(row["vwc5_observed"] == 0 for row in output["files"]),
    }
    destination = ROOT / "source_format_audit.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print("Overlaps:", output["overlapping_station_years"])
    print("Saved", destination)


if __name__ == "__main__":
    main()
