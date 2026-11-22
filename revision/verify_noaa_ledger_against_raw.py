"""Cross-check 2026 forecast ledgers against archived USCRN text bytes.

This deliberately uses Python's standard library and a separate whitespace
parser, rather than importing the modelling or numerical verification code.
It checks source hashes, every scored forecast's origin and target VWC, and
the exact UTC horizon. It is an internal cross-check, not human review.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "external_noaa"
SM5_INDEX = 28  # 0-based SOIL_MOISTURE_5 in NOAA hourly02's 38 fields.
EXPECTED_FIELDS = 38
SEVEN_STATIONS = {
    "AZ_Tucson_11_W", "CA_Merced_23_WSW", "FL_Sebring_23_SSE",
    "IA_Des_Moines_17_E", "IL_Champaign_9_SW", "SC_Blackville_3_W",
    "TX_Austin_33_NW",
}


def source_entries() -> dict[str, list[Path]]:
    entries: dict[str, list[Path]] = {}
    for base, manifest_name in ((ROOT, "manifest.json"), (ROOT / "zero_shot", "manifest.json")):
        manifest = json.loads((base / manifest_name).read_text(encoding="utf-8"))
        for item in manifest["files"]:
            year = item.get("year", 2026)
            if (base == ROOT and year not in (2025, 2026)) or (base != ROOT and year != 2026):
                continue
            if base == ROOT and item["station"] not in SEVEN_STATIONS:
                continue  # Brewton was excluded from the seven-station analysis.
            path = base / item["file"]
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != item["sha256"]:
                raise AssertionError(f"Archived source hash mismatch: {path}")
            station = item["station"]
            entries.setdefault(station, []).append(path)
    return entries


def parse_source(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            fields = raw.decode("ascii").split()
            if not fields:
                continue  # NOAA's occasional NUL padding is not a measurement.
            if len(fields) != EXPECTED_FIELDS:
                raise AssertionError(f"{path.name}:{line_number}: {len(fields)} fields")
            date, time = fields[1], fields[2].zfill(4)
            stamp = f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:]}:00+00:00"
            if stamp in values:
                raise AssertionError(f"Duplicate source timestamp: {path.name} {stamp}")
            values[stamp] = float(fields[SM5_INDEX]) * 100.0
    return values


def verify_ledger(path: Path, sources: dict[str, dict[str, float]], label: str) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            station = row["station"]
            if station not in sources:
                raise AssertionError(f"{label} row {count}: station absent from raw archive: {station}")
            origin, target = row["origin_time"], row["target_time"]
            horizon = int(row["horizon_hours"])
            if datetime.fromisoformat(target) - datetime.fromisoformat(origin) != timedelta(hours=horizon):
                raise AssertionError(f"{label} row {count}: forecast horizon mismatch")
            raw = sources[station]
            if origin not in raw or target not in raw:
                raise AssertionError(f"{label} row {count}: timestamp absent from raw archive")
            for field, stamp in (("persistence_pred", origin), ("truth", target)):
                source_vwc = raw[stamp]
                if source_vwc < 0 or not math.isclose(float(row[field]), source_vwc, abs_tol=1e-8):
                    raise AssertionError(
                        f"{label} row {count}: {field}={row[field]} differs from raw "
                        f"{station} {stamp} VWC={source_vwc}"
                    )
    return count


def main() -> None:
    paths = source_entries()
    sources = {}
    for station, files in paths.items():
        values = {}
        for path in files:
            parsed = parse_source(path)
            if values.keys() & parsed.keys():
                raise AssertionError(f"Overlapping source timestamps: {station}")
            values.update(parsed)
        sources[station] = values
    ledgers = (
        (ROOT / "site_blend_2026_expanded_heldout_ledger.csv", "seven-station"),
        (ROOT / "zero_shot" / "zero_shot_2026_ledger.csv.gz", "new-station"),
    )
    for path, label in ledgers:
        count = verify_ledger(path, sources, label)
        print(f"{label}: {count:,} origin/target pairs matched to archived raw VWC")
    print(f"Source SHA-256 verified for {sum(map(len, paths.values()))} files across {len(sources)} stations")


if __name__ == "__main__":
    main()
