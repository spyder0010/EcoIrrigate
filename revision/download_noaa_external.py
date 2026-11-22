"""Fetch a fixed, public NOAA USCRN hourly benchmark with file provenance.

The four stations were fixed by geography before examining their model scores.
This dataset is for a separate forecasting replication, not EcoIrrigate sensor
calibration or field-irrigation validation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent / "external_noaa"
BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
STATIONS = (
    "AL_Brewton_3_NNE",
    "AZ_Tucson_11_W",
    "CA_Merced_23_WSW",
    "FL_Sebring_23_SSE",
    "IA_Des_Moines_17_E",
    "IL_Champaign_9_SW",
    "SC_Blackville_3_W",
    "TX_Austin_33_NW",
)
YEARS = (2023, 2024, 2025, 2026)


def main() -> None:
    raw_dir = ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for year in YEARS:
        for station in STATIONS:
            name = f"CRNH0203-{year}-{station}.txt"
            url = f"{BASE}/{year}/{name}"
            destination = raw_dir / name
            if destination.exists():
                content = destination.read_bytes()
            else:
                print(f"Downloading {name}", flush=True)
                with urlopen(url, timeout=90) as response:
                    content = response.read()
                destination.write_bytes(content)
            manifest.append(
                {
                    "station": station,
                    "year": year,
                    "url": url,
                    "file": destination.relative_to(ROOT).as_posix(),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "rows": len(content.splitlines()),
                }
            )
    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "source": "NOAA USCRN hourly02",
                "source_page": "https://www.ncei.noaa.gov/access/crn/products.html",
                "format_documentation": f"{BASE}/readme.txt",
                "download_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "station_selection": "Initial cohort: Alabama, Arizona, California and Florida fixed before model-score inspection. Expanded cohort: Iowa, Illinois, South Carolina and Texas fixed after inspecting initial 2026 scores but before downloading expanded-station outcomes. See EXTERNAL_EVALUATION_PROTOCOL.md.",
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} files to {raw_dir}")


if __name__ == "__main__":
    main()
