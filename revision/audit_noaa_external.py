"""Audit availability of NOAA 5-cm VWC and past-only weather features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "external_noaa"
COLUMNS = (
    "WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE "
    "T_CALC T_HR_AVG T_MAX T_MIN P_CALC SOLARAD SOLARAD_FLAG "
    "SOLARAD_MAX SOLARAD_MAX_FLAG SOLARAD_MIN SOLARAD_MIN_FLAG "
    "SUR_TEMP_TYPE SUR_TEMP SUR_TEMP_FLAG SUR_TEMP_MAX SUR_TEMP_MAX_FLAG "
    "SUR_TEMP_MIN SUR_TEMP_MIN_FLAG RH_HR_AVG RH_HR_AVG_FLAG "
    "SOIL_MOISTURE_5 SOIL_MOISTURE_10 SOIL_MOISTURE_20 "
    "SOIL_MOISTURE_50 SOIL_MOISTURE_100 SOIL_TEMP_5 SOIL_TEMP_10 "
    "SOIL_TEMP_20 SOIL_TEMP_50 SOIL_TEMP_100"
).split()


def load_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, dtype={"UTC_TIME": str})
    frame["timestamp_utc"] = pd.to_datetime(
        frame.UTC_DATE.astype(str) + frame.UTC_TIME.str.zfill(4), format="%Y%m%d%H%M", utc=True
    )
    for field in ("SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "SOIL_MOISTURE_20", "T_HR_AVG", "P_CALC", "SOLARAD", "RH_HR_AVG"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame.loc[frame[field] < -90, field] = np.nan
    return frame


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    report = []
    for entry in manifest["files"]:
        frame = load_file(ROOT / entry["file"])
        valid = frame.SOIL_MOISTURE_5.notna()
        gaps = frame.timestamp_utc.diff().dropna().ne(pd.Timedelta(hours=1)).sum()
        longest = valid.groupby((valid != valid.shift()).cumsum()).sum().max()
        report.append(
            {
                "station": entry["station"],
                "year": entry["year"],
                "rows": len(frame),
                "timestamp_first": frame.timestamp_utc.iloc[0].isoformat(),
                "timestamp_last": frame.timestamp_utc.iloc[-1].isoformat(),
                "timestamp_gaps": int(gaps),
                "vwc5_available": int(valid.sum()),
                "vwc5_fraction": round(float(valid.mean()), 4),
                "vwc5_longest_run": int(longest),
                "vwc5_min": None if not valid.any() else round(float(frame.SOIL_MOISTURE_5.min()), 3),
                "vwc5_max": None if not valid.any() else round(float(frame.SOIL_MOISTURE_5.max()), 3),
                "temp_fraction": round(float(frame.T_HR_AVG.notna().mean()), 4),
                "precip_fraction": round(float(frame.P_CALC.notna().mean()), 4),
                "solar_fraction": round(float(frame.SOLARAD.notna().mean()), 4),
                "rh_fraction": round(float(frame.RH_HR_AVG.notna().mean()), 4),
            }
        )
    output = ROOT / "availability_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report:
        print(row)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
