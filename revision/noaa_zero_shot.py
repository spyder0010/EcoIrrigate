"""Frozen 2026 transport check at 36 previously unseen USCRN stations.

Read NOAA_ZERO_SHOT_PROTOCOL.md before running. The selected stations are
defined by filename, not by their 2026 measurements or model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from audit_noaa_external import load_file
from noaa_residual_baseline import ALPHAS, HORIZONS, ROOT, make_frame, make_frame_from_raw
from noaa_site_blend import EXPANDED_CANDIDATES, fit_model, rmse


ZERO = ROOT / "zero_shot"
BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2026"
EXCLUDED_STATES = {"AK", "HI", "ON", "AZ", "CA", "FL", "IA", "IL", "SC", "TX"}
EXPECTED = (
    "AL_Brewton_3_NNE", "AR_Batesville_8_WNW", "CO_Boulder_14_W",
    "GA_Brunswick_23_S", "ID_Arco_17_SW", "IN_Bedford_5_WNW",
    "KS_Manhattan_6_SSW", "KY_Bowling_Green_21_NNE", "LA_Lafayette_13_SE",
    "MA_Blue_Hill_0_W", "ME_Limestone_4_NNW", "MI_Chatham_1_SE",
    "MN_Goodridge_12_NNW", "MO_Chillicothe_22_ENE", "MS_Holly_Springs_4_N",
    "MT_Dillon_18_WSW", "NC_Asheville_13_S", "ND_Jamestown_38_WSW",
    "NE_Harrison_20_SSE", "NH_Durham_2_N", "NM_Las_Cruces_20_N",
    "NV_Baker_5_W", "NY_Ithaca_13_E", "OH_Wooster_3_SSE",
    "OK_Goodwell_2_E", "OR_Coos_Bay_8_SW", "PA_Avondale_2_N",
    "RI_Kingston_1_NW", "SD_Aberdeen_35_WNW", "TN_Crossville_7_NW",
    "UT_Brigham_City_28_WNW", "VA_Cape_Charles_5_ENE",
    "WA_Darrington_21_NNE", "WI_Necedah_5_WNW", "WV_Elkins_21_ENE",
    "WY_Lander_11_SSE",
)
TRAIN_STATIONS = (
    "AZ_Tucson_11_W", "CA_Merced_23_WSW", "FL_Sebring_23_SSE",
    "IA_Des_Moines_17_E", "IL_Champaign_9_SW", "SC_Blackville_3_W",
    "TX_Austin_33_NW",
)
MIN_ORIGINS = 1000


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_from_catalog(html: bytes) -> tuple[str, ...]:
    names = sorted(set(re.findall(r"CRNH0203-2026-([A-Z]{2}_[^\"<>/]+)\.txt", html.decode("utf-8"))))
    states = sorted({name[:2] for name in names if name[:2] not in EXCLUDED_STATES})
    chosen = tuple(next(name for name in names if name.startswith(state + "_")) for state in states)
    if chosen != EXPECTED:
        raise ValueError("Current directory listing no longer matches the frozen 36-station cohort")
    return chosen


def download() -> dict:
    ZERO.mkdir(parents=True, exist_ok=True)
    raw = ZERO / "raw"
    raw.mkdir(exist_ok=True)
    catalog_path = ZERO / "catalog_2026.html"
    if catalog_path.exists():
        html = catalog_path.read_bytes()
    else:
        with urlopen(BASE + "/", timeout=90) as response:
            html = response.read()
        catalog_path.write_bytes(html)
    chosen = choose_from_catalog(html)
    manifest_path = ZERO / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    old_entries = {r["station"]: r for r in old.get("files", [])}
    files = []
    for station in chosen:
        name = f"CRNH0203-2026-{station}.txt"
        url = f"{BASE}/{name}"
        path = raw / name
        if path.exists():
            content = path.read_bytes()
            retrieved = old_entries.get(station, {}).get("download_utc", "cached before current manifest")
        else:
            print("Downloading", station, flush=True)
            with urlopen(url, timeout=90) as response:
                content = response.read()
            path.write_bytes(content)
            retrieved = stamp()
        files.append({
            "station": station, "url": url, "file": path.relative_to(ZERO).as_posix(),
            "download_utc": retrieved, "bytes": len(content),
            "rows": len(content.splitlines()),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    manifest = {
        "source": "NOAA USCRN hourly02 2026",
        "protocol": "NOAA_ZERO_SHOT_PROTOCOL.md",
        "catalog_url": BASE + "/",
        "catalog_file": catalog_path.name,
        "catalog_sha256": hashlib.sha256(html).hexdigest(),
        "catalog_selection": "alphabetically first station per state after protocol exclusions",
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def evaluate() -> None:
    manifest = json.loads((ZERO / "manifest.json").read_text(encoding="utf-8"))
    if tuple(r["station"] for r in manifest["files"]) != EXPECTED:
        raise ValueError("Manifest station list differs from frozen protocol")
    if tuple(s for s in EXPANDED_CANDIDATES if s != "AL_Brewton_3_NNE") != TRAIN_STATIONS:
        raise ValueError("Seven training stations differ from the prior check")
    original = pd.concat([make_frame(s) for s in TRAIN_STATIONS], ignore_index=True)
    features = [c for c in original if c not in ("station", "origin_time", "valid_history") and not c.startswith("target_")]
    prior = json.loads((ROOT / "site_blend_2026_expanded_heldout.json").read_text(encoding="utf-8"))
    prior_alpha = {r["horizon_hours"]: r["selected_alpha"] for r in prior["results"] if r["station"] == "ALL"}
    new_frames = {}
    availability = []
    for entry in manifest["files"]:
        path = ZERO / entry["file"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("Source file hash changed: " + str(path))
        raw = load_file(path)
        station = entry["station"]
        new_frames[station] = make_frame_from_raw(raw, station)
        availability.append({
            "station": station, "rows": len(raw),
            "vwc5_observed": int(raw.SOIL_MOISTURE_5.notna().sum()),
            "vwc5_fraction": round(float(raw.SOIL_MOISTURE_5.notna().mean()), 5),
            "timestamp_gaps": int(raw.timestamp_utc.diff().dropna().ne(pd.Timedelta(hours=1)).sum()),
        })
    report = {"method": {
        "protocol": "NOAA_ZERO_SHOT_PROTOCOL.md",
        "source_manifest": "zero_shot/manifest.json",
        "training_stations": TRAIN_STATIONS,
        "new_stations": EXPECTED,
        "horizons_hours": HORIZONS,
        "alpha_grid": ALPHAS,
        "min_eligible_origins": MIN_ORIGINS,
        "feature_columns": features,
        "model": "2024-selected pooled residual Ridge refitted on 2023–2025 original seven stations; no new-station fitting or blend weight",
    }, "availability": availability, "station_results": [], "pooled_results": []}
    ledgers = []
    for h in HORIZONS:
        target = f"target_{h}"
        target_time = f"target_time_{h}"
        exact = (original[target_time] - original.origin_time).eq(pd.Timedelta(hours=h))
        same_year = original.origin_time.dt.year.eq(original[target_time].dt.year)
        subset = original.loc[original.valid_history & original[target].notna() & exact & same_year].copy()
        subset["target_year"] = subset[target_time].dt.year
        train = subset[subset.target_year == 2023]
        valid = subset[subset.target_year == 2024]
        scores = []
        for alpha in ALPHAS:
            model = fit_model(train[features].to_numpy(), (train[target] - train.vwc_current).to_numpy(), alpha)
            pred = valid.vwc_current.to_numpy() + model.predict(valid[features].to_numpy())
            scores.append(rmse(valid[target].to_numpy(), pred))
        alpha = ALPHAS[int(np.argmin(scores))]
        if alpha != prior_alpha[h]:
            raise ValueError(f"Selected alpha for {h}h differs from seven-station frozen check")
        fit = subset[subset.target_year.between(2023, 2025)]
        model = fit_model(fit[features].to_numpy(), (fit[target] - fit.vwc_current).to_numpy(), alpha)
        horizon_ledgers = []
        for station in EXPECTED:
            frame = new_frames[station]
            exact_new = (frame[target_time] - frame.origin_time).eq(pd.Timedelta(hours=h))
            same_year_new = frame.origin_time.dt.year.eq(frame[target_time].dt.year)
            test = frame.loc[frame.valid_history & frame[target].notna() & exact_new &
                             same_year_new & frame[target_time].dt.year.eq(2026)]
            n = len(test)
            station_row = {"station": station, "horizon_hours": h, "n_origins": n, "eligible": n >= MIN_ORIGINS}
            if n:
                truth = test[target].to_numpy()
                persistence = test.vwc_current.to_numpy()
                ridge = persistence + model.predict(test[features].to_numpy())
                station_row.update({
                    "persistence_rmse": round(rmse(truth, persistence), 5),
                    "ridge_rmse": round(rmse(truth, ridge), 5),
                    "ridge_minus_persistence_rmse": round(rmse(truth, ridge) - rmse(truth, persistence), 5),
                    "persistence_mae": round(float(mean_absolute_error(truth, persistence)), 5),
                    "ridge_mae": round(float(mean_absolute_error(truth, ridge)), 5),
                })
                ledger = test[["origin_time", target_time, target]].copy()
                ledger["station"] = station
                ledger["horizon_hours"] = h
                ledger["persistence_pred"] = persistence
                ledger["ridge_pred"] = ridge
                ledger = ledger.rename(columns={target_time: "target_time", target: "truth"})
                ledgers.append(ledger)
                if n >= MIN_ORIGINS:
                    horizon_ledgers.append(ledger)
            report["station_results"].append(station_row)
        if horizon_ledgers:
            pooled = pd.concat(horizon_ledgers, ignore_index=True)
            truth = pooled.truth.to_numpy()
            persistence = pooled.persistence_pred.to_numpy()
            ridge = pooled.ridge_pred.to_numpy()
            site_rows = [r for r in report["station_results"] if r["horizon_hours"] == h and r["eligible"]]
            report["pooled_results"].append({
                "horizon_hours": h,
                "selected_alpha": alpha,
                "n_stations": len(site_rows),
                "n_origins": len(pooled),
                "persistence_rmse": round(rmse(truth, persistence), 5),
                "ridge_rmse": round(rmse(truth, ridge), 5),
                "persistence_mae": round(float(mean_absolute_error(truth, persistence)), 5),
                "ridge_mae": round(float(mean_absolute_error(truth, ridge)), 5),
                "median_station_rmse_difference": round(float(np.median([r["ridge_minus_persistence_rmse"] for r in site_rows])), 5),
                "stations_ridge_better_rmse": sum(r["ridge_minus_persistence_rmse"] < 0 for r in site_rows),
            })
    (ZERO / "zero_shot_2026.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.concat(ledgers, ignore_index=True).to_csv(
        ZERO / "zero_shot_2026_ledger.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    for row in report["pooled_results"]:
        print(row)
    print("Saved", ZERO / "zero_shot_2026.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.download_only and args.evaluate_only:
        parser.error("Choose at most one mode")
    if not args.evaluate_only:
        download()
    if not args.download_only:
        evaluate()
