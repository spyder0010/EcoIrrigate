"""Paired calendar-block uncertainty for frozen NOAA 2026 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "external_noaa"
N_BOOT = 3000
SEED = 20261114
BLOCK_HOURS = (24, 72)


def main(expanded: bool = False) -> None:
    source_name = "site_blend_2026_expanded_heldout_ledger.csv" if expanded else "site_blend_2026_heldout_ledger.csv"
    source = ROOT / source_name
    ledger = pd.read_csv(source, parse_dates=["origin_time", "target_time"])
    if not ledger.origin_time.dt.year.eq(ledger.target_time.dt.year).all():
        raise ValueError("Cross-year origin found in annual NOAA evaluation ledger")
    rng = np.random.default_rng(SEED)
    report = {"method": {
        "source": source.name,
        "resamples": N_BOOT,
        "seed": SEED,
        "block_hours": BLOCK_HOURS,
        "resampling": "Resample fixed calendar-time blocks with replacement within each station; use paired errors for both methods.",
        "interpretation": "Conditional intervals for the observed stations and their incomplete 2026 coverage; stations are not independent population replicates.",
    }, "results": []}
    for h in sorted(ledger.horizon_hours.unique()):
        frame = ledger[ledger.horizon_hours == h].copy()
        for block_hours in BLOCK_HOURS:
            timestamp = pd.to_datetime(frame.origin_time, utc=True)
            frame["block"] = (timestamp.astype("int64") // (block_hours * 3600 * 10**9)).astype("int64")
            frame["persistence_sq"] = (frame.truth - frame.persistence_pred) ** 2
            frame["ridge_sq"] = (frame.truth - frame.ridge_pred) ** 2
            frame["blend_sq"] = (frame.truth - frame.blend_pred) ** 2
            grouped = frame.groupby(["station", "block"], sort=True).agg(
                n=("truth", "size"),
                persistence_sq=("persistence_sq", "sum"),
                ridge_sq=("ridge_sq", "sum"),
                blend_sq=("blend_sq", "sum"),
            ).reset_index()
            station_arrays = {
                s: g[["n", "persistence_sq", "ridge_sq", "blend_sq"]].to_numpy(dtype=float)
                for s, g in grouped.groupby("station", sort=True)
            }
            for station in ("ALL",) + tuple(station_arrays):
                selected = station_arrays if station == "ALL" else {station: station_arrays[station]}
                observed = np.sum([a.sum(axis=0) for a in selected.values()], axis=0)
                persistence = np.sqrt(observed[1] / observed[0])
                ridge = np.sqrt(observed[2] / observed[0])
                blend = np.sqrt(observed[3] / observed[0])
                boot_ridge = np.empty(N_BOOT)
                boot_blend = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    totals = np.sum([
                        arr[rng.integers(0, len(arr), size=len(arr))].sum(axis=0)
                        for arr in selected.values()
                    ], axis=0)
                    base = np.sqrt(totals[1] / totals[0])
                    boot_ridge[b] = np.sqrt(totals[2] / totals[0]) - base
                    boot_blend[b] = np.sqrt(totals[3] / totals[0]) - base
                report["results"].append({
                    "horizon_hours": int(h),
                    "block_hours": block_hours,
                    "station": station,
                    "n_origins": int(observed[0]),
                    "n_blocks": int(sum(len(a) for a in selected.values())),
                    "persistence_rmse": round(float(persistence), 5),
                    "ridge_rmse": round(float(ridge), 5),
                    "blend_rmse": round(float(blend), 5),
                    "ridge_minus_persistence": round(float(ridge - persistence), 5),
                    "blend_minus_persistence": round(float(blend - persistence), 5),
                    "ridge_minus_persistence_ci": np.round(np.quantile(boot_ridge, (0.025, 0.975)), 5).tolist(),
                    "blend_minus_persistence_ci": np.round(np.quantile(boot_blend, (0.025, 0.975)), 5).tolist(),
                })
    suffix = "expanded_" if expanded else ""
    (ROOT / f"site_blend_2026_{suffix}paired_blocks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["results"]:
        if row["block_hours"] == 72:
            print({k: row[k] for k in ("horizon_hours", "station", "n_blocks", "blend_minus_persistence", "blend_minus_persistence_ci")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", action="store_true")
    main(parser.parse_args().expanded)
