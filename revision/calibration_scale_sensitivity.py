"""Exploratory sensitivity of proxy-scale forecast rankings to monotone rescaling.

This does not estimate or validate an EcoIrrigate calibration map. It answers only
whether a model comparison made on the supplied sensor-derived scale is invariant
to a family of hypothetical, strictly increasing maps.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from analysis_paths import SOURCE_CSV

ROOT = Path(__file__).resolve().parent
SOURCE = SOURCE_CSV
LEDGERS = [ROOT / "architecture_benchmark" / f"bilstm_no_aux_seed{seed}" / "architecture_predictions.csv" for seed in range(42, 47)]
OUT = ROOT / "calibration_scale_sensitivity.json"
BETAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(x: float, lo: float, hi: float, beta: float) -> float:
    if beta == 0:
        return x
    z = (x - lo) / (hi - lo)
    return lo + (hi - lo) * math.expm1(beta * z) / math.expm1(beta)


def main() -> None:
    if not SOURCE.is_file() or any(not p.is_file() for p in LEDGERS):
        raise FileNotFoundError("Required archived data or prediction ledger is missing")
    with SOURCE.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    columns = list(rows[0])
    target = "Volumetric_Moisture_Pct"
    if target not in columns:
        raise ValueError(f"Expected {target}; got {columns}")
    # These bounds are descriptive fixed anchors, not an inferred calibration.
    vals = [float(row[target]) for row in rows]
    lo, hi = min(vals), max(vals)
    output = {
        "status": "illustrative only; final test ledgers were already inspected",
        "source_sha256": sha256(SOURCE),
        "source_vwc_range": [lo, hi],
        "map": "g_b(x)=lo+(hi-lo)*expm1(b*(x-lo)/(hi-lo))/expm1(b); g_0(x)=x",
        "interpretation": "Each map is strictly increasing and fixes the supplied range endpoints. No map is asserted to be a physical calibration.",
        "ledgers": {},
    }
    for seed, path in zip(range(42, 47), LEDGERS):
        accum = {(h, b): [0, 0.0, 0.0] for h in ("1h", "6h", "12h", "24h") for b in BETAS}
        seen = set()
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["variant"] != "bilstm_residual":
                    continue
                key = (row["farm"], row["horizon"], row["origin_timestamp"])
                if key in seen:
                    raise ValueError(f"Duplicate origin {key} in {path}")
                seen.add(key)
                h = row["horizon"]
                y = float(row["target_vwc_pct"])
                p = float(row["current_vwc_pct"])
                m = float(row["model_vwc_pct"])
                for b in BETAS:
                    gy = transform(y, lo, hi, b)
                    ep = transform(p, lo, hi, b) - gy
                    em = transform(m, lo, hi, b) - gy
                    a = accum[h, b]
                    a[0] += 1
                    a[1] += ep * ep
                    a[2] += em * em
        output["ledgers"][str(seed)] = {
            "sha256": sha256(path),
            "by_horizon": {
                h: {
                    str(b): {
                        "n": accum[h, b][0],
                        "persistence_rmse": math.sqrt(accum[h, b][1] / accum[h, b][0]),
                        "model_rmse": math.sqrt(accum[h, b][2] / accum[h, b][0]),
                        "model_minus_persistence_rmse": math.sqrt(accum[h, b][2] / accum[h, b][0]) - math.sqrt(accum[h, b][1] / accum[h, b][0]),
                    }
                    for b in BETAS
                }
                for h in ("1h", "6h", "12h", "24h")
            },
        }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for h in ("1h", "6h", "12h", "24h"):
        print(h, {str(b): round(sum(output["ledgers"][str(s)]["by_horizon"][h][str(b)]["model_minus_persistence_rmse"] for s in range(42, 47)) / 5, 4) for b in BETAS})


if __name__ == "__main__":
    main()
