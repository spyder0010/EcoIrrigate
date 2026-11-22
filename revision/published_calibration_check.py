"""Compare a published nonlinear calibration on fixed forecasts of GN1 probes."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration_ranking_envelope import DELTAS, SEGMENTS, audit_cases
from calibration_scale_sensitivity import sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE / "external_calibration" / "strypsteen"
UNIVERSAL = (170.7, 296.8, 10.0)
SANDY_CLAY_LOAM = (180.20, 364.30, 6.2)
ADC_RANGE = (185.0, 467.5)
HORIZONS = (1, 6, 12, 24)
LAGS = (1, 3, 6, 12)
MODEL_FIELDS = ["ridge", "persistence", "target"]


def moisture(adc, coefficients):
    a, b, c = coefficients  # c is expressed in percentage points
    adc = np.asarray(adc)
    assert (adc > a).all()
    return -c * np.log((adc - a) / b)


def adc_from_universal(vwc):
    a, b, c = UNIVERSAL
    return a + b * np.exp(-np.asarray(vwc) / c)


def read_probe_workbook():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    source = next(f for f in manifest["files"] if f["name"] == "Field experiment GN1.xlsx")
    path = ROOT / source["name"]
    assert sha256(path) == source["sha256"]
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        header = next(workbook.active.iter_rows(min_row=8, max_row=8, values_only=True))
        assert header == ("year", "month", "day", "hour", 1, 2, 3, 4, None, "WET sensor")
        records = [{"time": datetime(*map(int, row[:4])), "probe_2": row[5], "probe_4": row[7],
                    "wet_reference_pct": row[9]}
                   for row in workbook.active.iter_rows(min_row=9, values_only=True)]
    finally:
        workbook.close()
    frame = pd.DataFrame(records).set_index("time").sort_index()
    for field in ("probe_2", "probe_4"):
        frame[field] = pd.to_numeric(frame[field].mask(frame[field].eq("#DIV/0!")), errors="raise")
    assert frame.index.is_unique and len(frame) == 962
    assert frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq="h", name="time"))
    return frame, source


def forecasts(raw, cutoff):
    valid = raw.between(*ADC_RANGE)
    vwc = pd.Series(np.nan, index=raw.index, dtype=float)
    vwc[valid] = moisture(raw[valid], UNIVERSAL)
    clip = sorted(moisture(ADC_RANGE, UNIVERSAL))
    complete = valid.rolling(13).sum().eq(13)
    features = pd.DataFrame({"current": vwc, **{f"change_{lag}": vwc - vwc.shift(lag) for lag in LAGS}})
    ledgers = []
    for h in HORIZONS:
        future = vwc.shift(-h)
        target_time = raw.index + pd.Timedelta(hours=h)
        fit = complete & future.notna() & (target_time < cutoff)
        test = complete & future.notna() & (raw.index >= cutoff)
        assert fit.sum() > 48 and test.sum() > 48
        model = make_pipeline(StandardScaler(), Ridge(alpha=1))
        model.fit(features[fit], (future - vwc)[fit])
        origin = raw.index[test]
        residual = model.predict(features[test])
        predicted = (vwc[test] + residual).clip(*clip)
        ledgers.append(pd.DataFrame({"origin": origin, "target_time": target_time[test],
                                     "horizon": h, "ridge": predicted.to_numpy(),
                                     "persistence": vwc[test].to_numpy(), "target": future[test].to_numpy(),
                                     "n_fit": int(fit.sum())}))
    return pd.concat(ledgers, ignore_index=True)


def curves(ledger, bounds):
    edges = np.asarray(bounds["edges"])
    low, high = edges[[0, -1]]
    y = moisture(adc_from_universal(edges), SANDY_CLAY_LOAM)
    gain = (high - low) / (y[-1] - y[0])
    intercept = low - gain * y[0]
    knots = gain * y + intercept
    assert gain > 0
    a0, _, c0 = UNIVERSAL
    a, _, c = SANDY_CLAY_LOAM
    adc_low, adc_high = adc_from_universal([high, low])
    adc_ends = np.array([adc_low, adc_high])
    slope_ends = gain * c / c0 * (adc_ends - a0) / (adc_ends - a)
    slope_min, slope_max = sorted(slope_ends)
    delta = max(1 - slope_min, slope_max - 1)
    interval_slopes = np.diff(knots) / np.diff(edges)
    assert (interval_slopes >= slope_min - 1e-12).all()
    assert (interval_slopes <= slope_max + 1e-12).all()
    # Here a > a0, so (ADC-a0)/(ADC-a)^2 decreases with ADC on its domain.
    assert a > a0 and adc_low > a
    curvature = gain * c / c0 ** 2 * (a - a0) * (adc_low - a0) / (adc_low - a) ** 2
    interpolation_error = curvature * ((high - low) / SEGMENTS) ** 2 / 8
    output = {"scored_ucl_range": [low, high], "normalized_slope_range": [slope_min, slope_max],
              "minimum_symmetric_slope_budget": delta, "calibration_knots": knots.tolist(),
              "pointwise_interpolation_error_upper": interpolation_error, "by_horizon": {}}
    for h, block in ledger.groupby("horizon"):
        focus = block[MODEL_FIELDS].to_numpy()
        ref = moisture(adc_from_universal(focus), SANDY_CLAY_LOAM)
        exact = gain * ref + intercept
        approx = np.interp(focus, edges, knots)
        np.testing.assert_allclose(adc_from_universal(moisture(adc_from_universal(focus), UNIVERSAL)),
                                   adc_from_universal(focus), atol=1e-10, rtol=0)
        assert np.max(np.abs(exact - approx)) <= interpolation_error + 1e-10

        def delta_mse(values):
            m, p, y = values.T
            return float(np.mean((m - y) ** 2 - (p - y) ** 2))

        nominal, direct, interpolated = (delta_mse(x) for x in (focus, exact, approx))
        unnormalized = delta_mse(ref)
        np.testing.assert_allclose(direct, gain ** 2 * unnormalized, atol=1e-10, rtol=0)
        m, p, actual = focus.T
        slack = 4 * interpolation_error * slope_max * np.mean(np.abs(m - actual) + np.abs(p - actual))
        assert abs(direct - interpolated) <= slack + 1e-10
        certs = {}
        for budget in DELTAS:
            outer = bounds["horizons"][str(h)]["bounds"][str(budget)]["mse"]
            outer_lo, outer_hi = outer["min"]["outer"], outer["max"]["outer"]
            if budget >= delta:
                assert outer_lo <= interpolated <= outer_hi
                assert outer_lo - slack <= direct <= outer_hi + slack
            certs[str(budget)] = {"smooth_curve_eligible": bool(budget >= delta),
                                  "piecewise_outer": [outer_lo, outer_hi],
                                  "smooth_curve_outer_if_eligible": [outer_lo - slack, outer_hi + slack]
                                  if budget >= delta else None}
        output["by_horizon"][str(h)] = {
            "fit_origins": int(block.n_fit.iloc[0]), "test_origins": len(block),
            "nominal_ucl_mse_delta": nominal, "published_sscl_mse_delta": unnormalized,
            "normalized_smooth_mse_delta": direct, "normalized_interpolated_mse_delta": interpolated,
            "interpolation_score_difference_upper": float(slack), "envelopes": certs,
        }
    return output


def main():
    frame, source = read_probe_workbook()
    cutoff = frame.index[int(0.8 * len(frame))]
    result = {"source": source, "first_hour": str(frame.index[0]), "last_hour": str(frame.index[-1]),
              "test_cutoff": str(cutoff), "source_hour_rows": len(frame),
              "wet_reference_rows": int(frame.wet_reference_pct.notna().sum()), "probes": {}}
    ledgers = []
    for name in ("probe_2", "probe_4"):
        forecasts_for_probe = forecasts(frame[name], cutoff)
        bounds = audit_cases(forecasts_for_probe, MODEL_FIELDS, "horizon")
        result["probes"][name] = curves(forecasts_for_probe, bounds)
        forecasts_for_probe.insert(0, "probe", name)
        ledgers.append(forecasts_for_probe)
        print(name, result["probes"][name]["normalized_slope_range"],
              {h: (row["test_origins"], row["nominal_ucl_mse_delta"], row["normalized_smooth_mse_delta"])
               for h, row in result["probes"][name]["by_horizon"].items()}, flush=True)
    (ROOT / "published_calibration_check.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.concat(ledgers, ignore_index=True).to_csv(ROOT / "published_calibration_forecasts.csv", index=False)


if __name__ == "__main__":
    main()
