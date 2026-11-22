"""Numerical inner and outer error bounds for common monotone scale changes."""
from __future__ import annotations

from itertools import combinations_with_replacement
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from calibration_scale_sensitivity import LEDGERS, ROOT, sha256

OUT = ROOT / "calibration_ranking_envelope.json"
DELTAS = (0.10, 0.25, 0.50)
SEGMENTS = 8


def length_vectors(values, edges):
    return np.clip(values[:, None] - edges[:-1], 0, np.diff(edges))


def relaxation(q, widths, lo, hi):
    """Build the linear outer relaxation and lifted quadratic objective."""
    n = len(widths)
    pairs = list(combinations_with_replacement(range(n), 2))
    k = {pair: i + n for i, pair in enumerate(pairs)}
    basis = np.eye(n + len(pairs))
    area = np.zeros(len(basis))
    area[:n] = widths
    equations = [area]
    inequalities, right = [], []
    objective = np.zeros(len(basis))
    for i, j in pairs:
        s_i, s_j, product = basis[i], basis[j], basis[k[i, j]]
        objective[k[i, j]] = q[i, j] * (1 if i == j else 2)
        if i == j:
            for slope in np.linspace(lo, hi, 5):
                inequalities.append(2 * slope * s_i - product)
                right.append(slope ** 2)
            inequalities.append(product - (lo + hi) * s_i)
            right.append(-lo * hi)
        else:
            inequalities.extend([lo * (s_i + s_j) - product, hi * (s_i + s_j) - product,
                                 product - hi * s_i - lo * s_j, product - lo * s_i - hi * s_j])
            right.extend([lo ** 2, hi ** 2, -lo * hi, -lo * hi])
    for i in range(n):
        row = -basis[i]
        for j in range(n):
            row[k[min(i, j), max(i, j)]] += widths[j]
        equations.append(row)
    options = {"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9}
    return objective, {"A_ub": inequalities, "b_ub": right, "A_eq": equations,
                       "b_eq": [1] + [0] * n,
                       "bounds": [(lo, hi)] * n + [(lo ** 2, hi ** 2)] * len(pairs),
                       "method": "highs", "options": options}


def dual_bound(objective, args, solution):
    """Correct approximate solver multipliers by minimizing over the variable box."""
    inequality_weights = np.minimum(solution.ineqlin.marginals, 0)
    equality_weights = solution.eqlin.marginals
    residual = (objective - inequality_weights @ np.asarray(args["A_ub"])
                - equality_weights @ np.asarray(args["A_eq"]))
    limits = np.asarray(args["bounds"])
    box_min = np.minimum(residual * limits[:, 0], residual * limits[:, 1]).sum()
    lower = inequality_weights @ args["b_ub"] + equality_weights @ args["b_eq"] + box_min
    assert lower <= solution.fun + 1e-8
    return float(lower) - 1e-7


def solve(absolute, q, widths, delta):
    lo, hi = 1 - delta, 1 + delta
    objective, args = relaxation(q, widths, lo, hi)
    rng = np.random.default_rng(42)
    absolute_results, squared_results = {}, {}
    for sign, label in ((1, "min"), (-1, "max")):
        lp = linprog(sign * objective, **args)
        abs_lp = linprog(sign * absolute, A_eq=[widths], b_eq=[1], bounds=[(lo, hi)] * len(widths))
        assert lp.success and abs_lp.success, (lp.message, abs_lp.message)
        absolute_results[label] = {"value": float(absolute @ abs_lp.x), "slopes": abs_lp.x.tolist()}
        best = np.ones(len(widths))
        for start in [best, lp.x[:len(widths)], abs_lp.x] + [
            1 + delta * 0.45 * (v - widths @ v) for v in rng.uniform(-1, 1, (6, len(widths)))
        ]:
            found = minimize(lambda s: sign * float(s @ q @ s), start,
                             jac=lambda s: 2 * sign * (q @ s),
                             bounds=[(lo, hi)] * len(widths), method="SLSQP",
                             constraints={"type": "eq", "fun": lambda s: widths @ s - 1,
                                          "jac": lambda s: widths},
                             options={"ftol": 1e-12, "maxiter": 300})
            s = found.x
            if abs(widths @ s - 1) <= 1e-8 and (s >= lo - 1e-8).all() and (s <= hi + 1e-8).all():
                if sign * (s @ q @ s) < sign * (best @ q @ best):
                    best = s
        witness = float(best @ q @ best)
        bound = sign * dual_bound(sign * objective, args, lp)
        assert sign * (witness - bound) >= -1e-8
        squared_results[label] = {"witness": witness, "slopes": best.tolist(), "outer": bound}
    return {"mae": absolute_results, "mse": squared_results}


def direct_validate(result, m, p, y, edges, delta):
    widths = np.diff(edges)
    for metric, solver_value in (("mae", "value"), ("mse", "witness")):
        for entry in result[metric].values():
            slopes = np.array(entry["slopes"])
            assert (np.abs(slopes - 1) <= delta + 1e-8).all()
            levels = edges[0] + np.r_[0, (slopes * widths).cumsum()]
            np.testing.assert_allclose(levels[-1], edges[-1], atol=1e-8, rtol=0)
            obs = np.interp(y, edges, levels)
            m_error = np.abs(np.interp(m, edges, levels) - obs)
            p_error = np.abs(np.interp(p, edges, levels) - obs)
            loss = (m_error - p_error) if metric == "mae" else (m_error ** 2 - p_error ** 2)
            np.testing.assert_allclose(loss.mean(), entry[solver_value], atol=1e-10, rtol=0)


def audit_cases(df, fields, horizon_col):
    values = df[fields].to_numpy()
    assert np.isfinite(values).all()
    edges = np.linspace(values.min(), values.max(), SEGMENTS + 1)
    widths = np.diff(edges) / (edges[-1] - edges[0])
    output = {"edges": edges.tolist(), "horizons": {}}
    for horizon, block in df.groupby(horizon_col):
        m, p, y = block[fields].to_numpy().T
        y_len = length_vectors(y, edges)
        a = np.abs(length_vectors(m, edges) - y_len)
        b = np.abs(length_vectors(p, edges) - y_len)
        absolute = (a - b).mean(axis=0)
        q = (a.T @ a - b.T @ b) / len(a)
        nominal_mse = float(((m - y) ** 2 - (p - y) ** 2).mean())
        nominal_mae = float((np.abs(m - y) - np.abs(p - y)).mean())
        np.testing.assert_allclose([q.sum(), absolute.sum()], [nominal_mse, nominal_mae], atol=1e-10, rtol=0)
        results = {}
        for delta in DELTAS:
            result = solve(absolute, q, widths, delta)
            direct_validate(result, m, p, y, edges, delta)
            results[str(delta)] = result
        output["horizons"][str(horizon)] = {"rows": len(a), "nominal_mae_delta": nominal_mae,
                                           "nominal_mse_delta": nominal_mse, "bounds": results}
    return output


def combine_sites(audits):
    horizons = sorted({h for audit in audits.values() for h in audit["horizons"]})
    pooled = {}
    for h in horizons:
        available = [audit["horizons"][h] for audit in audits.values() if h in audit["horizons"]]
        weights = np.array([r["rows"] for r in available])
        data = {"rows": int(weights.sum()), "sites": len(available), "bounds": {}}
        for statistic in ("nominal_mae_delta", "nominal_mse_delta"):
            data[statistic] = float(np.average([r[statistic] for r in available], weights=weights))
        for delta in map(str, DELTAS):
            data["bounds"][delta] = {
                metric: {end: {field: float(np.average(
                    [r["bounds"][delta][metric][end][field] for r in available], weights=weights))
                    for field in fields} for end in ("min", "max")}
                for metric, fields in (("mae", ("value",)), ("mse", ("witness", "outer")))
            }
        pooled[h] = data
    return pooled


def main():
    cols = ["farm", "horizon", "origin_timestamp"]
    fields = ["model_vwc_pct", "current_vwc_pct", "target_vwc_pct"]
    ledgers = []
    for path in LEDGERS:
        frame = pd.read_csv(path)
        frame = frame.loc[frame.variant == "bilstm_residual", cols + fields].sort_values(cols).reset_index(drop=True)
        assert not frame.duplicated(cols).any()
        if ledgers:
            pd.testing.assert_frame_equal(frame[cols + fields[1:]], ledgers[0][cols + fields[1:]])
        ledgers.append(frame)
    df = pd.concat(ledgers)
    output = audit_cases(df, fields, "horizon")
    output["ledger_sha256"] = {p.parent.name: sha256(p) for p in LEDGERS}
    output["row_weights"] = "equal weight per row and seed, common origin within each horizon"
    for result in output["horizons"].values():
        result["origins"] = result["rows"] // len(LEDGERS)
    per_farm = {farm: audit_cases(block, fields, "horizon") for farm, block in df.groupby("farm")}
    output["ecoirrigate_separate_scales"] = {"sites": per_farm, "pooled": combine_sites(per_farm)}
    print("EcoIrrigate common and separate scales complete", flush=True)

    noaa_path = ROOT / "external_noaa/zero_shot/zero_shot_2026_ledger.csv.gz"
    noaa = pd.read_csv(noaa_path)
    assert not noaa.duplicated(["station", "origin_time", "horizon_hours"]).any()
    assert noaa.station.nunique() == 35
    noaa_fields = ["ridge_pred", "persistence_pred", "truth"]
    per_station = {}
    for station, block in noaa.groupby("station"):
        per_station[station] = audit_cases(block, noaa_fields, "horizon_hours")
    output["uscrn_separate_scales"] = {"ledger_sha256": sha256(noaa_path), "sites": per_station,
                                      "pooled": combine_sites(per_station)}
    print("USCRN separate station scales complete", flush=True)
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
