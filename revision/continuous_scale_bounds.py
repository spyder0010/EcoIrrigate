"""Enclose paired MSE gaps under continuous slope-bounded score transformations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from calibration_ranking_envelope import DELTAS, OUT as EIGHT_BIN_OUT
from calibration_ranking_envelope import dual_bound, length_vectors, relaxation
from calibration_scale_sensitivity import LEDGERS, ROOT, sha256

GRIDS = (8, 16, 32)
OUT = ROOT / "continuous_scale_bounds.json"
FIELDS = ["model_vwc_pct", "current_vwc_pct", "target_vwc_pct"]
KEYS = ["farm", "horizon", "origin_timestamp"]


def increment_bound(u, v, edges, delta):
    left, right = np.minimum(u, v), np.maximum(u, v)
    widths = np.diff(edges)
    overlaps = np.clip(
        np.minimum(right[:, None], edges[1:]) - np.maximum(left[:, None], edges[:-1]),
        0,
        widths,
    )
    return (2 * delta * overlaps * (1 - overlaps / widths)).sum(axis=1)


def gap_bound(m, p, y, edges, delta):
    model_dist, persistence_dist = np.abs(m - y), np.abs(p - y)
    model_bound = increment_bound(m, y, edges, delta)
    persistence_bound = increment_bound(p, y, edges, delta)
    error_bound = 2 * (model_dist * model_bound + persistence_dist * persistence_bound)
    paired_bound = ((model_dist + persistence_dist) * increment_bound(m, p, edges, delta)
                    + np.abs(m - p) * (model_bound + persistence_bound))
    return (1 + delta) * np.minimum(error_bound, paired_bound)


def grid_interval(m, p, y, edges, delta):
    anchor = length_vectors(y, edges)
    a = length_vectors(m, edges) - anchor
    b = length_vectors(p, edges) - anchor
    q = (a.T @ a - b.T @ b) / len(y)
    widths = np.diff(edges) / (edges[-1] - edges[0])
    objective, args = relaxation(q, widths, 1 - delta, 1 + delta)
    outer = []
    for sign in (1, -1):
        solution = linprog(sign * objective, **args)
        assert solution.success, solution.message
        outer.append(sign * dual_bound(sign * objective, args, solution))
    margin = float(gap_bound(m, p, y, edges, delta).mean())
    return {"grid_outer": outer, "margin": margin, "continuous_outer": [outer[0] - margin, outer[1] + margin]}


def kernel_moments(m, p, y, endpoints):
    """Integrate the signed coverage of paired error squares on the exact score grid."""
    edges = np.unique(np.r_[m, p, y, endpoints])
    widths = np.diff(edges)
    n = len(widths)
    low = np.searchsorted(edges, np.minimum(np.r_[m, p], np.tile(y, 2)))
    high = np.searchsorted(edges, np.maximum(np.r_[m, p], np.tile(y, 2)))
    weights = np.repeat([1, -1], len(y))
    event_rows = np.r_[low, high]
    event_start = np.tile(low, 2)
    event_end = np.tile(high, 2)
    event_weights = np.r_[weights, -weights]
    order = np.argsort(event_rows)
    offsets = np.searchsorted(event_rows[order], np.arange(n + 1))
    differences = np.zeros(n + 1, dtype=np.int64)
    linear = np.zeros(n)
    absolute = 0.0
    for row in range(n):
        updates = order[offsets[row]:offsets[row + 1]]
        np.add.at(differences, event_start[updates], event_weights[updates])
        np.add.at(differences, event_end[updates], -event_weights[updates])
        coverage = np.cumsum(differences[:-1])
        linear[row] = coverage @ widths
        absolute += widths[row] * (np.abs(coverage) @ widths)
    linear /= len(y)
    absolute /= len(y)
    by_value = np.argsort(linear)
    median = linear[by_value[np.searchsorted(widths[by_value].cumsum(), widths.sum() / 2)]]
    first_order = 2 * (np.abs(linear - median) @ widths)
    direction = np.sign(linear - median)
    tied = direction == 0
    direction[tied] = -(widths @ direction) / widths[tied].sum()
    np.testing.assert_allclose(widths @ direction, 0, atol=1e-10, rtol=0)
    assert np.max(np.abs(direction)) <= 1 + 1e-8
    displacement = np.r_[0, (widths * direction).cumsum()]
    return float(linear @ widths), float(first_order), float(absolute), edges, displacement


def score_witness(m, p, y, edges, levels):
    assert np.isclose(levels[-1], edges[-1], atol=1e-9, rtol=0)
    obs = np.interp(y, edges, levels)
    model_error = np.interp(m, edges, levels) - obs
    persistence_error = np.interp(p, edges, levels) - obs
    return float((model_error ** 2 - persistence_error ** 2).mean())


def witness_coefficients(m, p, y, knots, displacement):
    """Return the linear and quadratic terms of a feasible map's MSE difference."""
    widths = np.diff(knots)
    assert np.isclose(displacement[[0, -1]], 0, atol=1e-9, rtol=0).all()
    assert np.all(np.abs(np.diff(displacement)) <= widths + 1e-11)
    observed = np.interp(y, knots, displacement)
    dm = np.interp(m, knots, displacement) - observed
    dp = np.interp(p, knots, displacement) - observed
    return [float(2 * ((m - y) * dm - (p - y) * dp).mean()), float((dm ** 2 - dp ** 2).mean())]


def audit_site(frame, fields, horizon_col, eight_bin, grids_to_run):
    scores = frame[fields].to_numpy()
    # Use the full site scoring range across horizons, as in the eight-bin audit.
    assert np.isfinite(scores).all() and scores.max() > scores.min()
    endpoints = [float(scores.min()), float(scores.max())]
    np.testing.assert_allclose(endpoints, np.array(eight_bin["edges"])[[0, -1]], atol=1e-12, rtol=0)
    site = {}
    for horizon, block in frame.groupby(horizon_col):
        m, p, y = block[fields].to_numpy().T
        identity = float(((m - y) ** 2 - (p - y) ** 2).mean())
        kernel_identity, first_order, second_order, knots, displacement = kernel_moments(m, p, y, endpoints)
        np.testing.assert_allclose(kernel_identity, identity, atol=1e-10, rtol=0)
        increase = witness_coefficients(m, p, y, knots, displacement)
        directions = {"identity": [0.0, 0.0], "first_order_decrease": [-increase[0], increase[1]],
                      "first_order_increase": increase}
        coarse_knots = np.array(eight_bin["edges"])
        bounds = {}
        for delta in DELTAS:
            grids = {str(n): grid_interval(m, p, y, np.linspace(*endpoints, n + 1), delta) for n in grids_to_run}
            radius = delta * first_order + delta ** 2 * second_order + 1e-9
            kernel = [identity - radius, identity + radius]
            intervals = [kernel] + [r["continuous_outer"] for r in grids.values()]
            interval = [max(r[0] for r in intervals), min(r[1] for r in intervals)]
            previous = eight_bin["horizons"][str(horizon)]["bounds"][str(delta)]["mse"]
            coarse_witnesses = [previous[end]["witness"] for end in ("min", "max")]
            linear_witnesses = [score_witness(m, p, y, knots, knots + sign * delta * displacement)
                                for sign in (-1, 1)]
            for sign, score in zip((-1, 1), linear_witnesses, strict=True):
                np.testing.assert_allclose(score, identity + sign * delta * increase[0] + delta ** 2 * increase[1],
                                           atol=1e-10, rtol=0)
            for end in ("min", "max"):
                direction = (np.array(previous[end]["slopes"]) - 1) / delta
                widths = np.diff(coarse_knots)
                # Restore the endpoint equality after numerical optimization.
                direction -= np.average(direction, weights=widths)
                direction /= max(1, np.max(np.abs(direction)))
                coarse_displacement = np.r_[0, (widths * direction).cumsum()]
                coeffs = witness_coefficients(m, p, y, coarse_knots, coarse_displacement)
                rescored = score_witness(m, p, y, coarse_knots, coarse_knots + delta * coarse_displacement)
                np.testing.assert_allclose(rescored, identity + delta * coeffs[0] + delta ** 2 * coeffs[1],
                                           atol=1e-10, rtol=0)
                np.testing.assert_allclose(rescored, previous[end]["witness"], atol=1e-6, rtol=0)
                directions[f"eight_segment_{end}_{delta}"] = coeffs
            witnesses = [min(coarse_witnesses[0], *linear_witnesses, identity),
                         max(coarse_witnesses[1], *linear_witnesses, identity)]
            if "8" in grids:
                np.testing.assert_allclose(grids["8"]["grid_outer"],
                                           [previous[end]["outer"] for end in ("min", "max")], atol=1e-9, rtol=0)
            assert interval[0] <= witnesses[0] <= identity <= witnesses[1] <= interval[1]
            bounds[str(delta)] = {"outer": interval, "witnesses": witnesses, "kernel_outer": kernel,
                                  "linear_witnesses": linear_witnesses, "grids": grids}
        site[str(horizon)] = {"rows": len(y), "identity": identity, "kernel_first_order": first_order,
                              "kernel_second_order": second_order, "witness_polynomials": directions, "bounds": bounds}
    return site


def verify_margin():
    """Compare the bounds to direct interpolation and scoring at off-grid points."""
    rng = np.random.default_rng(2601)
    fine = np.linspace(-3, 7, 257)
    points = rng.uniform(-3, 7, (3, 2000))
    points[:, :6] = [[-3, -3, 7, 1, 1, 1], [7, -3, 7, 1, 1.01, 0.99], [1, 7, -3, 1.01, 1, 1.02]]
    for delta in DELTAS:
        for _ in range(8):
            slopes = 1 + delta * rng.permutation(np.repeat([-1, 1], 128))
            fine_levels = fine[0] + np.r_[0, (np.diff(fine) * slopes).cumsum()]
            fine_m, fine_p, fine_y = np.interp(points, fine, fine_levels)
            exact = (fine_m - fine_y) ** 2 - (fine_p - fine_y) ** 2
            for n in GRIDS:
                stride = (len(fine) - 1) // n
                coarse = fine[::stride]
                grid_m, grid_p, grid_y = np.interp(points, coarse, fine_levels[::stride])
                interpolated = (grid_m - grid_y) ** 2 - (grid_p - grid_y) ** 2
                bound = gap_bound(*points, coarse, delta)
                assert np.all(np.abs(exact - interpolated) <= bound + 1e-11)


def verify_kernel():
    """Check the sweep against a dense rectangle construction and direct scores."""
    rng = np.random.default_rng(6102)
    m, p, y = rng.choice(np.arange(18) / 8, (3, 29))
    m[:4] = [0, 0, 0.5, 0.5]
    p[:4] = [0, 0.5, 0.5, 0]
    y[:4] = [0, 0, 0.5, 0.5]
    endpoints = [-1, 3]
    edges = np.unique(np.r_[m, p, y, endpoints])
    widths = np.diff(edges)
    a = np.abs(length_vectors(m, edges) - length_vectors(y, edges)) / widths
    b = np.abs(length_vectors(p, edges) - length_vectors(y, edges)) / widths
    dense = (a.T @ a - b.T @ b) / len(y)
    identity, first, second, knots, displacement = kernel_moments(m, p, y, endpoints)
    np.testing.assert_array_equal(edges, knots)
    np.testing.assert_allclose([identity, second], [widths @ dense @ widths,
                               widths @ np.abs(dense) @ widths], atol=1e-12, rtol=0)
    optimum = linprog(-2 * widths * (dense @ widths), A_eq=[widths], b_eq=[0],
                      bounds=[(-1, 1)] * len(widths), method="highs")
    assert optimum.success, optimum.message
    np.testing.assert_allclose(first, -optimum.fun, atol=1e-12, rtol=0)
    np.testing.assert_allclose(2 * np.diff(displacement) @ dense @ widths, first, atol=1e-12, rtol=0)
    for delta in DELTAS:
        for slopes in rng.uniform(-0.5, 0.5, (30, len(widths))):
            residual = delta * (slopes - (widths @ slopes) / widths.sum())
            levels = edges[0] + np.r_[0, (widths * (1 + residual)).cumsum()]
            obs = np.interp(y, edges, levels)
            gap = ((np.interp(m, edges, levels) - obs) ** 2
                   - (np.interp(p, edges, levels) - obs) ** 2).mean()
            assert abs(gap - identity) <= delta * first + delta ** 2 * second + 1e-12


def reversal_budget(results, weights, identity):
    """Bracket the first strict ranking reversal for independent site maps."""
    first = np.average([r["kernel_first_order"] for r in results], weights=weights)
    second = np.average([r["kernel_second_order"] for r in results], weights=weights)
    target = abs(identity) - 1e-9
    denominator = first + np.sqrt(first ** 2 + 4 * second * max(target, 0))
    if target <= 0:
        certified = 0.0
    elif denominator == 0:
        certified = 1.0
    else:
        certified = min(1.0, float(2 * target / denominator))
    budgets = np.arange(1, 1000) / 1000
    sample_sign = np.sign(identity)
    site_minima, chosen_directions = [], []
    for result in results:
        polynomials = result["witness_polynomials"]
        names = np.array(list(polynomials))
        a, b = np.array(list(polynomials.values())).T
        signed_gaps = sample_sign * (result["identity"] + a[:, None] * budgets + b[:, None] * budgets ** 2)
        directions = np.argmin(signed_gaps, axis=0)
        site_minima.append(signed_gaps[directions, np.arange(len(budgets))])
        chosen_directions.append(names[directions])
    adverse_gap = np.average(site_minima, axis=0, weights=weights)
    crossing = np.flatnonzero(adverse_gap < -1e-9)
    result = {"sign_stable_below": certified, "feasible_reversal_at": None,
              "mse_gap_at_reversal": None, "site_map_counts": None}
    if len(crossing):
        idx = crossing[0]
        unique, counts = np.unique(np.array(chosen_directions)[:, idx], return_counts=True)
        assert certified <= budgets[idx]
        result.update(feasible_reversal_at=float(budgets[idx]), mse_gap_at_reversal=float(sample_sign * adverse_gap[idx]),
                      site_map_counts=dict(zip(unique.tolist(), counts.tolist(), strict=True)))
    return result


def pool(sites):
    pooled = {}
    for horizon in next(iter(sites.values())):
        results = [site[horizon] for site in sites.values()]
        weights = [entry["rows"] for entry in results]
        bounds = {}
        for delta in DELTAS:
            entries = [r["bounds"][str(delta)] for r in results]
            bounds[str(delta)] = {field: np.average([entry[field] for entry in entries], axis=0, weights=weights).tolist()
                                  for field in ("outer", "kernel_outer", "witnesses")}
        identity = float(np.average([entry["identity"] for entry in results], weights=weights))
        pooled[horizon] = {"rows": sum(weights), "identity": identity, "bounds": bounds,
                           "reversal_budget": reversal_budget(results, weights, identity)}
    return pooled


def main():
    verify_margin()
    verify_kernel()
    prior = json.loads(EIGHT_BIN_OUT.read_text(encoding="utf-8"))
    hashes = {path.parent.name: sha256(path) for path in LEDGERS}
    assert hashes == prior["ledger_sha256"]
    frames = []
    for path in LEDGERS:
        frame = pd.read_csv(path)
        frame = frame.loc[frame.variant == "bilstm_residual", KEYS + FIELDS].sort_values(KEYS).reset_index(drop=True)
        assert not frame.duplicated(KEYS).any()
        if frames:
            pd.testing.assert_frame_equal(frame[KEYS + FIELDS[1:]], frames[0][KEYS + FIELDS[1:]])
        frames.append(frame)
    eco_prior = prior["ecoirrigate_separate_scales"]["sites"]
    eco = {site: audit_site(frame, FIELDS, "horizon", eco_prior[site], GRIDS)
           for site, frame in pd.concat(frames, ignore_index=True).groupby("farm")}
    print("EcoIrrigate continuous bounds complete", flush=True)

    uscrn_path = ROOT / "external_noaa/zero_shot/zero_shot_2026_ledger.csv.gz"
    uscrn_prior = prior["uscrn_separate_scales"]
    uscrn_hash = sha256(uscrn_path)
    assert uscrn_hash == uscrn_prior["ledger_sha256"]
    uscrn_frame = pd.read_csv(uscrn_path)
    assert not uscrn_frame.duplicated(["station", "origin_time", "horizon_hours"]).any()
    assert set(uscrn_frame.station) == set(uscrn_prior["sites"])
    uscrn_fields = ["ridge_pred", "persistence_pred", "truth"]
    uscrn = {site: audit_site(frame, uscrn_fields, "horizon_hours", uscrn_prior["sites"][site], ())
             for site, frame in uscrn_frame.groupby("station")}
    output = {"ledger_sha256": hashes, "grids": GRIDS, "sites": eco, "pooled": pool(eco),
              "uscrn": {"ledger_sha256": uscrn_hash, "sites": uscrn, "pooled": pool(uscrn)}}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for label, pooled in (("EcoIrrigate", output["pooled"]), ("USCRN", output["uscrn"]["pooled"])):
        for horizon, result in pooled.items():
            print(label, horizon, result["reversal_budget"])


if __name__ == "__main__":
    main()
