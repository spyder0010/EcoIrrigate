"""Conditional resampling sensitivity over the declared measurement scales."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from calibration_ranking_envelope import OUT as SCALE_RESULTS, length_vectors
from calibration_scale_sensitivity import LEDGERS, ROOT, sha256

OUT = ROOT / "scale_sampling_sensitivity.json"
HORIZONS = (1, 6, 12, 24)
BUDGET = 0.25


def row_matrices(frame, columns, edges):
    predicted, persisted, actual = frame[columns].to_numpy().T
    target = length_vectors(actual, edges)
    model = length_vectors(predicted, edges) - target
    baseline = length_vectors(persisted, edges) - target
    return model[:, :, None] * model[:, None, :] - baseline[:, :, None] * baseline[:, None, :]


def deviation_bound(d, delta):
    """Return identity changes and a bound on the sup over separate site maps."""
    assert d.shape[-1] % 2 == 0
    rows = d.sum(axis=-1)
    identity = rows.sum(axis=(-1, -2))
    sorted_rows = np.sort(rows, axis=-1)
    half = sorted_rows.shape[-1] // 2
    linear = 2 * delta * (sorted_rows[..., half:] - sorted_rows[..., :half]).sum(axis=(-1, -2))
    # For u with sum(u) = 0, row/column additive effects vanish in u' D u.
    means = rows / d.shape[-1]
    centered = d - means[..., :, None] - means[..., None, :] + means.mean(axis=-1)[..., None, None]
    quadratic = delta ** 2 * np.abs(centered).sum(axis=(-1, -2, -3))
    return identity, np.abs(identity) + linear + quadratic


def percentiles(x):
    return np.quantile(x, [0.025, 0.975], axis=0).tolist()


def summarize(identity, uniform, empirical):
    rows = {}
    for i, h in enumerate(HORIZONS):
        h_key = f"{h}h" if f"{h}h" in empirical else str(h)
        nominal = empirical[h_key]["nominal_mse_delta"]
        if not np.isfinite(uniform[:, i]).all():
            raise ValueError("Nonfinite scale deviation")
        assert (uniform[:, i] >= np.abs(identity[:, i]) - 1e-12).all()
        point = nominal + identity[:, i]
        halfwidth = float(np.quantile(uniform[:, i], .95, method="higher"))
        scale = empirical[h_key]["bounds"][str(BUDGET)]["mse"]
        outer = [scale["min"]["outer"], scale["max"]["outer"]]
        rows[str(h)] = {
            "nominal": nominal, "identity_percentiles": percentiles(point),
            "identity_absolute_deviation_q95": float(np.quantile(np.abs(identity[:, i]), .95, method="higher")),
            "scale_only_outer": outer, "uniform_deviation_q95_upper": halfwidth,
            "scale_and_resampling_outer": [outer[0] - halfwidth, outer[1] + halfwidth],
        }
    return rows


def ecoirrigate(source):
    keys = ["farm", "horizon", "origin_timestamp", "target_timestamp"]
    columns = ["model_vwc_pct", "current_vwc_pct", "target_vwc_pct"]
    summary = source["ecoirrigate_separate_scales"]
    maps = source["horizons"]["24h"]
    common_edges = np.asarray(source["edges"])
    slopes = np.asarray(maps["bounds"][str(BUDGET)]["mse"]["min"]["slopes"])
    target = None
    site_tensors, witness_mse = [], []
    for path in LEDGERS:
        assert sha256(path) == source["ledger_sha256"][path.parent.name]
        frame = pd.read_csv(path)
        frame = frame.loc[frame.variant.eq("bilstm_residual"), keys + columns].sort_values(keys).reset_index(drop=True)
        assert not frame.duplicated(keys).any()
        aligned = frame[keys + columns[1:]]
        if target is not None:
            pd.testing.assert_frame_equal(aligned, target)
        target = aligned
        tensor, witness_by_site = [], []
        common_times = None
        for site, group in frame.groupby("farm"):
            edges = np.array(summary["sites"][site]["edges"])
            cubes = []
            for h in HORIZONS:
                slice_ = group.loc[group.horizon.eq(f"{h}h")].sort_values("origin_timestamp")
                times = pd.to_datetime(slice_.origin_timestamp)
                assert times.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
                assert (pd.to_datetime(slice_.target_timestamp) - times).eq(pd.Timedelta(hours=h)).all()
                timestamps = times.to_numpy()
                if common_times is None:
                    common_times = timestamps
                np.testing.assert_array_equal(timestamps, common_times)
                cubes.append(row_matrices(slice_, columns, edges))
                if h == 24:
                    common = row_matrices(slice_, columns, common_edges)
                    witness_by_site.append(np.einsum("i,tij,j->t", slopes, common, slopes))
            tensor.append(np.stack(cubes, axis=1))
        site_tensors.append(np.stack(tensor, axis=2))
        witness_mse.append(np.mean(witness_by_site, axis=0))
    time = np.mean(site_tensors, axis=0) / len(summary["sites"])
    witness = np.mean(witness_mse, axis=0)
    n = len(time)
    center = time.mean(axis=0)
    check_center(center, summary["pooled"], eco=True)
    np.testing.assert_allclose(witness.mean(), maps["bounds"][str(BUDGET)]["mse"]["min"]["witness"],
                               rtol=0, atol=1e-10)
    result = {"paired_timestamps": n, "replicates": 4000, "blocks": {}}
    for block in (48, 96, 192):
        rng = np.random.default_rng(20261119 + block)
        identity, uniform, witness_changes = [], [], []
        # Circular block sums make each timestamp equally likely in every position.
        circular = np.concatenate((time, time[:block]))
        prefix = np.concatenate((np.zeros_like(circular[:1]), circular.cumsum(axis=0)))
        witness_prefix = np.r_[0, np.r_[witness, witness[:block]].cumsum()]
        full, remainder = divmod(n, block)
        for _ in range(20):
            starts = rng.integers(0, n, size=(200, full + bool(remainder)))
            lengths = np.full(starts.shape[-1], block)
            if remainder:
                lengths[-1] = remainder
            ends = starts + lengths
            samples = (prefix[ends] - prefix[starts]).sum(axis=1) / n
            d = samples - center
            at_identity, supremum = deviation_bound(d, BUDGET)
            identity.append(at_identity)
            uniform.append(supremum)
            witness_changes.append((witness_prefix[ends] - witness_prefix[starts]).sum(axis=1) / n)
        sampled = np.concatenate(witness_changes)
        identity = np.concatenate(identity)
        result["blocks"][str(block)] = {
            "by_horizon": summarize(identity, np.concatenate(uniform), summary["pooled"]),
            "fixed_24h_witness_percentiles": percentiles(sampled),
            "fraction_with_identity_positive_and_witness_negative": float(np.mean((identity[:, -1] +
                maps["nominal_mse_delta"] > 0) & (sampled < 0))),
        }
    return result


def check_center(weighted, pooled, eco=False):
    nominal = [pooled[f"{h}h" if eco else str(h)]["nominal_mse_delta"] for h in HORIZONS]
    np.testing.assert_allclose(weighted.sum(axis=(-1, -2, -3)), nominal, rtol=0, atol=1e-10)


def uscrn(source):
    summary = source["uscrn_separate_scales"]
    path = ROOT / "external_noaa/zero_shot/zero_shot_2026_ledger.csv.gz"
    assert sha256(path) == summary["ledger_sha256"]
    frame = pd.read_csv(path)
    assert not frame.duplicated(["station", "horizon_hours", "origin_time"]).any()
    station_sum, station_size = [], []
    for station, group in frame.groupby("station"):
        station_stats = summary["sites"][station]
        edges = np.array(station_stats["edges"])
        statistics = [row_matrices(group.loc[group.horizon_hours.eq(h)],
                                   ["ridge_pred", "persistence_pred", "truth"], edges) for h in HORIZONS]
        station_sum.append([a.sum(axis=0) for a in statistics])
        station_size.append([len(a) for a in statistics])
    sums = np.array(station_sum).swapaxes(0, 1)
    counts = np.array(station_size).T
    n = len(station_sum)
    assert n == 35 and (counts > 0).all()
    center = sums / counts.sum(axis=-1)[:, None, None, None]
    check_center(center, summary["pooled"])
    rng = np.random.default_rng(20261119)
    identity, uniform = [], []
    for _ in range(20):
        selection = rng.multinomial(n, np.full(n, 1 / n), size=500)
        total = selection @ counts.T
        weight = selection[:, None, :] / total[:, :, None]
        d = weight[..., None, None] * sums - center
        at_identity, supremum = deviation_bound(d, BUDGET)
        identity.append(at_identity)
        uniform.append(supremum)
    return {"stations": n, "replicates": 10000,
            "by_horizon": summarize(np.concatenate(identity), np.concatenate(uniform), summary["pooled"])}


def check_deviation_function():
    rng = np.random.default_rng(3412)
    d = rng.normal(size=(80, 2, 8, 8))
    d = d + d.swapaxes(-1, -2)
    _, absolute = deviation_bound(d, BUDGET)
    for _ in range(30):
        u = rng.permuted(np.tile(np.r_[[BUDGET] * 4, [-BUDGET] * 4], (80, 2, 1)), axis=-1)
        actual = np.einsum("bsi,bsij,bsj->b", 1 + u, d, 1 + u)
        assert (np.abs(actual) <= absolute + 1e-12).all()


def main():
    check_deviation_function()
    audit = json.loads(SCALE_RESULTS.read_text(encoding="utf-8"))
    output = {"slope_budget": BUDGET, "scale_audit_sha256": sha256(SCALE_RESULTS),
              "ecoirrigate": ecoirrigate(audit), "uscrn": uscrn(audit)}
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, result in (("EcoIrrigate", output["ecoirrigate"]["blocks"]["96"]), ("USCRN", output["uscrn"])):
        print(name, {h: {"identity": row["identity_percentiles"], "combined": row["scale_and_resampling_outer"]}
                     for h, row in result["by_horizon"].items()})


if __name__ == "__main__":
    main()
