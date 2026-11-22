"""Explore whether an event head is learnable from the sealed EcoIrrigate series.

This is an exploratory diagnostic. The final 2025 period was previously inspected
for other models, so these are not confirmatory generalization results.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from analysis_paths import SOURCE_CSV

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SOURCE = SOURCE_CSV
LEDGER = HERE / "architecture_benchmark" / "bilstm_no_aux_seed42" / "architecture_predictions.csv"
OUT = HERE / "event_model_feasibility.json"
HORIZON = 24  # 6 hours at 15-minute sampling
JUMP_STEPS = 4  # trailing one-hour change
THRESHOLD = 0.5  # VWC percentage points in one hour
FEATURES = ["current_vwc", "change_1h", "change_6h", "change_24h", "volatility_6h",
            "volatility_24h", "max_rise_24h", "board_temp", "air_temp", "soil_temp",
            "pressure", "hour_sin", "hour_cos", "day_sin", "day_cos", "farm_2"]


def event_runs(indices: np.ndarray, min_separation: int = 96) -> int:
    if len(indices) == 0:
        return 0
    return 1 + int(np.sum(np.diff(indices) > min_separation))


def make_data() -> tuple[pd.DataFrame, list[dict]]:
    source = pd.read_csv(SOURCE, parse_dates=["Timestamp"])
    frames, episodes = [], []
    for farm, group in source.groupby("Farm_ID", sort=True):
        group = group.sort_values("Timestamp").reset_index(drop=True)
        n = len(group)
        v = group.Volumetric_Moisture_Pct
        jump = v.diff(JUMP_STEPS)
        # For an origin i, evaluate only jumps ending at i+1,...,i+H.
        future_jumps = pd.concat([jump.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
        event = future_jumps.max(axis=1).ge(THRESHOLD)
        frame = pd.DataFrame({
            "farm": farm, "row_index": np.arange(n), "timestamp": group.Timestamp,
            "current_vwc": v,
            "change_1h": v.diff(4), "change_6h": v.diff(24), "change_24h": v.diff(96),
            "volatility_6h": v.diff().rolling(24, min_periods=24).std(),
            "volatility_24h": v.diff().rolling(96, min_periods=96).std(),
            "max_rise_24h": jump.rolling(96, min_periods=96).max(),
            "board_temp": group.Sensor_Board_Temperature_C,
            "air_temp": group.Atm_Temperature_C,
            "soil_temp": group.Soil_Temperature_C,
            "pressure": group.Atm_Pressure_inHg,
            "hour_sin": group.Hour_sin, "hour_cos": group.Hour_cos,
            "day_sin": group.Day_sin, "day_cos": group.Day_cos,
            "farm_2": int(farm.endswith("2")),
            "event": event.astype(int),
        })
        cut1, cut2 = int(n * .70), int(n * .85)
        frame["split"] = np.where(frame.row_index < cut1, "train",
                                  np.where(frame.row_index < cut2, "validation", "test"))
        frame = frame.loc[(frame.row_index >= 96) &
                          (((frame.row_index + HORIZON < cut1) & (frame.split == "train")) |
                           ((frame.row_index >= cut1) & (frame.row_index + HORIZON < cut2) & (frame.split == "validation")) |
                           ((frame.row_index >= cut2) & (frame.row_index + HORIZON < n) & (frame.split == "test")))].copy()
        frames.append(frame)
        for split, lower, upper in (("train", 96, cut1), ("validation", cut1, cut2), ("test", cut2, n)):
            hit = np.flatnonzero((jump >= THRESHOLD) & (jump.index >= lower) & (jump.index < upper))
            starts = [int(hit[i]) for i in range(len(hit)) if i == 0 or hit[i] - hit[i - 1] > 96]
            episodes.append({"farm": farm, "split": split,
                             "one_hour_jump_windows": len(hit),
                             "episodes_separated_by_24h": event_runs(hit),
                             "episode_first_jump_times": [str(group.Timestamp.iloc[i]) for i in starts]})
    return pd.concat(frames, ignore_index=True), episodes


def scores(y: np.ndarray, probability: np.ndarray) -> dict:
    result = {"n": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
              "brier": float(brier_score_loss(y, probability)),
              "average_precision": float(average_precision_score(y, probability))}
    if len(np.unique(y)) > 1:
        result["roc_auc"] = float(roc_auc_score(y, probability))
    return result


def main() -> None:
    data, episodes = make_data()
    train = data[data.split == "train"]
    valid = data[data.split == "validation"]
    test = data[data.split == "test"]
    params = []
    for c in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=c, max_iter=1000))
        model.fit(train[FEATURES], train.event)
        pred = model.predict_proba(valid[FEATURES])[:, 1]
        params.append({"C": c, **scores(valid.event.to_numpy(), pred)})
    chosen = min(params, key=lambda r: r["brier"])["C"]
    fit = data[data.split.isin(("train", "validation"))]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=chosen, max_iter=1000))
    model.fit(fit[FEATURES], fit.event)
    pred = model.predict_proba(test[FEATURES])[:, 1]
    prior = float(fit.event.mean())
    result = {"design": {"status": "exploratory; final 2025 period already inspected for other models",
                         "event": "at least one one-hour VWC rise >= 0.5 percentage points ending within next six hours",
                         "target_horizon_steps": HORIZON, "features": FEATURES,
                         "selection": "C chosen by validation Brier; refit on train plus validation; no future weather",
                         "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                         "neural_ledger_sha256": hashlib.sha256(LEDGER.read_bytes()).hexdigest()},
              "episodes": episodes, "validation_by_C": params, "chosen_C": chosen,
              "test_logistic": scores(test.event.to_numpy(), pred),
              "test_constant_prevalence": scores(test.event.to_numpy(), np.full(len(test), prior)),
              "test_by_farm": {farm: scores(part.event.to_numpy(), pred[test.farm.to_numpy() == farm])
                               for farm, part in test.groupby("farm")},
              "test_positive_probability_mean": float(pred[test.event.to_numpy() == 1].mean()),
              "test_negative_probability_mean": float(pred[test.event.to_numpy() == 0].mean())}
    # Outcome-conditioned error concentration is diagnostic, never a deployable gate.
    ledger = pd.read_csv(LEDGER)
    six = ledger[ledger.horizon == "6h"].copy()
    six["truth_change"] = six.target_vwc_pct - six.current_vwc_pct
    for label, sel in (("wetting_1pt", six.truth_change >= 1.0),
                       ("drying_1pt", six.truth_change <= -1.0),
                       ("other", six.truth_change.abs() < 1.0)):
        n = int(sel.sum())
        p_error = six.target_vwc_pct - six.current_vwc_pct
        m_error = six.target_vwc_pct - six.model_vwc_pct
        result.setdefault("six_hour_error_regimes", []).append({
            "regime": label, "n": n, "fraction": n / len(six),
            "persistence_sse_share": float(np.square(p_error[sel]).sum() / np.square(p_error).sum()),
            "bilstm_sse_share": float(np.square(m_error[sel]).sum() / np.square(m_error).sum())})
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": episodes, "chosen_C": chosen,
                      "test_logistic": result["test_logistic"],
                      "test_constant_prevalence": result["test_constant_prevalence"],
                      "six_hour_error_regimes": result["six_hour_error_regimes"]}, indent=2))


if __name__ == "__main__":
    main()
