"""Re-evaluate the supplied MultiTaskNet checkpoint on its original test recipe.

Reports calibration and forecasting separately, in percentage points on the supplied target scale. This is a
diagnostic of the existing artifact, not the corrected common-origin benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from analysis_paths import DATA_ROOT, SOURCE_CSV

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = DATA_ROOT / "deep_learning_model"
sys.path.insert(0, str(MODEL_ROOT))
from data.data_loader import MultiTaskDataset  # noqa: E402
from data.preprocessing import DataPreprocessor  # noqa: E402
from models.architectures import MultiTaskNet  # noqa: E402

SOURCE = SOURCE_CSV
CHECKPOINT = next((MODEL_ROOT / "results" / "models").glob("MultiTaskNet_multi-task_*.pt"))
OUT = Path(__file__).resolve().parent / "baseline_audit"
HORIZONS = {"1h": 4, "6h": 24, "12h": 48, "24h": 96}


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "rmse_vwc_percentage_points": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_vwc_percentage_points": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    torch.set_num_threads(4)
    processed = DataPreprocessor().preprocess_pipeline(str(SOURCE))
    feature_groups = processed["feature_groups"]
    test = processed["test"]
    dataset = MultiTaskDataset(
        test,
        feature_groups["calibration_features"],
        feature_groups["calibration_features"] + [feature_groups["target"]],
        farm_encoding={farm: i for i, farm in enumerate(sorted(test.Farm_ID.unique()))},
    )
    model = MultiTaskNet(calib_input_dim=6, seq_input_dim=7)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    calib_pred, forecast_pred = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=128, shuffle=False):
            c, f = model(batch["calib_features"], batch["seq_features"], batch["farm_id"])
            calib_pred.append(c.numpy().reshape(-1))
            forecast_pred.append(f.numpy())
    calib_pred = np.concatenate(calib_pred)
    forecast_pred = np.concatenate(forecast_pred)
    scaler = processed["scalers"]["Volumetric_Moisture_Pct"]
    to_vwc = lambda z: np.asarray(z) * scaler.scale_[0] + scaler.mean_[0]
    rows = []
    index = 0
    for farm in sorted(test.Farm_ID.unique()):
        frame = test[test.Farm_ID.eq(farm)].sort_values("Timestamp").reset_index(drop=True)
        y = to_vwc(frame.Volumetric_Moisture_Pct.to_numpy())
        for t in range(96, len(frame) - 96):
            sample = dataset[index]
            # The original sequence stops at t-1; the calibration input is at t.
            assert np.isclose(float(sample["calib_target"][0]), frame.Volumetric_Moisture_Pct.iloc[t])
            for j, (name, h) in enumerate(HORIZONS.items()):
                rows.append({
                    "farm": farm,
                    "horizon": name,
                    "last_sequence_timestamp": frame.Timestamp.iloc[t - 1],
                    "calibration_timestamp": frame.Timestamp.iloc[t],
                    "target_timestamp": frame.Timestamp.iloc[t + h],
                    "target_vwc_pct": y[t + h],
                    "model_vwc_pct": float(to_vwc(forecast_pred[index, j])),
                    "persistence_last_sequence_vwc_pct": y[t - 1],
                    "persistence_current_vwc_pct": y[t],
                })
            index += 1
    assert index == len(dataset)
    ledger = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    ledger.to_csv(OUT / "saved_multitask_prediction_ledger.csv", index=False, float_format="%.10g")
    results = {"checkpoint": str(CHECKPOINT), "checkpoint_epoch": int(checkpoint["epoch"]), "n_calibration": len(dataset)}
    results["calibration"] = score(to_vwc(np.array([float(dataset[i]["calib_target"][0]) for i in range(len(dataset))])), to_vwc(calib_pred))
    results["forecast"] = {}
    for horizon, part in ledger.groupby("horizon"):
        target = part.target_vwc_pct.to_numpy()
        results["forecast"][horizon] = {
            "model": score(target, part.model_vwc_pct.to_numpy()),
            "persistence_last_sequence": score(target, part.persistence_last_sequence_vwc_pct.to_numpy()),
            "persistence_current": score(target, part.persistence_current_vwc_pct.to_numpy()),
        }
    (OUT / "saved_multitask_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
