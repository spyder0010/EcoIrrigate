"""Expanding-window BiLSTM backtest on disjoint chronological test blocks.

Four folds have train/validation/test fractions 40/10/10, 50/10/10,
60/10/10, and 70/10/10 of each site's timeline. Test blocks are disjoint.
The same architecture and training recipe are used in every fold. The final
backtest block overlaps part of the separate final-15% benchmark period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from forecast_architecture_benchmark import (
    FEATURES, HORIZONS, HORIZON_NAMES, WINDOW, ROOT, SOURCE,
    ForecastModel, OriginDataset, infer, run_epoch, seed_all,
)

BASE_OUT = Path(__file__).resolve().parent / "architecture_benchmark"
FOLDS = [0.4, 0.5, 0.6, 0.7]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 12
PATIENCE = 3
BATCH = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-auxiliary", action="store_true", help="Train a residual BiLSTM without the reconstruction head")
    args = parser.parse_args()
    auxiliary = not args.no_auxiliary
    output = BASE_OUT / ("rolling_backtest" if auxiliary else "rolling_backtest_no_aux")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(SOURCE, parse_dates=["Timestamp"])
    frames = [f.sort_values("Timestamp").reset_index(drop=True) for _, f in data.groupby("Farm_ID", sort=True)]
    all_results = []
    all_predictions = []
    for fold, train_fraction in enumerate(FOLDS, start=1):
        n = len(frames[0])
        train_end = int(n * train_fraction)
        val_end = int(n * (train_fraction + 0.1))
        test_end = int(n * (train_fraction + 0.2))
        scaler = StandardScaler().fit(pd.concat([f.iloc[:train_end][FEATURES] for f in frames]).to_numpy(dtype=np.float32))
        arrays = [scaler.transform(f[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32) for f in frames]
        origins = {
            "train": [(farm, t) for farm in range(len(frames)) for t in range(WINDOW, train_end - 96, 4)],
            "validation": [(farm, t) for farm in range(len(frames)) for t in range(train_end, val_end - 96, 4)],
            "test": [(farm, t) for farm in range(len(frames)) for t in range(val_end, test_end - 96)],
        }
        loaders = {
            split: DataLoader(OriginDataset(arrays, rows), batch_size=BATCH, shuffle=(split == "train"), pin_memory=device.type == "cuda")
            for split, rows in origins.items()
        }
        for seed in SEEDS:
            seed_all(seed)
            model = ForecastModel("bilstm", residual=True, auxiliary=auxiliary).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            best_mse = float("inf")
            best_state = None
            wait = 0
            for epoch in range(1, EPOCHS + 1):
                run_epoch(model, loaders["train"], device, optimizer)
                val_loss = run_epoch(model, loaders["validation"], device)
                if val_loss[1] < best_mse - 1e-6:
                    best_mse = val_loss[1]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        break
            assert best_state is not None
            model.load_state_dict(best_state)
            pred, truth, current, _ = infer(model, loaders["test"], device)
            pred = pred * scaler.scale_[0] + scaler.mean_[0]
            truth = truth * scaler.scale_[0] + scaler.mean_[0]
            current = current * scaler.scale_[0] + scaler.mean_[0]
            for j, horizon in enumerate(HORIZON_NAMES):
                model_rmse = float(np.sqrt(mean_squared_error(truth[:, j], pred[:, j])))
                persistence_rmse = float(np.sqrt(mean_squared_error(truth[:, j], current)))
                all_results.append({
                    "fold": fold,
                    "seed": seed,
                    "horizon": horizon,
                    "train_end": str(frames[0].Timestamp.iloc[train_end - 1]),
                    "validation_end": str(frames[0].Timestamp.iloc[val_end - 1]),
                    "test_end": str(frames[0].Timestamp.iloc[test_end - 1]),
                    "test_origins": len(origins["test"]),
                    "best_epoch": best_epoch,
                    "model_rmse": model_rmse,
                    "persistence_rmse": persistence_rmse,
                    "difference_model_minus_persistence": model_rmse - persistence_rmse,
                })
            for k, (farm, t) in enumerate(origins["test"]):
                for j, horizon in enumerate(HORIZON_NAMES):
                    all_predictions.append({
                        "fold": fold,
                        "seed": seed,
                        "farm": frames[farm].Farm_ID.iloc[0],
                        "horizon": horizon,
                        "origin_timestamp": frames[farm].Timestamp.iloc[t],
                        "target_timestamp": frames[farm].Timestamp.iloc[t + HORIZONS[j]],
                        "target_vwc_pct": truth[k, j],
                        "current_vwc_pct": current[k],
                        "model_vwc_pct": pred[k, j],
                    })
            result_table = pd.DataFrame(all_results)
            result_table.to_csv(output / "rolling_metrics.csv", index=False, float_format="%.10g")
            print(f"fold {fold} seed {seed}: " + ", ".join(
                f"{h} dRMSE={result_table[(result_table.fold.eq(fold)) & (result_table.seed.eq(seed)) & (result_table.horizon.eq(h))].difference_model_minus_persistence.iloc[0]:+.3f}"
                for h in HORIZON_NAMES
            ), flush=True)
    pd.DataFrame(all_predictions).to_csv(output / "rolling_predictions.csv", index=False, float_format="%.10g")
    summary = {}
    result_table = pd.DataFrame(all_results)
    for horizon, part in result_table.groupby("horizon"):
        summary[horizon] = {
            "folds": len(FOLDS), "seeds_per_fold": len(SEEDS),
            "mean_difference_model_minus_persistence": float(part.difference_model_minus_persistence.mean()),
            "fold_seed_pairs_beating_persistence": int((part.difference_model_minus_persistence < 0).sum()),
            "fold_mean_differences": {
                str(fold): float(fold_part.difference_model_minus_persistence.mean())
                for fold, fold_part in part.groupby("fold")
            },
        }
    (output / "rolling_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
