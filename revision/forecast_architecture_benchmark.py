"""Common-origin neural benchmark on the supplied EcoIrrigate VWC series.

This is an exploratory model comparison, not independent field validation.
All candidate models see the same historical window and current observation,
use one chronological split, and score on the same held-out timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from analysis_paths import SOURCE_CSV

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = SOURCE_CSV
OUT = Path(__file__).resolve().parent / "architecture_benchmark"
FEATURES = [
    "Volumetric_Moisture_Pct",
    "Raw_Capacitive_ADC",
    "Sensor_Voltage_V",
    "Sensor_Board_Temperature_C",
    "Atm_Temperature_C",
    "Soil_Temperature_C",
    "Atm_Pressure_inHg",
    "Hour_sin",
    "Hour_cos",
]
HORIZONS = [4, 24, 48, 96]
HORIZON_NAMES = ["1h", "6h", "12h", "24h"]
WINDOW = 96


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OriginDataset(Dataset):
    def __init__(self, arrays: list[np.ndarray], origins: list[tuple[int, int]]):
        self.arrays = arrays
        self.origins = origins

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int):
        farm, t = self.origins[index]
        arr = self.arrays[farm]
        return (
            torch.from_numpy(arr[t - WINDOW:t]),
            torch.from_numpy(arr[t, [1, 2]].copy()),
            torch.from_numpy(arr[t, [3, 4, 5, 6, 7, 8]].copy()),
            torch.tensor(arr[t, 0], dtype=torch.float32),
            torch.tensor([arr[t + h, 0] for h in HORIZONS], dtype=torch.float32),
            torch.tensor(farm, dtype=torch.long),
        )


def make_data(train_stride: int):
    df = pd.read_csv(SOURCE, parse_dates=["Timestamp"])
    frames = [f.sort_values("Timestamp").reset_index(drop=True) for _, f in df.groupby("Farm_ID", sort=True)]
    if len(frames) != 2:
        raise ValueError("This benchmark expects the two supplied site IDs")
    for frame in frames:
        if not frame.Timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
            raise ValueError("Irregular timestamps; explicit missing-data policy required")
    boundaries = []
    for frame in frames:
        train_end = int(len(frame) * 0.70)
        val_end = train_end + int(len(frame) * 0.15)
        boundaries.append((train_end, val_end))
    train_raw = pd.concat([f.iloc[:b[0]][FEATURES] for f, b in zip(frames, boundaries)], ignore_index=True)
    scaler = StandardScaler().fit(train_raw.to_numpy(dtype=np.float32))
    arrays = [scaler.transform(f[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32) for f in frames]
    origins = {"train": [], "validation": [], "test": []}
    for farm, (frame, (train_end, val_end)) in enumerate(zip(frames, boundaries)):
        definitions = [
            ("train", WINDOW, train_end, train_stride),
            ("validation", train_end, val_end, 4),
            ("test", val_end, len(frame), 1),
        ]
        for split, start, end, stride in definitions:
            origins[split].extend((farm, t) for t in range(start, end - max(HORIZONS), stride))
    return frames, scaler, arrays, origins


class CausalBlock(nn.Module):
    """Left-padded convolution block over observed history.

    GroupNorm uses the whole supplied history window, so intermediate outputs
    are not strictly causal within that window. The model takes only the final
    context, and no input later than the forecast origin enters the window.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.pad = dilation * 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = nn.functional.pad(x, (self.pad, 0))
        return self.act(self.norm(self.conv(z)) + x)


class ForecastModel(nn.Module):
    def __init__(self, encoder: str, residual: bool, auxiliary: bool, hidden: int = 64):
        super().__init__()
        self.encoder_type = encoder
        self.residual = residual
        self.auxiliary = auxiliary
        self.farm_embedding = nn.Embedding(2, 4)
        if encoder == "tcn":
            self.input_projection = nn.Conv1d(len(FEATURES), hidden, 1)
            self.sequence_encoder = nn.Sequential(*(CausalBlock(hidden, d) for d in (1, 2, 4, 8)))
            context_dim = hidden
        elif encoder in ("gru", "bilstm"):
            self.sequence_encoder = (
                nn.GRU(len(FEATURES), hidden, batch_first=True)
                if encoder == "gru"
                else nn.LSTM(len(FEATURES), hidden // 2, batch_first=True, bidirectional=True)
            )
            context_dim = hidden
        else:
            raise ValueError(encoder)
        self.forecast_head = nn.Sequential(
            nn.Linear(context_dim + 1 + 6 + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(HORIZONS)),
        )
        if auxiliary:
            # Current A2 sensor inputs plus the historical shared representation.
            # The head cannot see current VWC directly.
            self.current_value_head = nn.Sequential(
                nn.Linear(context_dim + 2 + 4, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, history, current_a2, current_env, current_vwc, farm):
        if self.encoder_type == "tcn":
            z = self.input_projection(history.transpose(1, 2))
            context = self.sequence_encoder(z)[:, :, -1]
        else:
            _, state = self.sequence_encoder(history)
            if self.encoder_type == "gru":
                context = state[-1]
            else:
                h, _ = state
                context = torch.cat([h[-2], h[-1]], dim=1)
        farm_vec = self.farm_embedding(farm)
        forecast_input = torch.cat([context, current_vwc[:, None], current_env, farm_vec], dim=1)
        output = self.forecast_head(forecast_input)
        if self.residual:
            output = current_vwc[:, None] + output
        current_prediction = None
        if self.auxiliary:
            current_prediction = self.current_value_head(torch.cat([context, current_a2, farm_vec], dim=1)).squeeze(1)
        return output, current_prediction


@dataclass(frozen=True)
class Variant:
    name: str
    encoder: str
    residual: bool
    auxiliary: bool


VARIANTS = [
    Variant("gru_direct", "gru", False, False),
    Variant("gru_residual", "gru", True, False),
    Variant("bilstm_residual", "bilstm", True, False),
    Variant("shared_gru_residual", "gru", True, True),
    Variant("shared_tcn_residual", "tcn", True, True),
    Variant("shared_bilstm_residual", "bilstm", True, True),
]


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=float)
    count = 0
    for history, current_a2, current_env, current_vwc, targets, farm in loader:
        batch = [v.to(device, non_blocking=True) for v in (history, current_a2, current_env, current_vwc, targets, farm)]
        with torch.set_grad_enabled(training):
            forecast, current_prediction = model(batch[0], batch[1], batch[2], batch[3], batch[5])
            forecast_loss = nn.functional.mse_loss(forecast, batch[4])
            current_loss = (
                nn.functional.mse_loss(current_prediction, batch[3])
                if current_prediction is not None else torch.zeros((), device=device)
            )
            total_loss = forecast_loss + current_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = len(batch[3])
        totals += np.array([total_loss.item(), forecast_loss.item(), current_loss.item()]) * n
        count += n
    return (totals / count).tolist()


def infer(model, loader, device):
    model.eval()
    preds, actuals, currents, aux = [], [], [], []
    with torch.no_grad():
        for history, current_a2, current_env, current_vwc, targets, farm in loader:
            forecast, current_prediction = model(
                history.to(device), current_a2.to(device), current_env.to(device),
                current_vwc.to(device), farm.to(device)
            )
            preds.append(forecast.cpu().numpy())
            actuals.append(targets.numpy())
            currents.append(current_vwc.numpy())
            if current_prediction is not None:
                aux.append(current_prediction.cpu().numpy())
    return np.concatenate(preds), np.concatenate(actuals), np.concatenate(currents), (np.concatenate(aux) if aux else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="*", default=[v.name for v in VARIANTS])
    parser.add_argument("--output-subdir", type=str, default="")
    args = parser.parse_args()
    if args.epochs < 1 or args.train_stride < 1:
        raise ValueError("epochs and train stride must be positive")
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = OUT / args.output_subdir if args.output_subdir else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, scaler, arrays, origins = make_data(args.train_stride)
    datasets = {split: OriginDataset(arrays, rows) for split, rows in origins.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda"),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda"),
    }
    report = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "feature_columns": FEATURES,
        "history": "t-96 through t-1; current A2 and environmental features at t; current observed VWC at t anchors residual forecast",
        "target": "VWC[t+h] for h in 4,24,48,96 steps; same origins for every horizon",
        "selection": "lowest validation forecast MSE, early stopping; test scored once per specified variant",
        "limits": "supplied VWC provenance unverified; architecture search after historical test inspection is exploratory",
        "device": str(device),
        "seed": args.seed,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "train_stride": args.train_stride,
        "n_origins": {split: len(rows) for split, rows in origins.items()},
        "variants": {},
    }
    all_ledger = []
    for variant in VARIANTS:
        if variant.name not in args.variants:
            continue
        seed_all(args.seed)
        model = ForecastModel(variant.encoder, variant.residual, variant.auxiliary).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        best_val = float("inf")
        wait = 0
        best_state = None
        history_log = []
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, loaders["train"], device, optimizer)
            val_loss = run_epoch(model, loaders["validation"], device)
            history_log.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
            print(f"{variant.name} epoch {epoch}: train forecast MSE={train_loss[1]:.4f} val forecast MSE={val_loss[1]:.4f}", flush=True)
            if val_loss[1] < best_val - 1e-6:
                best_val = val_loss[1]
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            else:
                wait += 1
                if wait >= args.patience:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        checkpoint = output_dir / f"{variant.name}_seed{args.seed}.pt"
        torch.save({"model_state_dict": best_state, "variant": variant.__dict__, "best_epoch": best_epoch, "config": vars(args)}, checkpoint)
        predictions, truth, current, current_prediction = infer(model, loaders["test"], device)
        vwc_scale = float(scaler.scale_[0])
        vwc_mean = float(scaler.mean_[0])
        predictions = predictions * vwc_scale + vwc_mean
        truth = truth * vwc_scale + vwc_mean
        current = current * vwc_scale + vwc_mean
        current_rmse = None
        if current_prediction is not None:
            current_rmse = float(np.sqrt(mean_squared_error(current, current_prediction * vwc_scale + vwc_mean)))
        horizon_metrics = {}
        for j, horizon in enumerate(HORIZON_NAMES):
            horizon_metrics[horizon] = {
                "model_rmse_vwc_percentage_points": float(np.sqrt(mean_squared_error(truth[:, j], predictions[:, j]))),
                "persistence_rmse_vwc_percentage_points": float(np.sqrt(mean_squared_error(truth[:, j], current))),
                "model_mae_vwc_percentage_points": float(mean_absolute_error(truth[:, j], predictions[:, j])),
                "model_r2": float(r2_score(truth[:, j], predictions[:, j])),
            }
        report["variants"][variant.name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch,
            "validation_forecast_mse_scaled": best_val,
            "current_value_head_rmse_vwc_percentage_points": current_rmse,
            "horizons": horizon_metrics,
            "training_log": history_log,
            "checkpoint": str(checkpoint),
        }
        for k, (farm, t) in enumerate(origins["test"]):
            for j, horizon in enumerate(HORIZON_NAMES):
                all_ledger.append({
                    "variant": variant.name,
                    "farm": frames[farm].Farm_ID.iloc[0],
                    "horizon": horizon,
                    "origin_timestamp": frames[farm].Timestamp.iloc[t],
                    "target_timestamp": frames[farm].Timestamp.iloc[t + HORIZONS[j]],
                    "target_vwc_pct": truth[k, j],
                    "current_vwc_pct": current[k],
                    "model_vwc_pct": predictions[k, j],
                })
        (output_dir / "architecture_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame(all_ledger).to_csv(output_dir / "architecture_predictions.csv", index=False, float_format="%.10g")
    print(json.dumps({name: value["horizons"] for name, value in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
