"""
DL Architecture Comparison: TCN vs ConvLSTM vs Transformer vs BiLSTM-Attention
===============================================================================

Benchmarks four deep learning architectures on BOTH:
  1. Calibration task (ADC → Moisture %, point-wise)
  2. Forecasting task (24h history → multi-horizon prediction)

Usage:
    $env:PYTHONIOENCODING='utf-8'; python experiments/dl_architecture_comparison.py
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import DataPreprocessor
from data.data_loader import (CalibrationDataset, ForecastingDataset,
                               create_data_loaders)
from torch.utils.data import DataLoader
from models.architectures import CalibrationNet, ForecastingNet
from models.baseline_architectures import (
    TCNCalibrator, TCNForecaster,
    ConvLSTMCalibrator, ConvLSTMForecaster,
    TransformerCalibrator, TransformerForecaster,
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_calibration_model(model, train_loader, val_loader, test_loader,
                            device, epochs=100, lr=0.001, patience=20):
    """Train and evaluate a calibration model."""
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            features = batch['features'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets = batch['target'].to(device)
            outputs = model(features, farm_ids)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0; n = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                farm_ids = batch['farm_id'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features, farm_ids)
                val_loss += criterion(outputs, targets).item(); n += 1
        val_loss /= n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    # Evaluate on test set
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets = batch['target'].to(device)
            outputs = model(features, farm_ids)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets_arr = np.concatenate(all_targets).flatten()

    return {
        'test_r2': float(r2_score(targets_arr, preds)),
        'test_rmse': float(np.sqrt(mean_squared_error(targets_arr, preds))),
        'test_mae': float(mean_absolute_error(targets_arr, preds)),
        'epochs_trained': epoch,
        'params': sum(p.numel() for p in model.parameters()),
    }


def train_forecasting_model(model, train_loader, val_loader, test_loader,
                             device, epochs=100, lr=0.001, patience=20):
    """Train and evaluate a forecasting model."""
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets = batch['targets'].to(device)
            outputs, _ = model(sequences, farm_ids)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0; n = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                farm_ids = batch['farm_id'].to(device)
                targets = batch['targets'].to(device)
                outputs, _ = model(sequences, farm_ids)
                val_loss += criterion(outputs, targets).item(); n += 1
        val_loss /= n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    # Evaluate per horizon
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets = batch['targets'].to(device)
            outputs, _ = model(sequences, farm_ids)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    horizon_names = ['1h', '6h', '12h', '24h']
    results = {'epochs_trained': epoch,
               'params': sum(p.numel() for p in model.parameters())}
    for i, h_name in enumerate(horizon_names):
        p, t = preds[:, i], targets_arr[:, i]
        results[f'{h_name}_r2'] = float(r2_score(t, p))
        results[f'{h_name}_rmse'] = float(np.sqrt(mean_squared_error(t, p)))
        results[f'{h_name}_mae'] = float(mean_absolute_error(t, p))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    print("=" * 80)
    print("DL ARCHITECTURE COMPARISON")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    preprocessor = DataPreprocessor(scaling_method='standard')
    processed = preprocessor.preprocess_pipeline(filepath=data_path)

    feature_groups = processed['feature_groups']
    target_col = feature_groups['target']
    calib_features = feature_groups['calibration_features'] + feature_groups['temporal_features']
    seq_features = calib_features.copy()
    if target_col not in seq_features:
        seq_features = [target_col] + seq_features

    unique_farms = sorted(processed['train_raw']['Farm_ID'].unique())
    farm_encoding = {f: i for i, f in enumerate(unique_farms)}
    num_farms = len(farm_encoding)

    # ── Calibration loaders ──────────────────────────────────────────────
    train_calib = CalibrationDataset(processed['train_raw'], calib_features, target_col, farm_encoding)
    val_calib   = CalibrationDataset(processed['val_raw'],   calib_features, target_col, farm_encoding)
    test_calib  = CalibrationDataset(processed['test_raw'],  calib_features, target_col, farm_encoding)
    train_cl = DataLoader(train_calib, 64, shuffle=True)
    val_cl   = DataLoader(val_calib,   64, shuffle=False)
    test_cl  = DataLoader(test_calib,  64, shuffle=False)

    # ── Forecasting loaders ──────────────────────────────────────────────
    train_fc = ForecastingDataset(processed['train_raw'], seq_features, target_col,
                                  sequence_length=96, forecast_horizons=[4,24,48,96],
                                  farm_encoding=farm_encoding)
    val_fc   = ForecastingDataset(processed['val_raw'],   seq_features, target_col,
                                  sequence_length=96, forecast_horizons=[4,24,48,96],
                                  farm_encoding=farm_encoding)
    test_fc  = ForecastingDataset(processed['test_raw'],  seq_features, target_col,
                                  sequence_length=96, forecast_horizons=[4,24,48,96],
                                  farm_encoding=farm_encoding)
    train_fl = DataLoader(train_fc, 32, shuffle=True)
    val_fl   = DataLoader(val_fc,   32, shuffle=False)
    test_fl  = DataLoader(test_fc,  32, shuffle=False)

    input_dim_calib = len(calib_features)
    input_dim_seq   = len(seq_features)

    # ── Architecture registry ────────────────────────────────────────────
    architectures = {
        'BiLSTM-Attention': {
            'calib': lambda: CalibrationNet(input_dim_calib, [256,128,64],
                                           num_farms, dropout=0.3).to(device),
            'forecast': lambda: ForecastingNet(input_dim_seq, 128, 2, 4,
                                              num_farms, dropout=0.3).to(device),
        },
        'TCN': {
            'calib': lambda: TCNCalibrator(input_dim_calib, [256,128,64],
                                          num_farms, dropout=0.3).to(device),
            'forecast': lambda: TCNForecaster(input_dim_seq, 128, 4, 3, 4,
                                             num_farms, dropout=0.3).to(device),
        },
        'ConvLSTM': {
            'calib': lambda: ConvLSTMCalibrator(input_dim_calib, [256,128,64],
                                               num_farms, dropout=0.3).to(device),
            'forecast': lambda: ConvLSTMForecaster(input_dim_seq, 128, 2, 4,
                                                   num_farms, dropout=0.3).to(device),
        },
        'Transformer': {
            'calib': lambda: TransformerCalibrator(input_dim_calib, [256,128,64],
                                                  num_farms, dropout=0.3).to(device),
            'forecast': lambda: TransformerForecaster(input_dim_seq, 128, 4, 2, 4,
                                                     num_farms, dropout=0.3).to(device),
        },
    }

    all_results = {}

    for arch_name, builders in architectures.items():
        print(f"\n{'='*60}")
        print(f"  {arch_name}")
        print(f"{'='*60}")

        # ── CALIBRATION ──
        print(f"  [Calibration] Training...")
        t0 = time.time()
        calib_model = builders['calib']()
        calib_res = train_calibration_model(calib_model, train_cl, val_cl, test_cl,
                                            device, epochs=100, patience=20)
        calib_res['train_time_s'] = round(time.time() - t0, 1)
        print(f"    R2={calib_res['test_r2']:.4f}  RMSE={calib_res['test_rmse']:.4f}  "
              f"MAE={calib_res['test_mae']:.4f}  epochs={calib_res['epochs_trained']}  "
              f"time={calib_res['train_time_s']}s")

        # ── FORECASTING ──
        print(f"  [Forecasting] Training...")
        t0 = time.time()
        fc_model = builders['forecast']()
        fc_res = train_forecasting_model(fc_model, train_fl, val_fl, test_fl,
                                         device, epochs=100, patience=20)
        fc_res['train_time_s'] = round(time.time() - t0, 1)
        for h in ['1h','6h','12h','24h']:
            print(f"    {h}: R2={fc_res[f'{h}_r2']:.4f}  "
                  f"RMSE={fc_res[f'{h}_rmse']:.4f}  MAE={fc_res[f'{h}_mae']:.4f}")

        all_results[arch_name] = {'calibration': calib_res, 'forecasting': fc_res}

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs('results/experiments', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f'results/experiments/dl_architecture_comparison_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print(f"{'Architecture':<20} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'Params':>10}")
    print("-" * 56)
    for name, res in all_results.items():
        c = res['calibration']
        print(f"{name:<20} {c['test_r2']:>8.4f} {c['test_rmse']:>8.4f} "
              f"{c['test_mae']:>8.4f} {c['params']:>10}")

    print("\nFORECASTING SUMMARY (RMSE)")
    print(f"{'Architecture':<20} {'1h':>8} {'6h':>8} {'12h':>8} {'24h':>8}")
    print("-" * 52)
    for name, res in all_results.items():
        fc = res['forecasting']
        print(f"{name:<20} {fc['1h_rmse']:>8.4f} {fc['6h_rmse']:>8.4f} "
              f"{fc['12h_rmse']:>8.4f} {fc['24h_rmse']:>8.4f}")

    return all_results


if __name__ == '__main__':
    run_comparison()
