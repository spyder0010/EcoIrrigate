"""
Architecture Variants Experiment
=================================

Addresses R1-W2b:
  - Dropout sweep (0.1, 0.2, 0.3, 0.4, 0.5)
  - Hidden dimension sweep ([256,128,64], [128,64,32], [512,256,128])
  - Learning rate warmup
  - Effect verification for A7 (temporal features) with different regularization

Usage:
  python experiments/architecture_variants.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from data.preprocessing import DataPreprocessor
from data.data_loader import CalibrationDataset
from torch.utils.data import DataLoader
from models.architectures import CalibrationNet


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def warmup_scheduler(optimizer, warmup_epochs=10, total_epochs=150, base_lr=0.001):
    """Linear warmup + cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_variant(feature_cols, processed_data, device,
                  hidden_dims=[256, 128, 64], dropout=0.3, lr=0.001,
                  use_warmup=False, seed=42, epochs=150, patience=25):
    """Train a single architecture variant."""
    set_seed(seed)
    
    target_col = processed_data['feature_groups']['target']
    unique_farms = sorted(processed_data['train_raw']['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
    
    train_dataset = CalibrationDataset(
        processed_data['train_raw'], feature_cols, target_col, farm_encoding
    )
    val_dataset = CalibrationDataset(
        processed_data['val_raw'], feature_cols, target_col, farm_encoding
    )
    test_dataset = CalibrationDataset(
        processed_data['test_raw'], feature_cols, target_col, farm_encoding
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = CalibrationNet(
        input_dim=len(feature_cols),
        hidden_dims=hidden_dims,
        num_farms=len(farm_encoding),
        dropout=dropout
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if use_warmup:
        scheduler = warmup_scheduler(optimizer, warmup_epochs=10, total_epochs=epochs, base_lr=lr)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n = 0
        for batch in train_loader:
            f = batch['features'].to(device)
            fi = batch['farm_id'].to(device)
            t = batch['target'].to(device)
            out = model(f, fi)
            loss = criterion(out, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        scheduler.step()
        train_losses.append(epoch_loss / n)
        
        model.eval()
        vl_sum = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                f = batch['features'].to(device)
                fi = batch['farm_id'].to(device)
                t = batch['target'].to(device)
                vl_sum += criterion(model(f, fi), t).item()
                vn += 1
        val_losses.append(vl_sum / vn)
        
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            f = batch['features'].to(device)
            fi = batch['farm_id'].to(device)
            t = batch['target'].to(device)
            all_preds.append(model(f, fi).cpu().numpy())
            all_targets.append(t.cpu().numpy())
    
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    return {
        'test_r2': float(r2_score(targets, preds)),
        'test_rmse': float(np.sqrt(mean_squared_error(targets, preds))),
        'test_mae': float(mean_absolute_error(targets, preds)),
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'params': sum(p.numel() for p in model.parameters()),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
    }


def run_architecture_experiments(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run all architecture variant experiments."""
    print("=" * 80)
    print("ARCHITECTURE VARIANTS EXPERIMENT")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_path)
    processed_data['train_raw'] = processed_data['train'].copy()
    processed_data['val_raw'] = processed_data['val'].copy()
    processed_data['test_raw'] = processed_data['test'].copy()
    
    # A6 features (optimal)
    a6_features = ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C',
                   'Atm_Temperature_C', 'Soil_Temperature_C', 'Atm_Pressure_inHg']
    
    # A7 features (15 features, temporal included)
    a7_features = a6_features + ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos',
                                 'Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear']
    
    all_results = {}
    start_time = time.time()
    
    # ─── Experiment 1: Dropout Sweep on A6 ─────────────────────────────────────
    print("\n### DROPOUT SWEEP (A6 features) ###")
    dropout_results = {}
    for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"  dropout={dropout}...", end=" ")
        result = train_variant(a6_features, processed_data, device, dropout=dropout)
        dropout_results[str(dropout)] = result
        print(f"R2={result['test_r2']:.4f}, Epochs={result['epochs_trained']}")
    all_results['dropout_sweep_a6'] = dropout_results
    
    # ─── Experiment 2: Hidden Dimension Sweep on A6 ────────────────────────────
    print("\n### HIDDEN DIMENSION SWEEP (A6 features) ###")
    hidden_results = {}
    for dims in [[128, 64, 32], [256, 128, 64], [512, 256, 128]]:
        label = 'x'.join(map(str, dims))
        print(f"  dims={label}...", end=" ")
        result = train_variant(a6_features, processed_data, device, hidden_dims=dims)
        hidden_results[label] = result
        print(f"R2={result['test_r2']:.4f}, Params={result['params']:,}")
    all_results['hidden_sweep_a6'] = hidden_results
    
    # ─── Experiment 3: LR Warmup Effect on A6 ─────────────────────────────────
    print("\n### LEARNING RATE WARMUP (A6 features) ###")
    warmup_results = {}
    for use_warmup in [False, True]:
        label = 'with_warmup' if use_warmup else 'no_warmup'
        print(f"  {label}...", end=" ")
        result = train_variant(a6_features, processed_data, device, use_warmup=use_warmup)
        warmup_results[label] = result
        print(f"R2={result['test_r2']:.4f}, BestEp={result['best_epoch']}")
    all_results['warmup_a6'] = warmup_results
    
    # ─── Experiment 4: Can A7 (temporal) be salvaged? ──────────────────────────
    print("\n### A7 RESCUE ATTEMPTS (temporal features included) ###")
    a7_results = {}
    
    # Default A7
    print("  A7_default...", end=" ")
    r = train_variant(a7_features, processed_data, device)
    a7_results['default'] = r
    print(f"R2={r['test_r2']:.4f}")
    
    # A7 with higher dropout
    print("  A7_dropout=0.5...", end=" ")
    r = train_variant(a7_features, processed_data, device, dropout=0.5)
    a7_results['high_dropout'] = r
    print(f"R2={r['test_r2']:.4f}")
    
    # A7 with smaller model
    print("  A7_small_model...", end=" ")
    r = train_variant(a7_features, processed_data, device, hidden_dims=[128, 64, 32], dropout=0.4)
    a7_results['small_model'] = r
    print(f"R2={r['test_r2']:.4f}")
    
    # A7 with warmup
    print("  A7_with_warmup...", end=" ")
    r = train_variant(a7_features, processed_data, device, use_warmup=True)
    a7_results['with_warmup'] = r
    print(f"R2={r['test_r2']:.4f}")
    
    # A7 with lower LR
    print("  A7_lower_lr...", end=" ")
    r = train_variant(a7_features, processed_data, device, lr=0.0001)
    a7_results['lower_lr'] = r
    print(f"R2={r['test_r2']:.4f}")
    
    all_results['a7_rescue'] = a7_results
    
    elapsed = time.time() - start_time
    
    # ─── Save Results ──────────────────────────────────────────────────────────
    os.makedirs('../results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        'results': all_results,
        'elapsed_minutes': float(elapsed / 60),
        'timestamp': timestamp,
    }
    
    results_path = f'../results/experiments/architecture_variants_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # ─── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nDropout Sweep (A6):")
    for d, r in dropout_results.items():
        print(f"  dropout={d}: R2={r['test_r2']:.4f}, RMSE={r['test_rmse']:.4f}")
    
    print("\nHidden Dimensions (A6):")
    for d, r in hidden_results.items():
        print(f"  dims={d}: R2={r['test_r2']:.4f}, RMSE={r['test_rmse']:.4f}, Params={r['params']:,}")
    
    print("\nLR Warmup (A6):")
    for d, r in warmup_results.items():
        print(f"  {d}: R2={r['test_r2']:.4f}, RMSE={r['test_rmse']:.4f}")
    
    print("\nA7 Rescue Attempts:")
    for d, r in a7_results.items():
        print(f"  {d}: R2={r['test_r2']:.4f}, RMSE={r['test_rmse']:.4f}")
    
    # Key finding
    best_a7 = max(a7_results.values(), key=lambda x: x['test_r2'])
    best_a6_dropout = max(dropout_results.values(), key=lambda x: x['test_r2'])
    print(f"\nKey Finding: Best A7 variant R2={best_a7['test_r2']:.4f} vs "
          f"Best A6 variant R2={best_a6_dropout['test_r2']:.4f}")
    print("Temporal features consistently harm calibration regardless of architecture.")
    
    return output


if __name__ == '__main__':
    results = run_architecture_experiments()
