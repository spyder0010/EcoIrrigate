"""
Comprehensive Ablation Study with Training Curves & Statistical Rigor
=====================================================================

Addresses:
  R1-W2: Training curves for all configs, alternative architectures
  R1-W2c: Overfitting analysis with held-out data
  R3-W3: Repeated runs (5 seeds) for confidence intervals
  R3-W5: Permutation test for non-monotonic ablation

Usage:
  python experiments/comprehensive_ablation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from sklearn.metrics import mean_absolute_error
from scipy import stats
from data.preprocessing import DataPreprocessor
from data.data_loader import CalibrationDataset
from torch.utils.data import DataLoader
from models.architectures import CalibrationNet


# ─── Ablation Configurations ───────────────────────────────────────────────────
ABLATION_CONFIGS = {
    'A1_ADC_Only': ['Raw_Capacitive_ADC'],
    'A2_Plus_Voltage': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V'],
    'A3_Plus_BoardTemp': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C'],
    'A4_Plus_AtmTemp': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C',
                        'Atm_Temperature_C'],
    'A5_Plus_SoilTemp': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C',
                         'Atm_Temperature_C', 'Soil_Temperature_C'],
    'A6_Plus_Pressure': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C',
                         'Atm_Temperature_C', 'Soil_Temperature_C', 'Atm_Pressure_inHg'],
    'A7_Full_Model': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C',
                      'Atm_Temperature_C', 'Soil_Temperature_C', 'Atm_Pressure_inHg',
                      'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos',
                      'Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear'],
}

SEEDS = [42, 123, 456, 789, 2025]  # 5 random seeds for CIs
EPOCHS = 150
PATIENCE = 25
BATCH_SIZE = 64
LR = 0.001
HIDDEN_DIMS = [256, 128, 64]


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_config(config_name, feature_cols, processed_data, device,
                        seed=42, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        lr=LR, patience=PATIENCE, hidden_dims=HIDDEN_DIMS,
                        dropout=0.3):
    """Train a single ablation configuration and return full training history."""
    set_seed(seed)
    
    target_col = processed_data['feature_groups']['target']
    unique_farms = sorted(processed_data['train_raw']['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
    
    # Create datasets
    train_dataset = CalibrationDataset(
        processed_data['train_raw'], feature_cols, target_col, farm_encoding
    )
    val_dataset = CalibrationDataset(
        processed_data['val_raw'], feature_cols, target_col, farm_encoding
    )
    test_dataset = CalibrationDataset(
        processed_data['test_raw'], feature_cols, target_col, farm_encoding
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = len(feature_cols)
    model = CalibrationNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_farms=len(farm_encoding),
        dropout=dropout
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    # Training loop with full history
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_train_loss = 0
        epoch_train_preds = []
        epoch_train_targets = []
        n_batches = 0
        
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
            
            epoch_train_loss += loss.item()
            epoch_train_preds.append(outputs.detach().cpu().numpy())
            epoch_train_targets.append(targets.detach().cpu().numpy())
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_train_loss / n_batches
        
        # Compute train R²
        train_preds_all = np.concatenate(epoch_train_preds).flatten()
        train_targets_all = np.concatenate(epoch_train_targets).flatten()
        train_r2 = r2_score(train_targets_all, train_preds_all)
        
        # Validate
        model.eval()
        epoch_val_loss = 0
        epoch_val_preds = []
        epoch_val_targets = []
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                farm_ids = batch['farm_id'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features, farm_ids)
                epoch_val_loss += criterion(outputs, targets).item()
                epoch_val_preds.append(outputs.cpu().numpy())
                epoch_val_targets.append(targets.cpu().numpy())
                n_val += 1
        
        avg_val_loss = epoch_val_loss / n_val
        val_preds_all = np.concatenate(epoch_val_preds).flatten()
        val_targets_all = np.concatenate(epoch_val_targets).flatten()
        val_r2 = r2_score(val_targets_all, val_preds_all)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model and evaluate on test set
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
    targets = np.concatenate(all_targets).flatten()
    
    test_r2 = r2_score(targets, preds)
    test_rmse = np.sqrt(mean_squared_error(targets, preds))
    test_mae = mean_absolute_error(targets, preds)
    
    # Check for overfitting: gap between train and val R²
    final_train_r2 = train_r2s[best_epoch - 1] if best_epoch <= len(train_r2s) else train_r2s[-1]
    final_val_r2 = val_r2s[best_epoch - 1] if best_epoch <= len(val_r2s) else val_r2s[-1]
    overfit_gap = final_train_r2 - final_val_r2
    
    return {
        'config': config_name,
        'seed': seed,
        'num_features': len(feature_cols),
        'features': feature_cols,
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        'params': sum(p.numel() for p in model.parameters()),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_r2_history': [float(x) for x in train_r2s],
        'val_r2_history': [float(x) for x in val_r2s],
        'final_train_r2': float(final_train_r2),
        'final_val_r2': float(final_val_r2),
        'overfit_gap': float(overfit_gap),
        'test_predictions': preds.tolist(),
        'test_actuals': targets.tolist(),
    }


def run_multi_seed_ablation(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run complete ablation study with 5 seeds for all 7 configurations."""
    print("=" * 80)
    print("COMPREHENSIVE ABLATION STUDY (5 seeds x 7 configs = 35 runs)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_path)
    
    # Store raw data for later use
    processed_data['train_raw'] = processed_data['train'].copy()
    processed_data['val_raw'] = processed_data['val'].copy()
    processed_data['test_raw'] = processed_data['test'].copy()
    
    all_results = {}
    summary_stats = {}
    
    total_runs = len(ABLATION_CONFIGS) * len(SEEDS)
    run_idx = 0
    start_time = time.time()
    
    for config_name, feature_cols in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name} ({len(feature_cols)} features)")
        print(f"{'='*60}")
        
        config_results = []
        
        for seed in SEEDS:
            run_idx += 1
            print(f"\n  Seed {seed} [{run_idx}/{total_runs}]...")
            
            result = train_single_config(
                config_name, feature_cols, processed_data, device, seed=seed
            )
            config_results.append(result)
            
            print(f"    R2={result['test_r2']:.4f}, RMSE={result['test_rmse']:.4f}, "
                  f"Epochs={result['epochs_trained']}, BestEp={result['best_epoch']}, "
                  f"OverfitGap={result['overfit_gap']:.4f}")
        
        all_results[config_name] = config_results
        
        # Compute summary statistics
        r2_vals = [r['test_r2'] for r in config_results]
        rmse_vals = [r['test_rmse'] for r in config_results]
        mae_vals = [r['test_mae'] for r in config_results]
        overfit_gaps = [r['overfit_gap'] for r in config_results]
        
        summary_stats[config_name] = {
            'r2_mean': float(np.mean(r2_vals)),
            'r2_std': float(np.std(r2_vals)),
            'r2_ci_lower': float(np.mean(r2_vals) - 1.96 * np.std(r2_vals) / np.sqrt(len(r2_vals))),
            'r2_ci_upper': float(np.mean(r2_vals) + 1.96 * np.std(r2_vals) / np.sqrt(len(r2_vals))),
            'rmse_mean': float(np.mean(rmse_vals)),
            'rmse_std': float(np.std(rmse_vals)),
            'mae_mean': float(np.mean(mae_vals)),
            'mae_std': float(np.std(mae_vals)),
            'overfit_gap_mean': float(np.mean(overfit_gaps)),
            'overfit_gap_std': float(np.std(overfit_gaps)),
            'individual_r2': r2_vals,
            'individual_rmse': rmse_vals,
        }
        
        print(f"\n  Summary: R2={np.mean(r2_vals):.4f} +/- {np.std(r2_vals):.4f} "
              f"({summary_stats[config_name]['r2_ci_lower']:.4f}, {summary_stats[config_name]['r2_ci_upper']:.4f})")
        print(f"           RMSE={np.mean(rmse_vals):.4f} +/- {np.std(rmse_vals):.4f}")
        print(f"           Overfit Gap={np.mean(overfit_gaps):.4f} +/- {np.std(overfit_gaps):.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n\nTotal elapsed time: {elapsed/60:.1f} minutes")
    
    # ─── Permutation Test (R3-W5) ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PERMUTATION TEST: A6 vs A2 (best vs second-best)")
    print("=" * 60)
    
    a6_r2 = summary_stats['A6_Plus_Pressure']['individual_r2']
    a2_r2 = summary_stats['A2_Plus_Voltage']['individual_r2']
    
    observed_diff = np.mean(a6_r2) - np.mean(a2_r2)
    combined = a6_r2 + a2_r2
    n_perms = 10000
    perm_diffs = []
    
    for _ in range(n_perms):
        perm = np.random.permutation(combined)
        perm_diff = np.mean(perm[:5]) - np.mean(perm[5:])
        perm_diffs.append(perm_diff)
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    permutation_result = {
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'n_permutations': n_perms,
        'a6_r2_values': [float(v) for v in a6_r2],
        'a2_r2_values': [float(v) for v in a2_r2],
        'significant': bool(p_value < 0.05),
    }
    
    print(f"  Observed A6-A2 R2 diff: {observed_diff:.6f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {p_value < 0.05}")
    
    # ─── Save Results ────────────────────────────────────────────────────────
    os.makedirs('../results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training histories (first seed only, for training curve plots)
    training_histories = {}
    for config_name, config_results in all_results.items():
        seed42_result = [r for r in config_results if r['seed'] == 42][0]
        training_histories[config_name] = {
            'train_losses': seed42_result['train_losses'],
            'val_losses': seed42_result['val_losses'],
            'train_r2_history': seed42_result['train_r2_history'],
            'val_r2_history': seed42_result['val_r2_history'],
            'epochs_trained': seed42_result['epochs_trained'],
            'best_epoch': seed42_result['best_epoch'],
        }
    
    # Save summary results  
    output = {
        'summary_stats': summary_stats,
        'permutation_test': permutation_result,
        'training_histories': training_histories,
        'config': {
            'seeds': SEEDS,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'hidden_dims': HIDDEN_DIMS,
            'dropout': 0.3,
        },
        'timestamp': timestamp,
        'elapsed_minutes': float(elapsed / 60),
    }
    
    results_path = f'../results/experiments/comprehensive_ablation_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_path}")
    
    # ─── Generate Training Curve Plots (R1-W2) ───────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING TRAINING CURVE PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Training Curves: All Ablation Configurations (seed=42)', fontsize=14)
    
    config_names = list(ABLATION_CONFIGS.keys())
    labels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    
    for i, (config_name, label) in enumerate(zip(config_names, labels)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        h = training_histories[config_name]
        epochs_range = range(1, len(h['train_losses']) + 1)
        
        ax.plot(epochs_range, h['train_losses'], 'b-', alpha=0.7, label='Train Loss')
        ax.plot(epochs_range, h['val_losses'], 'r-', alpha=0.7, label='Val Loss')
        ax.axvline(x=h['best_epoch'], color='g', linestyle='--', alpha=0.5, label=f'Best Ep={h["best_epoch"]}')
        ax.set_title(f'{label} ({len(ABLATION_CONFIGS[config_name])} feat)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes[1, 3].set_visible(False)
    
    plt.tight_layout()
    fig_path = '../manuscript/springer/figures/fig_ablation_training_curves.png'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to {fig_path}")
    
    # R² convergence plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('R² Convergence: Train vs Validation (seed=42)', fontsize=14)
    
    for i, (config_name, label) in enumerate(zip(config_names, labels)):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        h = training_histories[config_name]
        epochs_range = range(1, len(h['train_r2_history']) + 1)
        
        ax.plot(epochs_range, h['train_r2_history'], 'b-', alpha=0.7, label='Train R²')
        ax.plot(epochs_range, h['val_r2_history'], 'r-', alpha=0.7, label='Val R²')
        ax.axvline(x=h['best_epoch'], color='g', linestyle='--', alpha=0.5)
        ax.set_title(f'{label} ({len(ABLATION_CONFIGS[config_name])} feat)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R²')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    axes[1, 3].set_visible(False)
    plt.tight_layout()
    r2_fig_path = '../manuscript/springer/figures/fig_ablation_r2_convergence.png'
    plt.savefig(r2_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  R² convergence saved to {r2_fig_path}")
    
    # Confidence interval bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [summary_stats[c]['r2_mean'] for c in config_names]
    stds = [summary_stats[c]['r2_std'] for c in config_names]
    ci_errors = [1.96 * s / np.sqrt(5) for s in stds]
    
    colors = ['#4a90d9' if l != 'A6' else '#e74c3c' for l in labels]
    bars = ax.bar(labels, means, yerr=ci_errors, capsize=5, color=colors, alpha=0.8)
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Ablation Configuration')
    ax.set_ylabel('Test R² (mean ± 95% CI)')
    ax.set_title('Feature Ablation: R² with 95% Confidence Intervals (5 seeds)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    ci_fig_path = '../manuscript/springer/figures/fig_ablation_confidence_intervals.png'
    plt.savefig(ci_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  CI bar plot saved to {ci_fig_path}")
    
    # ─── Print Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':<8} {'R² Mean':>10} {'R² Std':>10} {'95% CI':>20} {'RMSE':>10} {'Overfit':>10}")
    print("-" * 68)
    
    best_config = None
    best_r2 = -1
    
    for config_name, label in zip(config_names, labels):
        s = summary_stats[config_name]
        ci_str = f"[{s['r2_ci_lower']:.4f}, {s['r2_ci_upper']:.4f}]"
        marker = " <-- BEST" if s['r2_mean'] > best_r2 else ""
        
        if s['r2_mean'] > best_r2:
            best_r2 = s['r2_mean']
            best_config = label
        
        print(f"{label:<8} {s['r2_mean']:>10.4f} {s['r2_std']:>10.4f} {ci_str:>20} "
              f"{s['rmse_mean']:>10.4f} {s['overfit_gap_mean']:>10.4f}{marker}")
    
    print(f"\nBest configuration: {best_config}")
    print(f"Permutation test A6 vs A2: p={permutation_result['p_value']:.4f}")
    
    return output


if __name__ == '__main__':
    results = run_multi_seed_ablation()

