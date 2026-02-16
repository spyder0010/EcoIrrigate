"""
A2 Statistical Recomputation & Figure Generation
=================================================

Recomputes ALL statistics with A2 (ADC + Voltage) as primary model:
1. Train CalibrationNet(A2) and compare against all 9 ML baselines
2. Cohen's d, Wilcoxon, bootstrap CI for A2 (not A6)
3. Generate 4 missing figures
4. Generate training curves for all ablation configs
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
from data.preprocessing import DataPreprocessor
from data.data_loader import CalibrationDataset
from torch.utils.data import DataLoader
from models.architectures import CalibrationNet

# ─── Publication-quality style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_calibration_net(features_list, X_train, y_train, X_val, y_val, 
                          X_test, y_test, farm_train, farm_val, farm_test,
                          farm_encoding, device, seed=42, return_curves=False):
    """Train CalibrationNet with specific feature subset and return predictions."""
    set_seed(seed)
    
    input_dim = len(features_list)
    
    # Create simple datasets
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, farms):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y).unsqueeze(1)
            self.farms = torch.LongTensor(farms)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return {'features': self.X[idx], 'target': self.y[idx], 'farm_id': self.farms[idx]}
    
    train_ds = SimpleDataset(X_train, y_train, farm_train)
    val_ds = SimpleDataset(X_val, y_val, farm_val)
    test_ds = SimpleDataset(X_test, y_test, farm_test)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = CalibrationNet(
        input_dim=input_dim, hidden_dims=[256, 128, 64],
        num_farms=len(farm_encoding), dropout=0.3
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.0001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(1, 151):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets_b = batch['target'].to(device)
            outputs = model(features, farm_ids)
            loss = criterion(outputs, targets_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        
        train_losses.append(epoch_loss / n_batches)
        
        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                f = batch['features'].to(device)
                fi = batch['farm_id'].to(device)
                t = batch['target'].to(device)
                val_loss += criterion(model(f, fi), t).item()
                n += 1
        val_loss /= n
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= 25:
            break
    
    model.load_state_dict(best_state)
    model.eval()
    
    dl_preds = []
    dl_targets = []
    with torch.no_grad():
        for batch in test_loader:
            f = batch['features'].to(device)
            fi = batch['farm_id'].to(device)
            t = batch['target'].to(device)
            out = model(f, fi)
            dl_preds.append(out.cpu().numpy())
            dl_targets.append(t.cpu().numpy())
    
    dl_preds = np.concatenate(dl_preds).flatten()
    dl_targets = np.concatenate(dl_targets).flatten()
    
    result = {
        'preds': dl_preds,
        'targets': dl_targets,
        'residuals': dl_targets - dl_preds,
        'r2': r2_score(dl_targets, dl_preds),
        'rmse': np.sqrt(mean_squared_error(dl_targets, dl_preds)),
        'mae': mean_absolute_error(dl_targets, dl_preds),
    }
    
    if return_curves:
        result['train_losses'] = train_losses
        result['val_losses'] = val_losses
    
    return result


def run_a2_recomputation():
    """Main recomputation pipeline."""
    print("=" * 80)
    print("A2 STATISTICAL RECOMPUTATION")
    print("=" * 80)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    preprocessor = DataPreprocessor()
    data_path = os.path.join('..', 'New_Dataset', 'kolkata_unified_dataset.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'New_Dataset', 'kolkata_unified_dataset.csv')
    processed_data = preprocessor.preprocess_pipeline(data_path)
    
    feature_groups = processed_data['feature_groups']
    all_calib = feature_groups['calibration_features']  # 6 features
    target = feature_groups['target']
    
    # Define ablation configs
    ablation_configs = {
        'A1': ['Raw_Capacitive_ADC'],
        'A2': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V'],
        'A3': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C'],
        'A4': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C', 'Atm_Temperature_C'],
        'A5': ['Raw_Capacitive_ADC', 'Sensor_Voltage_V', 'Sensor_Board_Temperature_C', 'Atm_Temperature_C', 'Soil_Temperature_C'],
        'A6': all_calib,
        'A7': all_calib + feature_groups['temporal_features'][:5],  # Add temporal
    }
    
    # Prepare farm encoding
    unique_farms = sorted(processed_data['train']['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
    
    farm_train = np.array([farm_encoding[f] for f in processed_data['train']['Farm_ID'].values])
    farm_val = np.array([farm_encoding[f] for f in processed_data['val']['Farm_ID'].values])
    farm_test = np.array([farm_encoding[f] for f in processed_data['test']['Farm_ID'].values])
    
    y_train = processed_data['train'][target].values
    y_val = processed_data['val'][target].values
    y_test = processed_data['test'][target].values
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: A2 Multi-seed ablation (5 seeds) 
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 1: MULTI-SEED ABLATION (5 seeds per config)")
    print("=" * 80)
    
    seeds = [42, 123, 456, 789, 2025]
    ablation_results = {}
    all_training_curves = {}
    
    for config_name, features in ablation_configs.items():
        print(f"\n--- Config {config_name}: {features} ---")
        config_r2s = []
        config_rmses = []
        
        for i, seed in enumerate(seeds):
            # Make sure all features exist in data
            available = [f for f in features if f in processed_data['train'].columns]
            if len(available) != len(features):
                print(f"  WARNING: Missing features for {config_name}")
                break
            
            X_train_c = processed_data['train'][available].values
            X_val_c = processed_data['val'][available].values
            X_test_c = processed_data['test'][available].values
            
            return_curves = (i == 0)  # Only save curves for first seed
            result = train_calibration_net(
                available, X_train_c, y_train, X_val_c, y_val,
                X_test_c, y_test, farm_train, farm_val, farm_test,
                farm_encoding, device, seed=seed, return_curves=return_curves
            )
            
            config_r2s.append(result['r2'])
            config_rmses.append(result['rmse'])
            
            if return_curves:
                all_training_curves[config_name] = {
                    'train': result['train_losses'],
                    'val': result['val_losses'],
                }
            
            print(f"  Seed {seed}: R2={result['r2']:.4f}, RMSE={result['rmse']:.4f}")
        
        ablation_results[config_name] = {
            'features': features,
            'r2_mean': np.mean(config_r2s),
            'r2_std': np.std(config_r2s),
            'r2_ci_lower': np.percentile(config_r2s, 2.5),
            'r2_ci_upper': np.percentile(config_r2s, 97.5),
            'rmse_mean': np.mean(config_rmses),
            'rmse_std': np.std(config_rmses),
            'r2_values': config_r2s,
        }
        
        print(f"  MEAN: R2={np.mean(config_r2s):.4f} +/- {np.std(config_r2s):.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: A2 vs Baselines (with A2 features only)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 2: A2 vs ALL 9 ML BASELINES")
    print("=" * 80)
    
    a2_features = ablation_configs['A2']
    X_train_a2 = processed_data['train'][a2_features].values
    X_val_a2 = processed_data['val'][a2_features].values
    X_test_a2 = processed_data['test'][a2_features].values
    
    baselines = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'SVR (RBF)': SVR(kernel='rbf', C=10.0, epsilon=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=10),
        'MLP': MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                           random_state=42, early_stopping=True),
    }
    
    baseline_results = {}
    baseline_residuals = {}
    
    for name, model in baselines.items():
        model.fit(X_train_a2, y_train)
        preds = model.predict(X_test_a2)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        residuals = y_test - preds
        
        baseline_results[name] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
        }
        baseline_residuals[name] = residuals
        print(f"  {name:25s}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    # Train A2 CalibrationNet (seed=42 for primary)
    print("\n  Training CalibrationNet (A2)...")
    a2_result = train_calibration_net(
        a2_features, X_train_a2, y_train, X_val_a2, y_val,
        X_test_a2, y_test, farm_train, farm_val, farm_test,
        farm_encoding, device, seed=42
    )
    print(f"  CalibrationNet (A2): R2={a2_result['r2']:.4f}, RMSE={a2_result['rmse']:.4f}, MAE={a2_result['mae']:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: Statistical Tests (A2 vs all baselines)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 3: STATISTICAL TESTS (A2 vs BASELINES)")
    print("=" * 80)
    
    dl_residuals = a2_result['residuals']
    
    paired_tests = {}
    for name in baselines.keys():
        bl_res = np.abs(baseline_residuals[name])
        dl_res = np.abs(dl_residuals)
        
        diff = bl_res - dl_res
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        stat_w, p_w = stats.wilcoxon(bl_res, dl_res)
        stat_t, p_t = stats.ttest_rel(bl_res, dl_res)
        
        paired_tests[name] = {
            'wilcoxon_stat': float(stat_w),
            'wilcoxon_p': float(p_w),
            'ttest_stat': float(stat_t),
            'ttest_p': float(p_t),
            'cohens_d': float(d),
            'dl_mae': float(np.mean(dl_res)),
            'baseline_mae': float(np.mean(bl_res)),
        }
        
        sig = "***" if p_w < 0.001 else "**" if p_w < 0.01 else "*" if p_w < 0.05 else "ns"
        d_label = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        print(f"  A2 vs {name:25s}: Wilcoxon p={p_w:.4e} {sig}, Cohen's d={d:.4f} ({d_label})")
    
    # ─── Bootstrap CI for A2 ──────────────────────────────────────────────────
    print("\n### BOOTSTRAP R² CI for A2 ###")
    n_bootstrap = 10000
    bootstrap_r2s = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(a2_result['targets']), size=len(a2_result['targets']), replace=True)
        boot_r2 = r2_score(a2_result['targets'][idx], a2_result['preds'][idx])
        bootstrap_r2s.append(boot_r2)
    
    bootstrap_r2s = np.array(bootstrap_r2s)
    ci_lower = np.percentile(bootstrap_r2s, 2.5)
    ci_upper = np.percentile(bootstrap_r2s, 97.5)
    ci_mean = np.mean(bootstrap_r2s)
    
    print(f"  A2 Point R²: {a2_result['r2']:.4f}")
    print(f"  Bootstrap mean: {ci_mean:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Shapiro-Wilk
    sw_stat, sw_p = stats.shapiro(dl_residuals[:5000])
    print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.2e}")
    
    # Permutation test A2 vs A6
    a2_r2s = ablation_results['A2']['r2_values']
    a6_r2s = ablation_results['A6']['r2_values']
    observed_diff = np.mean(a2_r2s) - np.mean(a6_r2s)
    
    combined = a2_r2s + a6_r2s
    n_perm = 10000
    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diffs.append(np.mean(combined[:5]) - np.mean(combined[5:]))
    perm_p = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    print(f"\n  Permutation test A2 vs A6: diff={observed_diff:.4f}, p={perm_p:.4f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 4: GENERATE MISSING FIGURES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 4: GENERATING FIGURES")
    print("=" * 80)
    
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'manuscript', 'springer', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # ─── Figure A: Ablation Confidence Intervals ─────────────────────────────
    print("  Generating fig_ablation_confidence_intervals.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = list(ablation_results.keys())
    means = [ablation_results[c]['r2_mean'] for c in configs]
    ci_lowers = [ablation_results[c]['r2_mean'] - ablation_results[c]['r2_ci_lower'] for c in configs]
    ci_uppers = [ablation_results[c]['r2_ci_upper'] - ablation_results[c]['r2_mean'] for c in configs]
    stds = [ablation_results[c]['r2_std'] for c in configs]
    
    colors = ['#2ecc71' if c == 'A2' else '#e74c3c' if c == 'A7' else '#3498db' for c in configs]
    
    bars = ax.bar(configs, means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.errorbar(configs, means, yerr=[ci_lowers, ci_uppers], fmt='none', 
                ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci_uppers[i] + 0.01,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Ablation Configuration', fontweight='bold')
    ax.set_ylabel('Calibration R² (5-seed mean)', fontweight='bold')
    ax.set_title('Ablation Study: R² with 95% Confidence Intervals', fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=means[1], color='green', linestyle='--', alpha=0.3, label=f'A2 Best = {means[1]:.3f}')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add feature count labels
    feat_counts = [len(ablation_configs[c]) for c in configs]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([f'{n} feat.' for n in feat_counts], fontsize=8, color='gray')
    ax2.set_xlabel('Number of Features', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_ablation_confidence_intervals.png'))
    plt.close()
    print("    Done.")
    
    # ─── Figure B: Training Curves ───────────────────────────────────────────
    print("  Generating fig_ablation_training_curves.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    color_map = plt.cm.tab10
    for i, (config_name, curves) in enumerate(all_training_curves.items()):
        color = color_map(i / len(all_training_curves))
        axes[0].plot(curves['train'], label=config_name, color=color, linewidth=1.2)
        axes[1].plot(curves['val'], label=config_name, color=color, linewidth=1.2)
    
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Training Loss (Huber)', fontweight='bold')
    axes[0].set_title('Training Loss Convergence', fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Validation Loss (Huber)', fontweight='bold')
    axes[1].set_title('Validation Loss Convergence', fontweight='bold')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_ablation_training_curves.png'))
    plt.close()
    print("    Done.")
    
    # ─── Figure C: Sensitivity Lambda ────────────────────────────────────────
    print("  Generating fig_sensitivity_lambda.png...")
    # Use the known lambda sweep results from results_summary
    lambda_data = {
        0.1: {'calib_r2': 0.894, 'forecast_rmse': 1.375},
        0.5: {'calib_r2': 0.898, 'forecast_rmse': 1.395},
        1.0: {'calib_r2': 0.904, 'forecast_rmse': 1.366},
        2.0: {'calib_r2': 0.881, 'forecast_rmse': 1.892},
        5.0: {'calib_r2': 0.876, 'forecast_rmse': 1.593},
        10.0: {'calib_r2': 0.731, 'forecast_rmse': 1.523},
    }
    
    lambdas = list(lambda_data.keys())
    calib_r2s = [lambda_data[l]['calib_r2'] for l in lambdas]
    forecast_rmses = [lambda_data[l]['forecast_rmse'] for l in lambdas]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = '#2ecc71'
    color2 = '#e74c3c'
    
    ax1.set_xlabel('Loss Weight λ (Forecasting/Calibration)', fontweight='bold')
    ax1.set_ylabel('Calibration R²', color=color1, fontweight='bold')
    line1 = ax1.plot(lambdas, calib_r2s, 'o-', color=color1, linewidth=2, 
                     markersize=8, label='Calibration R²', zorder=5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.7, 0.92)
    
    # Highlight optimal
    best_idx = np.argmax(calib_r2s)
    ax1.scatter([lambdas[best_idx]], [calib_r2s[best_idx]], 
                color=color1, s=150, zorder=10, edgecolors='black', linewidth=2)
    ax1.annotate(f'Optimal\nλ = {lambdas[best_idx]}',
                xy=(lambdas[best_idx], calib_r2s[best_idx]),
                xytext=(lambdas[best_idx] + 1.5, calib_r2s[best_idx] + 0.005),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Forecast RMSE (% moisture)', color=color2, fontweight='bold')
    line2 = ax2.plot(lambdas, forecast_rmses, 's--', color=color2, linewidth=2, 
                     markersize=8, label='Forecast RMSE')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(1.2, 2.0)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', framealpha=0.9)
    
    ax1.set_title('Multi-Task Loss Weight Sensitivity Analysis', fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_sensitivity_lambda.png'))
    plt.close()
    print("    Done.")
    
    # ─── Figure D: MTL Feature Importance ────────────────────────────────────
    print("  Generating fig_mtl_feature_importance.png...")
    
    # From results_summary
    feature_importance = {
        'Sensor_Voltage_V':            {'standalone': 0.413, 'mtl': 0.514, 'ratio': 1.24},
        'Raw_Capacitive_ADC':          {'standalone': 0.358, 'mtl': 0.443, 'ratio': 1.24},
        'Soil_Temperature_C':          {'standalone': 0.018, 'mtl': 0.043, 'ratio': 2.36},
        'Atm_Pressure_inHg':           {'standalone': 0.023, 'mtl': 0.045, 'ratio': 1.95},
        'Atm_Temperature_C':           {'standalone': 0.026, 'mtl': 0.038, 'ratio': 1.48},
        'Sensor_Board_Temperature_C':  {'standalone': 0.024, 'mtl': 0.032, 'ratio': 1.33},
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names_short = ['Voltage', 'ADC', 'Soil Temp', 'Pressure', 'Atm Temp', 'Board Temp']
    standalone_vals = [v['standalone'] for v in feature_importance.values()]
    mtl_vals = [v['mtl'] for v in feature_importance.values()]
    ratios = [v['ratio'] for v in feature_importance.values()]
    
    x = np.arange(len(names_short))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, standalone_vals, width, label='CalibrationNet (Standalone)',
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = axes[0].bar(x + width/2, mtl_vals, width, label='CalibrationNet (MTL)',
                        color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[0].set_xlabel('Feature', fontweight='bold')
    axes[0].set_ylabel('Input Gradient Importance', fontweight='bold')
    axes[0].set_title('Feature Importance: Standalone vs MTL', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names_short, rotation=30, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Ratio plot
    colors = ['#e74c3c' if r > 1.5 else '#f39c12' if r > 1.3 else '#3498db' for r in ratios]
    bars_ratio = axes[1].barh(names_short, ratios, color=colors, alpha=0.8, 
                               edgecolor='black', linewidth=0.5)
    axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='No change')
    axes[1].set_xlabel('MTL / Standalone Importance Ratio', fontweight='bold')
    axes[1].set_title('MTL Feature Upweighting Ratio', fontweight='bold')
    
    for i, (val, bar) in enumerate(zip(ratios, bars_ratio)):
        axes[1].text(val + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}×', va='center', fontweight='bold', fontsize=10)
    
    axes[1].set_xlim(0.9, max(ratios) + 0.4)
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_mtl_feature_importance.png'))
    plt.close()
    print("    Done.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments')
    os.makedirs(results_dir, exist_ok=True)
    
    output = {
        'a2_primary': {
            'r2': float(a2_result['r2']),
            'rmse': float(a2_result['rmse']),
            'mae': float(a2_result['mae']),
            'bootstrap_ci_lower': float(ci_lower),
            'bootstrap_ci_upper': float(ci_upper),
            'bootstrap_mean': float(ci_mean),
            'n_bootstrap': n_bootstrap,
        },
        'ablation_results': {
            k: {
                'features': v['features'],
                'r2_mean': float(v['r2_mean']),
                'r2_std': float(v['r2_std']),
                'r2_ci_lower': float(v['r2_ci_lower']),
                'r2_ci_upper': float(v['r2_ci_upper']),
                'rmse_mean': float(v['rmse_mean']),
            } for k, v in ablation_results.items()
        },
        'baselines_a2_features': baseline_results,
        'paired_tests_a2_vs_baselines': paired_tests,
        'permutation_a2_vs_a6': {
            'observed_diff': float(observed_diff),
            'p_value': float(perm_p),
        },
        'shapiro_wilk': {
            'statistic': float(sw_stat),
            'p_value': float(sw_p),
        },
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f'a2_recomputation_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"A2 (primary): R² = {a2_result['r2']:.4f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"A2 vs RF: Cohen's d = {paired_tests['Random Forest']['cohens_d']:.4f}, "
          f"Wilcoxon p = {paired_tests['Random Forest']['wilcoxon_p']:.4e}")
    print(f"A2 vs A6 permutation: diff = {observed_diff:.4f}, p = {perm_p:.4f}")
    print(f"Results saved to {results_path}")
    print(f"Figures saved to {fig_dir}")
    
    return output


if __name__ == '__main__':
    results = run_a2_recomputation()
