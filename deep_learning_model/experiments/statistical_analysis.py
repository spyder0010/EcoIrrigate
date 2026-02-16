"""
Statistical Analysis Script
============================

Addresses:
  R3-W2: Cohen's d against Random Forest (best competitive baseline)
  R3-W4: Paired residual test (DL vs RF/other baselines)
  R3-W1: Bootstrap CI verification — ensure CI matches reported model

Usage:
  python experiments/statistical_analysis.py
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


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_statistical_analysis(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run all statistical analyses."""
    print("=" * 80)
    print("STATISTICAL ANALYSIS SUITE")
    print("=" * 80)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_path)
    
    feature_groups = processed_data['feature_groups']
    calib_features = feature_groups['calibration_features']  # A6 features (6)
    all_features = calib_features + feature_groups['temporal_features']  # 15 features
    target = feature_groups['target']
    
    # Prepare data for sklearn
    X_train = processed_data['train'][calib_features].values
    y_train = processed_data['train'][target].values
    X_val = processed_data['val'][calib_features].values
    y_val = processed_data['val'][target].values
    X_test = processed_data['test'][calib_features].values
    y_test = processed_data['test'][target].values
    
    # ─── Train All 9 ML Baselines (R5-W1) ─────────────────────────────────────
    print("\n### TRAINING ALL 9 ML BASELINES ###")
    
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
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
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
    
    # ─── Train CalibrationNet with A6 features ─────────────────────────────────
    print("\n### TRAINING CalibrationNet (A6 features) ###")
    
    unique_farms = sorted(processed_data['train']['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
    
    train_dataset = CalibrationDataset(processed_data['train'], calib_features, target, farm_encoding)
    val_dataset = CalibrationDataset(processed_data['val'], calib_features, target, farm_encoding)
    test_dataset = CalibrationDataset(processed_data['test'], calib_features, target, farm_encoding)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = CalibrationNet(
        input_dim=6, hidden_dims=[256, 128, 64],
        num_farms=len(farm_encoding), dropout=0.3
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.0001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, 151):
        model.train()
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
        scheduler.step()
        
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
    dl_residuals = dl_targets - dl_preds
    
    dl_r2 = r2_score(dl_targets, dl_preds)
    dl_rmse = np.sqrt(mean_squared_error(dl_targets, dl_preds))
    dl_mae = mean_absolute_error(dl_targets, dl_preds)
    
    print(f"  CalibrationNet (A6): R2={dl_r2:.4f}, RMSE={dl_rmse:.4f}, MAE={dl_mae:.4f}")
    
    # ─── R3-W2: Cohen's d Against Best Baseline (Random Forest) ───────────────
    print("\n### COHEN'S d ANALYSIS ###")
    
    rf_residuals = baseline_residuals['Random Forest']
    
    # Cohen's d for paired comparison: DL vs RF
    diff = np.abs(rf_residuals) - np.abs(dl_residuals)
    cohens_d_rf = np.mean(diff) / np.std(diff, ddof=1)
    
    # Also compute against mean baseline for reference
    mean_pred = np.full_like(dl_targets, np.mean(y_train))
    mean_residuals = dl_targets - mean_pred
    diff_mean = np.abs(mean_residuals) - np.abs(dl_residuals)
    cohens_d_mean = np.mean(diff_mean) / np.std(diff_mean, ddof=1)
    
    print(f"  Cohen's d (DL vs Random Forest): {cohens_d_rf:.4f}")
    print(f"  Cohen's d (DL vs Mean Baseline): {cohens_d_mean:.4f}")
    print(f"  Interpretation: ", end="")
    if abs(cohens_d_rf) < 0.2:
        print("negligible")
    elif abs(cohens_d_rf) < 0.5:
        print("small")
    elif abs(cohens_d_rf) < 0.8:
        print("medium")
    else:
        print("large")
    
    # ─── R3-W4: Paired Residual Tests ──────────────────────────────────────────
    print("\n### PAIRED RESIDUAL TESTS ###")
    
    paired_tests = {}
    for name in ['Random Forest', 'Gradient Boosting', 'MLP', 'Ridge']:
        bl_res = np.abs(baseline_residuals[name])
        dl_res = np.abs(dl_residuals)
        
        # Wilcoxon signed-rank test on |residuals|
        stat_w, p_w = stats.wilcoxon(bl_res, dl_res)
        
        # Paired t-test on |residuals|
        stat_t, p_t = stats.ttest_rel(bl_res, dl_res)
        
        # Cohen's d
        diff = bl_res - dl_res
        d = np.mean(diff) / np.std(diff, ddof=1)
        
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
        print(f"  DL vs {name:25s}: Wilcoxon p={p_w:.4e} {sig}, Cohen's d={d:.4f}")
    
    # ─── R3-W1: Bootstrap CI for CalibrationNet (A6) ──────────────────────────
    print("\n### BOOTSTRAP R² CONFIDENCE INTERVAL ###")
    
    n_bootstrap = 1000
    bootstrap_r2s = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(dl_targets), size=len(dl_targets), replace=True)
        boot_r2 = r2_score(dl_targets[idx], dl_preds[idx])
        bootstrap_r2s.append(boot_r2)
    
    bootstrap_r2s = np.array(bootstrap_r2s)
    ci_lower = np.percentile(bootstrap_r2s, 2.5)
    ci_upper = np.percentile(bootstrap_r2s, 97.5)
    ci_mean = np.mean(bootstrap_r2s)
    
    print(f"  CalibrationNet (A6 features)")
    print(f"  Point estimate R²: {dl_r2:.4f}")
    print(f"  Bootstrap mean R²: {ci_mean:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  N bootstrap samples: {n_bootstrap}")
    
    # Shapiro-Wilk on residuals
    sw_stat, sw_p = stats.shapiro(dl_residuals[:5000])  # Limit to 5000 for Shapiro-Wilk
    print(f"\n  Shapiro-Wilk (residuals): W={sw_stat:.4f}, p={sw_p:.2e}")
    print(f"  Residuals are {'non-' if sw_p < 0.05 else ''}normally distributed")
    
    # Mann-Whitney U for inter-farm comparison
    farm_ids_test = processed_data['test']['Farm_ID'].values[:len(dl_residuals)]
    unique_test_farms = sorted(set(farm_ids_test))
    if len(unique_test_farms) >= 2:
        farm1_mask = farm_ids_test == unique_test_farms[0]
        farm2_mask = farm_ids_test == unique_test_farms[1]
        
        farm1_residuals = np.abs(dl_residuals[farm1_mask])
        farm2_residuals = np.abs(dl_residuals[farm2_mask])
        
        mw_stat, mw_p = stats.mannwhitneyu(farm1_residuals, farm2_residuals, alternative='two-sided')
        print(f"\n  Mann-Whitney U (Farm 1 vs Farm 2 |residuals|):")
        print(f"    U={mw_stat:.1f}, p={mw_p:.4e}")
        print(f"    Farm 1 MAE: {np.mean(farm1_residuals):.4f}")
        print(f"    Farm 2 MAE: {np.mean(farm2_residuals):.4f}")
    
    # ─── Save All Results ──────────────────────────────────────────────────────
    os.makedirs('../results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        'calibration_net_a6': {
            'r2': float(dl_r2),
            'rmse': float(dl_rmse),
            'mae': float(dl_mae),
        },
        'all_baselines': baseline_results,
        'cohens_d': {
            'dl_vs_rf': float(cohens_d_rf),
            'dl_vs_mean': float(cohens_d_mean),
        },
        'paired_tests': paired_tests,
        'bootstrap_ci': {
            'model': 'CalibrationNet_A6',
            'point_r2': float(dl_r2),
            'bootstrap_mean_r2': float(ci_mean),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstrap': n_bootstrap,
        },
        'shapiro_wilk': {
            'statistic': float(sw_stat),
            'p_value': float(sw_p),
        },
        'timestamp': timestamp,
    }
    
    results_path = f'../results/experiments/statistical_analysis_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return output


if __name__ == '__main__':
    results = run_statistical_analysis()
