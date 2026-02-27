"""
Sensitivity Analysis of Multi-Task Learning Weighting
=====================================================

Addresses R1-W6: Analyze sensitivity to the task balancing parameter λ.
Loss = L_calibration + λ * L_forecasting

The default implementation uses learnable uncertainty weighting (Kendall et al.).
This script forces fixed weights to demonstrate the impact of λ and justify
the learnable approach (or find an optimal fixed λ).

Usage:
  python experiments/sensitivity_analysis.py
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from data.preprocessing import DataPreprocessor
from data.data_loader import MultiTaskDataset
from torch.utils.data import DataLoader
from models.architectures import MultiTaskNet


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_with_lambda(lambda_val, processed_data, device, epochs=50):
    """Train MultiTaskNet with a fixed lambda weight."""
    set_seed(42)
    
    calib_features = processed_data['feature_groups']['calibration_features']
    target_col = processed_data['feature_groups']['target']
    seq_features_cols = calib_features + [target_col]
    
    unique_farms = sorted(processed_data['train']['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
    
    # Create datasets
    train_dataset = MultiTaskDataset(
        processed_data['train'], calib_features, seq_features_cols, target_col,
        farm_encoding=farm_encoding
    )
    val_dataset = MultiTaskDataset(
        processed_data['val'], calib_features, seq_features_cols, target_col,
        farm_encoding=farm_encoding
    )
    test_dataset = MultiTaskDataset(
        processed_data['test'], calib_features, seq_features_cols, target_col,
        farm_encoding=farm_encoding
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = MultiTaskNet(
        calib_input_dim=len(calib_features),
        seq_input_dim=len(seq_features_cols),
        hidden_dim=128,
        lstm_hidden=128,
        lstm_layers=2,
        num_horizons=4,
        num_farms=len(farm_encoding),
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion_calib = nn.HuberLoss(delta=1.0)
    criterion_forecast = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            cf = batch['calib_features'].to(device)
            sf = batch['seq_features'].to(device)
            fi = batch['farm_id'].to(device)
            ct = batch['calib_target'].to(device)
            ft = batch['forecast_targets'].to(device)
            
            calib_out, forecast_out = model(cf, sf, fi)
            
            loss_calib = criterion_calib(calib_out, ct)
            loss_forecast = criterion_forecast(forecast_out, ft)
            
            # Weighted loss
            loss = loss_calib + lambda_val * loss_forecast
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cf = batch['calib_features'].to(device)
                sf = batch['seq_features'].to(device)
                fi = batch['farm_id'].to(device)
                ct = batch['calib_target'].to(device)
                ft = batch['forecast_targets'].to(device)
                
                co, fo = model(cf, sf, fi)
                l_c = criterion_calib(co, ct)
                l_f = criterion_forecast(fo, ft)
                val_loss += (l_c + lambda_val * l_f).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Evaluate
    model.load_state_dict(best_state)
    model.eval()
    
    calib_preds, calib_targets = [], []
    forecast_preds, forecast_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            cf = batch['calib_features'].to(device)
            sf = batch['seq_features'].to(device)
            fi = batch['farm_id'].to(device)
            ct = batch['calib_target'].to(device)
            ft = batch['forecast_targets'].to(device)
            
            co, fo = model(cf, sf, fi)
            
            calib_preds.append(co.cpu().numpy())
            calib_targets.append(ct.cpu().numpy())
            forecast_preds.append(fo.cpu().numpy())
            forecast_targets.append(ft.cpu().numpy())
            
    calib_preds = np.concatenate(calib_preds).flatten()
    calib_targets = np.concatenate(calib_targets).flatten()
    forecast_preds = np.concatenate(forecast_preds)
    forecast_targets = np.concatenate(forecast_targets)
    
    return {
        'lambda': lambda_val,
        'calib_r2': float(r2_score(calib_targets, calib_preds)),
        'forecast_rmse': float(np.sqrt(mean_squared_error(forecast_targets, forecast_preds))),
        'forecast_mae': float(np.mean(np.abs(forecast_targets - forecast_preds))),
        'epochs': epoch
    }


def run_sensitivity_analysis(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run lambda sensitivity analysis."""
    print("=" * 80)
    print("SENSITIVITY ANALYSIS (MULTI-TASK LAMBDA)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_path)
    
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    print("\nStarting sweep...")
    print(f"{'Lambda':<10} {'Calib R²':<12} {'Forecast RMSE':<15} {'Epochs':<8}")
    print("-" * 50)
    
    for l_val in lambda_values:
        res = train_with_lambda(l_val, processed_data, device)
        results.append(res)
        print(f"{l_val:<10.1f} {res['calib_r2']:<12.4f} {res['forecast_rmse']:<15.4f} {res['epochs']:<8}")
        
    # Plotting
    os.makedirs('results/figures', exist_ok=True)
    
    l_vals = [r['lambda'] for r in results]
    r2_vals = [r['calib_r2'] for r in results]
    rmse_vals = [r['forecast_rmse'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Lambda (Forecast Loss Weight)')
    ax1.set_ylabel('Calibration R²', color=color)
    ax1.plot(l_vals, r2_vals, marker='o', color=color, label='Calibration R²')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Forecast RMSE', color=color)
    ax2.plot(l_vals, rmse_vals, marker='s', color=color, linestyle='--', label='Forecast RMSE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    
    # Save JSON
    os.makedirs('../results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f'../results/experiments/sensitivity_analysis_{timestamp}.json'
    
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'lambdas': lambda_values}, f, indent=2)
        
    print(f"\n[OK] Results saved to {out_path}")

    
    return results


if __name__ == '__main__':
    run_sensitivity_analysis()
