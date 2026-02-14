"""
Ablation Study: 7-Configuration Feature Analysis
=================================================

Tests the contribution of each feature group by incrementally
adding features to the calibration model.

Configurations:
  A1: ADC only (1 feature)
  A2: A1 + Sensor Voltage (2 features)
  A3: A2 + Sensor Board Temp (3 features) 
  A4: A3 + Atmospheric Temp (4 features)
  A5: A4 + Soil Temp (5 features)
  A6: A5 + Pressure (6 features)  ← pressure contribution test
  A7: A6 + Temporal features (15 features) ← full model

Usage:
    python experiments/ablation_study.py
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
from data.data_loader import CalibrationDataset
from torch.utils.data import DataLoader
from models.architectures import CalibrationNet


# Ablation configurations — each adds one feature group
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


def train_ablation_config(config_name, feature_cols, processed_data, device, 
                           epochs=50, batch_size=64, lr=0.001, patience=10):
    """Train a single ablation configuration."""
    
    target_col = processed_data['feature_groups']['target']
    
    # Get farm encoding from training data
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
        hidden_dims=[128, 64, 32],  # Smaller model for ablation
        num_farms=len(farm_encoding),
        dropout=0.3
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    # Training
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
        
        # Validate
        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                farm_ids = batch['farm_id'].to(device)
                targets = batch['target'].to(device)
                outputs = model(features, farm_ids)
                val_loss += criterion(outputs, targets).item()
                n += 1
        val_loss /= n
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model and evaluate
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
    
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    
    return {
        'config': config_name,
        'num_features': len(feature_cols),
        'features': feature_cols,
        'test_r2': float(r2),
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'epochs_trained': epoch,
        'params': sum(p.numel() for p in model.parameters())
    }


def run_ablation_study(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run the full 7-config ablation study."""
    
    print("=" * 80)
    print("ABLATION STUDY: 7-Configuration Feature Analysis")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    preprocessor = DataPreprocessor(scaling_method='standard')
    processed = preprocessor.preprocess_pipeline(filepath=data_path)
    
    results = {}
    
    for config_name, features in ABLATION_CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"Config: {config_name} ({len(features)} features)")
        print(f"Features: {features}")
        print(f"{'─' * 60}")
        
        start_time = time.time()
        result = train_ablation_config(config_name, features, processed, device)
        train_time = time.time() - start_time
        result['train_time_seconds'] = float(train_time)
        
        results[config_name] = result
        
        print(f"  R² = {result['test_r2']:.4f} | RMSE = {result['test_rmse']:.4f} | "
              f"MAE = {result['test_mae']:.4f} | Time: {train_time:.1f}s")
    
    # Summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"\n{'Config':<25} {'#Feat':>6} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'ΔR²':>8}")
    print("─" * 65)
    
    prev_r2 = None
    for config_name, result in results.items():
        delta = f"{result['test_r2'] - prev_r2:+.4f}" if prev_r2 is not None else "  base"
        print(f"{config_name:<25} {result['num_features']:>6} {result['test_r2']:>8.4f} "
              f"{result['test_rmse']:>8.4f} {result['test_mae']:>8.4f} {delta:>8}")
        prev_r2 = result['test_r2']
    
    # Save results
    os.makedirs('results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/experiments/ablation_study_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_ablation_study()
