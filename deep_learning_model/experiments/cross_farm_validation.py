"""
Cross-Farm Validation: Generalization Test
==========================================

Tests model generalization by training on one farm and testing
on the other. This demonstrates the model's ability to handle
inter-sensor variability.

Experiments:
1. Train Farm_1 → Test Farm_2
2. Train Farm_2 → Test Farm_1
3. Combined training (baseline reference)

Usage:
    python experiments/cross_farm_validation.py
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


def train_and_evaluate(train_df, test_df, feature_cols, target_col, farm_encoding, 
                       device, epochs=50, batch_size=64, patience=10):
    """Train on one set, evaluate on another."""
    
    train_dataset = CalibrationDataset(train_df, feature_cols, target_col, farm_encoding)
    test_dataset = CalibrationDataset(test_df, feature_cols, target_col, farm_encoding)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = len(feature_cols)
    model = CalibrationNet(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_farms=len(farm_encoding),
        dropout=0.3
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n = 0
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
            epoch_loss += loss.item()
            n += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Evaluate
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
    
    return {
        'r2': float(r2_score(targets, preds)),
        'rmse': float(np.sqrt(mean_squared_error(targets, preds))),
        'mae': float(mean_absolute_error(targets, preds)),
        'epochs_trained': epoch,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }


def run_cross_farm_validation(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run cross-farm generalization experiments."""
    
    print("=" * 80)
    print("CROSS-FARM VALIDATION: Generalization Test")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    preprocessor = DataPreprocessor(scaling_method='standard')
    processed = preprocessor.preprocess_pipeline(filepath=data_path)
    
    feature_groups = processed['feature_groups']
    feature_cols = feature_groups['calibration_features'] + feature_groups['temporal_features']
    target_col = feature_groups['target']
    
    # Combine all data
    all_data = processed['train_raw'].copy()
    all_data = all_data._append(processed['val_raw'])
    all_data = all_data._append(processed['test_raw'])
    
    # Split by farm
    farms = sorted(all_data['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(farms)}
    
    print(f"\nFarms: {farms}")
    for farm in farms:
        print(f"  {farm}: {len(all_data[all_data['Farm_ID'] == farm])} records")
    
    results = {}
    
    # Experiment 1: Train Farm_1 → Test Farm_2
    print(f"\n{'═' * 60}")
    print(f"Experiment 1: Train {farms[0]} → Test {farms[1]}")
    print(f"{'═' * 60}")
    
    farm1_data = all_data[all_data['Farm_ID'] == farms[0]]
    farm2_data = all_data[all_data['Farm_ID'] == farms[1]]
    
    result = train_and_evaluate(farm1_data, farm2_data, feature_cols, target_col, farm_encoding, device)
    results['train_farm1_test_farm2'] = result
    print(f"  R² = {result['r2']:.4f} | RMSE = {result['rmse']:.4f} | MAE = {result['mae']:.4f}")
    
    # Experiment 2: Train Farm_2 → Test Farm_1
    print(f"\n{'═' * 60}")
    print(f"Experiment 2: Train {farms[1]} → Test {farms[0]}")
    print(f"{'═' * 60}")
    
    result = train_and_evaluate(farm2_data, farm1_data, feature_cols, target_col, farm_encoding, device)
    results['train_farm2_test_farm1'] = result
    print(f"  R² = {result['r2']:.4f} | RMSE = {result['rmse']:.4f} | MAE = {result['mae']:.4f}")
    
    # Experiment 3: Combined (80/20 split for reference)
    print(f"\n{'═' * 60}")
    print("Experiment 3: Combined Training (Reference)")
    print(f"{'═' * 60}")
    
    split_idx = int(0.8 * len(all_data))
    combined_train = all_data.iloc[:split_idx]
    combined_test = all_data.iloc[split_idx:]
    
    result = train_and_evaluate(combined_train, combined_test, feature_cols, target_col, farm_encoding, device)
    results['combined_reference'] = result
    print(f"  R² = {result['r2']:.4f} | RMSE = {result['rmse']:.4f} | MAE = {result['mae']:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CROSS-FARM VALIDATION RESULTS")
    print("=" * 80)
    print(f"\n{'Experiment':<35} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("─" * 60)
    for exp_name, result in results.items():
        print(f"{exp_name:<35} {result['r2']:>8.4f} {result['rmse']:>8.4f} {result['mae']:>8.4f}")
    
    # Save
    os.makedirs('results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/experiments/cross_farm_validation_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_cross_farm_validation()
