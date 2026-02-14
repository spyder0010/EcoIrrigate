"""
Main Training Script for Deep Learning Models
==============================================

Unified interface for training all model architectures:
- CalibrationNet
- ForecastingNet
- MultiModalFusionNet
- MultiTaskNet

Usage:
    python main.py --task calibration --model CalibrationNet --epochs 100
    python main.py --task forecasting --model ForecastingNet --epochs 100
    python main.py --task multi-task --model MultiTaskNet --epochs 100

Author: Research Team
Date: February 2026
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# Import custom modules
from data.preprocessing import DataPreprocessor
from data.data_loader import create_data_loaders
from models.architectures import (
    CalibrationNet,
    ForecastingNet,
    MultiModalFusionNet,
    MultiTaskNet
)
from models.losses import (
    HuberLoss,
    WeightedMSELoss,
    MultiTaskLoss
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Deterministic operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (CUDA/MPS/CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple M1/M2 GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (training will be slower)")
    
    return device


def create_model(args, num_farms=2):
    """
    Create model based on task and architecture.
    
    Args:
        args: Command-line arguments
        num_farms: Number of farms in dataset
        
    Returns:
        Initialized model
    """
    if args.model == 'CalibrationNet':
        model = CalibrationNet(
            input_dim=args.input_dim,
            hidden_dims=[256, 128, 64],
            num_farms=num_farms,
            dropout=args.dropout
        )
    elif args.model == 'ForecastingNet':
        model = ForecastingNet(
            input_dim=args.input_dim,
            hidden_dim=128,
            num_layers=2,
            num_horizons=4,
            num_farms=num_farms,
            dropout=args.dropout
        )
    elif args.model == 'MultiModalFusionNet':
        model = MultiModalFusionNet(
            sensor_dim=3,       # ADC + voltage + board_temp
            env_dim=3,          # Soil temp + atmospheric temp + pressure
            temporal_dim=9,     # Hour, Day, Month, DayOfWeek, DayOfYear, sin/cos features
            hidden_dim=128,
            num_heads=4,
            num_farms=num_farms,
            dropout=args.dropout
        )
    elif args.model == 'MultiTaskNet':
        # Multi-task data loader: calib=calibration_features only, seq=calib+target
        model = MultiTaskNet(
            calib_input_dim=args.calib_dim,
            seq_input_dim=args.calib_dim + 1,
            hidden_dim=128,
            lstm_hidden=128,
            lstm_layers=2,
            num_horizons=4,
            num_farms=num_farms,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def create_loss_function(args):
    """Create appropriate loss function for task."""
    if args.task == 'calibration':
        return HuberLoss(delta=1.0)
    elif args.task == 'forecasting':
        return WeightedMSELoss(horizon_weights=[1.0, 0.8, 0.6, 0.4])
    elif args.task == 'multi-task':
        return MultiTaskLoss(num_tasks=2)
    else:
        raise ValueError(f"Unknown task: {args.task}")


def train_epoch(model, train_loader, criterion, optimizer, device, task='calibration'):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        task: Task type
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        # Move data to device
        if task == 'calibration':
            features = batch['features'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(features, farm_ids)
            loss = criterion(outputs, targets)
            
        elif task == 'forecasting':
            sequences = batch['sequence'].to(device)
            farm_ids = batch['farm_id'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            outputs, _ = model(sequences, farm_ids)
            loss = criterion(outputs, targets)
            
        elif task == 'multi-task':
            calib_feat = batch['calib_features'].to(device)
            seq_feat = batch['seq_features'].to(device)
            farm_ids = batch['farm_id'].to(device)
            calib_target = batch['calib_target'].to(device)
            forecast_targets = batch['forecast_targets'].to(device)
            
            # Forward pass
            calib_out, forecast_out = model(calib_feat, seq_feat, farm_ids)
            
            # Compute losses
            calib_loss = nn.MSELoss()(calib_out, calib_target)
            forecast_loss = nn.MSELoss()(forecast_out, forecast_targets)
            
            loss = criterion([calib_loss, forecast_loss])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device, task='calibration'):
    """
    Validate model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        task: Task type
        
    Returns:
        Validation loss and predictions
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            if task == 'calibration':
                features = batch['features'].to(device)
                farm_ids = batch['farm_id'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(features, farm_ids)
                loss = criterion(outputs, targets)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
            elif task == 'forecasting':
                sequences = batch['sequence'].to(device)
                farm_ids = batch['farm_id'].to(device)
                targets = batch['targets'].to(device)
                
                outputs, _ = model(sequences, farm_ids)
                loss = criterion(outputs, targets)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
            elif task == 'multi-task':
                calib_feat = batch['calib_features'].to(device)
                seq_feat = batch['seq_features'].to(device)
                farm_ids = batch['farm_id'].to(device)
                calib_target = batch['calib_target'].to(device)
                forecast_targets = batch['forecast_targets'].to(device)
                
                calib_out, forecast_out = model(calib_feat, seq_feat, farm_ids)
                
                calib_loss = nn.MSELoss()(calib_out, calib_target)
                forecast_loss = nn.MSELoss()(forecast_out, forecast_targets)
                
                loss = criterion([calib_loss, forecast_loss])
                
                all_predictions.append(calib_out.cpu().numpy())
                all_targets.append(calib_target.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, predictions, targets


def main(args):
    """Main training pipeline."""
    
    print("="*80)
    print("DEEP LEARNING MODEL TRAINING")
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create output directories
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    # Load and preprocess data
    print("\n### LOADING DATA ###")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(
        filepath=args.data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader, farm_encoding = create_data_loaders(
        processed_data,
        batch_size=args.batch_size,
        task=args.task,
        num_workers=args.num_workers
    )
    
    # Get input dimension
    calib_dim = len(processed_data['feature_groups']['calibration_features'])
    temporal_dim = len(processed_data['feature_groups']['temporal_features'])
    args.calib_dim = calib_dim  # Used by MultiTaskNet
    if args.task == 'calibration':
        args.input_dim = calib_dim + temporal_dim
    elif args.task == 'forecasting':
        args.input_dim = calib_dim + temporal_dim + 1
    else:
        args.input_dim = calib_dim + temporal_dim
    
    # Create model
    print("\n### CREATING MODEL ###")
    model = create_model(args, num_farms=len(farm_encoding))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss and optimizer
    criterion = create_loss_function(args)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 10
    )
    
    # Training loop
    print("\n### TRAINING ###")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, args.task
        )
        
        # Validate
        val_loss, _, _ = validate(
            model, val_loader, criterion, device, args.task
        )
        
        # Step scheduler
        scheduler.step()
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if epoch % args.print_freq == 0:
            print(f"Epoch [{epoch}/{args.epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"results/models/{args.model}_{args.task}_{timestamp}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, model_path)
            
            print(f"✓ Saved best model to {model_path}")
        else:
            patience_counter += 1
            
        # Stop if no improvement
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation on test set
    print("\n### TESTING ###")
    test_loss, test_preds, test_targets = validate(
        model, test_loader, criterion, device, args.task
    )
    
    # Compute metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    if args.task in ['calibration', 'multi-task']:
        r2 = r2_score(test_targets, test_preds)
        rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        mae = mean_absolute_error(test_targets, test_preds)
        
        print(f"\n### TEST RESULTS ###")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Save results
        results = {
            'task': args.task,
            'model': args.model,
            'test_loss': float(test_loss),
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'best_epoch': epoch - patience_counter,
            'total_epochs': epoch,
            'best_val_loss': float(best_val_loss)
        }
    else:
        # Forecasting: compute per-horizon metrics
        horizons = ['1h', '6h', '12h', '24h']
        results = {
            'task': args.task,
            'model': args.model,
            'test_loss': float(test_loss)
        }
        
        print(f"\n### TEST RESULTS (Multi-Horizon) ###")
        for i, horizon in enumerate(horizons):
            horizon_preds = test_preds[:, i]
            horizon_targets = test_targets[:, i]
            
            rmse = np.sqrt(mean_squared_error(horizon_targets, horizon_preds))
            mae = mean_absolute_error(horizon_targets, horizon_preds)
            
            print(f"{horizon}: RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            results[f'{horizon}_rmse'] = float(rmse)
            results[f'{horizon}_mae'] = float(mae)
    
    # Save results to JSON
    results_path = f"results/logs/{args.model}_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deep Learning Models for Soil Moisture Prediction')
    
    # Data
    parser.add_argument('--data-path', type=str, 
                       default='../New_Dataset/kolkata_unified_dataset.csv',
                       help='Path to dataset')
    
    # Task & Model
    parser.add_argument('--task', type=str, required=True,
                       choices=['calibration', 'forecasting', 'multi-task'],
                       help='Training task')
    parser.add_argument('--model', type=str, required=True,
                       choices=['CalibrationNet', 'ForecastingNet', 'MultiModalFusionNet', 'MultiTaskNet'],
                       help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers')
    parser.add_argument('--print-freq', type=int, default=5,
                       help='Print frequency (epochs)')
    
    args = parser.parse_args()
    
    main(args)
