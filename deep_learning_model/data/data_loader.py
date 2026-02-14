"""
PyTorch DataLoader for Soil Moisture Prediction Models
========================================================

Custom Dataset classes for:
1. Calibration task (point-wise regression)
2. Forecasting task (sequence-to-sequence)
3. Multi-task learning (both tasks simultaneously)

Author: Research Team
Date: February 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CalibrationDataset(Dataset):
    """
    Dataset for sensor calibration task (ADC → Moisture %).
    
    Input: Raw ADC + environmental features (point-wise)
    Output: Volumetric moisture %
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'Volumetric_Moisture_Pct',
                 farm_encoding: Dict = None):
        """
        Initialize calibration dataset.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of input feature column names
            target_col: Target variable column name
            farm_encoding: Dictionary mapping farm IDs to integers
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Encode farm IDs
        if farm_encoding is None:
            unique_farms = sorted(df['Farm_ID'].unique())
            self.farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
        else:
            self.farm_encoding = farm_encoding
        
        # Prepare features
        self.X = torch.FloatTensor(df[feature_cols].values)
        self.y = torch.FloatTensor(df[target_col].values).unsqueeze(1)
        
        # Farm IDs as categorical
        farm_ids = df['Farm_ID'].map(self.farm_encoding).values
        self.farm_ids = torch.LongTensor(farm_ids)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'farm_id': self.farm_ids[idx],
            'target': self.y[idx]
        }


class ForecastingDataset(Dataset):
    """
    Dataset for moisture forecasting task (sequence-to-sequence).
    
    Input: Historical sequence (past 24h)
    Output: Future moisture at multiple horizons [+1h, +6h, +12h, +24h]
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'Volumetric_Moisture_Pct',
                 sequence_length: int = 96,  # 24 hours at 15-min intervals
                 forecast_horizons: List[int] = [4, 24, 48, 96],  # 1h, 6h, 12h, 24h
                 farm_encoding: Dict = None):
        """
        Initialize forecasting dataset.
        
        Args:
            df: DataFrame sorted by timestamp
            feature_cols: Input feature columns
            target_col: Target variable
            sequence_length: Length of input sequence
            forecast_horizons: Prediction horizons (in 15-min steps)
            farm_encoding: Farm ID encoding
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        
        # Encode farm IDs
        if farm_encoding is None:
            unique_farms = sorted(df['Farm_ID'].unique())
            self.farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
        else:
            self.farm_encoding = farm_encoding
        
        # Create sequences
        self.samples = self._create_sequences()
        
    def _create_sequences(self):
        """Create sequences from DataFrame."""
        samples = []
        max_horizon = max(self.forecast_horizons)
        
        # Process each farm separately to avoid cross-farm sequences
        for farm_id in self.df['Farm_ID'].unique():
            farm_df = self.df[self.df['Farm_ID'] == farm_id].reset_index(drop=True)
            
            X = farm_df[self.feature_cols].values
            y = farm_df[self.target_col].values
            
            # Create sequences
            for i in range(self.sequence_length, len(X) - max_horizon):
                X_seq = X[i-self.sequence_length:i]
                
                # Multi-horizon targets
                y_multi = []
                for horizon in self.forecast_horizons:
                    y_multi.append(y[i + horizon])
                
                samples.append({
                    'sequence': torch.FloatTensor(X_seq),
                    'targets': torch.FloatTensor(y_multi),
                    'farm_id': torch.tensor(self.farm_encoding[farm_id], dtype=torch.long)
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class MultiTaskDataset(Dataset):
    """
    Dataset for multi-task learning (calibration + forecasting).
    
    Combines both tasks:
    - Task 1: Point-wise calibration
    - Task 2: Sequence-based forecasting
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 calibration_features: List[str],
                 sequence_features: List[str],
                 target_col: str = 'Volumetric_Moisture_Pct',
                 sequence_length: int = 96,
                 forecast_horizons: List[int] = [4, 24, 48, 96],
                 farm_encoding: Dict = None):
        """
        Initialize multi-task dataset.
        
        Args:
            df: Input DataFrame
            calibration_features: Features for calibration task
            sequence_features: Features for sequence modeling
            target_col: Target variable
            sequence_length: Input sequence length
            forecast_horizons: Prediction horizons
            farm_encoding: Farm ID encoding
        """
        self.df = df.reset_index(drop=True)
        self.calibration_features = calibration_features
        self.sequence_features = sequence_features
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        
        # Farm encoding
        if farm_encoding is None:
            unique_farms = sorted(df['Farm_ID'].unique())
            self.farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
        else:
            self.farm_encoding = farm_encoding
        
        # Create samples
        self.samples = self._create_multi_task_samples()
    
    def _create_multi_task_samples(self):
        """Create samples for multi-task learning."""
        samples = []
        max_horizon = max(self.forecast_horizons)
        
        for farm_id in self.df['Farm_ID'].unique():
            farm_df = self.df[self.df['Farm_ID'] == farm_id].reset_index(drop=True)
            
            # Calibration features (point-wise)
            X_calib = farm_df[self.calibration_features].values
            
            # Sequence features
            X_seq = farm_df[self.sequence_features].values
            
            # Target
            y = farm_df[self.target_col].values
            
            # Create samples
            for i in range(self.sequence_length, len(X_calib) - max_horizon):
                # Calibration input (current point)
                calib_input = X_calib[i]
                
                # Sequence input (past sequence)
                seq_input = X_seq[i-self.sequence_length:i]
                
                # Targets
                calib_target = y[i]
                forecast_targets = [y[i + h] for h in self.forecast_horizons]
                
                samples.append({
                    'calib_features': torch.FloatTensor(calib_input),
                    'seq_features': torch.FloatTensor(seq_input),
                    'calib_target': torch.FloatTensor([calib_target]),
                    'forecast_targets': torch.FloatTensor(forecast_targets),
                    'farm_id': torch.tensor(self.farm_encoding[farm_id], dtype=torch.long)
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_data_loaders(processed_data: Dict,
                       batch_size: int = 64,
                       task: str = 'calibration',
                       num_workers: int = 0) -> Tuple:
    """
    Create PyTorch DataLoaders for training.
    
    Args:
        processed_data: Dictionary from preprocessing pipeline
        batch_size: Batch size for training
        task: 'calibration', 'forecasting', or 'multi-task'
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_df = processed_data['train']
    val_df = processed_data['val']
    test_df = processed_data['test']
    
    # Get feature columns
    feature_groups = processed_data['feature_groups']
    
    # Create farm encoding (consistent across splits)
    unique_farms = sorted(train_df['Farm_ID'].unique())
    farm_encoding = {farm: idx for idx, farm in enumerate(unique_farms)}
    
    if task == 'calibration':
        # Point-wise calibration task
        features = feature_groups['calibration_features'] + feature_groups['temporal_features']
        target = feature_groups['target']
        
        train_dataset = CalibrationDataset(
            train_df, features, target, farm_encoding
        )
        val_dataset = CalibrationDataset(
            val_df, features, target, farm_encoding
        )
        test_dataset = CalibrationDataset(
            test_df, features, target, farm_encoding
        )
        
    elif task == 'forecasting':
        # Sequence forecasting task
        features = feature_groups['calibration_features'] + feature_groups['temporal_features']
        target = feature_groups['target']
        
        # First include target in features for historical sequence
        if target not in features:
            features = [target] + features
        
        train_dataset = ForecastingDataset(
            train_df, features, target, 
            sequence_length=96, 
            forecast_horizons=[4, 24, 48, 96],
            farm_encoding=farm_encoding
        )
        val_dataset = ForecastingDataset(
            val_df, features, target,
            sequence_length=96,
            forecast_horizons=[4, 24, 48, 96],
            farm_encoding=farm_encoding
        )
        test_dataset = ForecastingDataset(
            test_df, features, target,
            sequence_length=96,
            forecast_horizons=[4, 24, 48, 96],
            farm_encoding=farm_encoding
        )
        
    elif task == 'multi-task':
        # Multi-task learning
        calib_features = feature_groups['calibration_features']
        seq_features = feature_groups['calibration_features'] + [feature_groups['target']]
        target = feature_groups['target']
        
        train_dataset = MultiTaskDataset(
            train_df, calib_features, seq_features, target,
            farm_encoding=farm_encoding
        )
        val_dataset = MultiTaskDataset(
            val_df, calib_features, seq_features, target,
            farm_encoding=farm_encoding
        )
        test_dataset = MultiTaskDataset(
            test_df, calib_features, seq_features, target,
            farm_encoding=farm_encoding
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n### DATA LOADERS CREATED ({task.upper()}) ###")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, farm_encoding


def main():
    """Test data loaders."""
    import sys
    sys.path.append('..')
    from data.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    data_path = '../../New_Dataset/kolkata_unified_dataset.csv'
    processed_data = preprocessor.preprocess_pipeline(data_path)
    
    # Create calibration loaders
    print("\n### TESTING CALIBRATION TASK ###")
    train_loader, val_loader, test_loader, farm_encoding = create_data_loaders(
        processed_data,
        batch_size=64,
        task='calibration'
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Features shape: {batch['features'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    print(f"Farm IDs shape: {batch['farm_id'].shape}")
    
    # Create forecasting loaders
    print("\n\n### TESTING FORECASTING TASK ###")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        processed_data,
        batch_size=32,
        task='forecasting'
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Sequence shape: {batch['sequence'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")


if __name__ == "__main__":
    main()
