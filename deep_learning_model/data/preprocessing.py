"""
Data Preprocessing Pipeline for Precision Agriculture Deep Learning Model
===========================================================================

This module handles:
1. Data loading and validation
2. Missing value imputation
3. Outlier detection and removal
4. Feature scaling and normalization
5. Temporal train/val/test splitting

Author: Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for soil moisture prediction.
    
    Features:
    - Handles missing values
    - Detects and removes outliers
    - Normalizes/standardizes features
    - Creates temporal splits (no data leakage)
    - Generates statistics for paper reporting
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            outlier_method: 'iqr' or 'zscore'
            outlier_threshold: Threshold for outlier detection
        """
        self.scaling_method = scaling_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Scalers will be fit on training data only
        self.scalers = {}
        self.feature_stats = {}
        self.preprocessing_report = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with parsed timestamps
        """
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse timestamps
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by timestamp and farm
        df = df.sort_values(['Farm_ID', 'Timestamp']).reset_index(drop=True)
        
        print(f"✓ Loaded {len(df):,} records")
        print(f"✓ Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"✓ Farms: {df['Farm_ID'].unique().tolist()}")
        
        # Store initial stats
        self.preprocessing_report['total_records'] = len(df)
        self.preprocessing_report['date_range'] = (
            str(df['Timestamp'].min()), str(df['Timestamp'].max())
        )
        self.preprocessing_report['farms'] = df['Farm_ID'].unique().tolist()
        
        return df
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check and report missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame (no changes, just reporting)
        """
        print("\n### MISSING VALUE ANALYSIS ###")
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
            self.preprocessing_report['missing_values'] = missing_df.to_dict('records')
        else:
            print("✓ No missing values detected")
            self.preprocessing_report['missing_values'] = []
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, 
                       feature_cols: List[str]) -> pd.DataFrame:
        """
        Detect outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame
            feature_cols: List of numerical columns to check
            
        Returns:
            DataFrame with outliers marked
        """
        print(f"\n### OUTLIER DETECTION ({self.outlier_method.upper()}) ###")
        
        df_clean = df.copy()
        outlier_counts = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            data = df[col].values
            
            if self.outlier_method == 'iqr':
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.outlier_threshold * IQR)
                upper_bound = Q3 + (self.outlier_threshold * IQR)
                
                outliers = (data < lower_bound) | (data > upper_bound)
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                outliers = z_scores > self.outlier_threshold
            
            outlier_count = np.sum(outliers)
            outlier_pct = (outlier_count / len(data) * 100)
            
            if outlier_count > 0:
                outlier_counts[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2)
                }
                print(f"{col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
        
        self.preprocessing_report['outliers'] = outlier_counts
        
        if len(outlier_counts) == 0:
            print("✓ No significant outliers detected")
        
        return df_clean
    
    def scale_features(self, 
                      df: pd.DataFrame, 
                      feature_cols: List[str],
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to scale
            fit: If True, fit scaler. If False, use existing scaler
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            # Initialize scaler if needed
            if fit:
                if self.scaling_method == 'standard':
                    scaler = StandardScaler()
                elif self.scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.scaling_method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {self.scaling_method}")
                
                # Fit and transform
                df_scaled[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
                
                # Store statistics
                self.feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
            else:
                # Transform only
                if col in self.scalers:
                    df_scaled[col] = self.scalers[col].transform(df[[col]])
        
        return df_scaled
    
    def temporal_split(self, 
                      df: pd.DataFrame,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (no random shuffle to prevent data leakage).
        
        Args:
            df: Input DataFrame (must be sorted by timestamp)
            train_ratio: Fraction for training (0.7 = 70%)
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        # Split by farm to maintain temporal continuity
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for farm_id in df['Farm_ID'].unique():
            farm_df = df[df['Farm_ID'] == farm_id].reset_index(drop=True)
            n = len(farm_df)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_dfs.append(farm_df.iloc[:train_end])
            val_dfs.append(farm_df.iloc[train_end:val_end])
            test_dfs.append(farm_df.iloc[val_end:])
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print("\n### TEMPORAL DATA SPLIT ###")
        print(f"Training:   {len(train_df):>6,} records ({train_ratio*100:.1f}%)")
        print(f"            {train_df['Timestamp'].min()} to {train_df['Timestamp'].max()}")
        print(f"Validation: {len(val_df):>6,} records ({val_ratio*100:.1f}%)")
        print(f"            {val_df['Timestamp'].min()} to {val_df['Timestamp'].max()}")
        print(f"Testing:    {len(test_df):>6,} records ({test_ratio*100:.1f}%)")
        print(f"            {test_df['Timestamp'].min()} to {test_df['Timestamp'].max()}")
        
        self.preprocessing_report['data_split'] = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_dates': (str(train_df['Timestamp'].min()), str(train_df['Timestamp'].max())),
            'val_dates': (str(val_df['Timestamp'].min()), str(val_df['Timestamp'].max())),
            'test_dates': (str(test_df['Timestamp'].min()), str(test_df['Timestamp'].max()))
        }
        
        return train_df, val_df, test_df
    
    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Define feature groups for model training.
        
        Returns:
            Dictionary with feature column names
        """
        features = {
            # Primary sensor + environmental features for calibration
            'calibration_features': [
                'Raw_Capacitive_ADC',
                'Sensor_Voltage_V',
                'Sensor_Board_Temperature_C',
                'Soil_Temperature_C',
                'Atm_Temperature_C',
                'Atm_Pressure_inHg'
            ],
            
            # Temporal features
            'temporal_features': [
                'Hour', 'Day', 'Month', 'DayOfWeek', 
                'DayOfYear',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos'
            ],
            
            # Target variable
            'target': 'Volumetric_Moisture_Pct',
            
            # Auxiliary targets (government soil moisture metrics)
            'auxiliary_targets': [
                'SM_Level_15cm', 'SM_Volume_15cm',
                'SM_Aggregate_Pct', 'SM_Volume_Pct'
            ],
            
            # Categorical
            'categorical': ['Farm_ID'],
            
            # All numerical features
            'all_numerical': [
                'Raw_Capacitive_ADC', 'Sensor_Voltage_V',
                'Sensor_Board_Temperature_C',
                'Volumetric_Moisture_Pct', 'Soil_Temperature_C',
                'Atm_Temperature_C', 'Atm_Pressure_inHg',
                'SM_Level_15cm', 'SM_Volume_15cm',
                'SM_Aggregate_Pct', 'SM_Volume_Pct',
                'Hour', 'Day', 'Month', 'DayOfWeek',
                'DayOfYear',
                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos'
            ]
        }
        
        return features
    
    def preprocess_pipeline(self, 
                           filepath: str,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to dataset CSV
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Dictionary containing train/val/test DataFrames and metadata
        """
        print("="*80)
        print("DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Load data
        df = self.load_data(filepath)
        
        # Check missing values
        df = self.check_missing_values(df)
        
        # Get feature columns
        feature_groups = self.get_feature_columns()
        
        # Detect outliers (don't remove, just report)
        df = self.detect_outliers(df, feature_groups['all_numerical'])
        
        # Temporal split BEFORE scaling (to prevent data leakage)
        train_df, val_df, test_df = self.temporal_split(
            df, train_ratio, val_ratio, test_ratio
        )
        
        # Scale features (fit on train only)
        print("\n### FEATURE SCALING ###")
        print(f"Method: {self.scaling_method}")
        
        features_to_scale = feature_groups['calibration_features'] + ['Volumetric_Moisture_Pct']
        
        train_df_scaled = self.scale_features(train_df, features_to_scale, fit=True)
        val_df_scaled = self.scale_features(val_df, features_to_scale, fit=False)
        test_df_scaled = self.scale_features(test_df, features_to_scale, fit=False)
        
        print(f"✓ Scaled {len(features_to_scale)} features")
        
        # Summary statistics
        print("\n### PREPROCESSING SUMMARY ###")
        print(f"Total records: {len(df):,}")
        print(f"Training set: {len(train_df_scaled):,} ({train_ratio*100:.1f}%)")
        print(f"Validation set: {len(val_df_scaled):,} ({val_ratio*100:.1f}%)")
        print(f"Test set: {len(test_df_scaled):,} ({test_ratio*100:.1f}%)")
        print(f"Features scaled: {len(features_to_scale)}")
        print("\n" + "="*80)
        
        return {
            'train': train_df_scaled,
            'val': val_df_scaled,
            'test': test_df_scaled,
            'train_raw': train_df,
            'val_raw': val_df,
            'test_raw': test_df,
            'feature_groups': feature_groups,
            'scalers': self.scalers,
            'stats': self.feature_stats,
            'report': self.preprocessing_report
        }
    
    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable back to original scale.
        
        Args:
            scaled_values: Scaled moisture predictions
            
        Returns:
            Original scale values
        """
        target_col = 'Volumetric_Moisture_Pct'
        if target_col in self.scalers:
            return self.scalers[target_col].inverse_transform(
                scaled_values.reshape(-1, 1)
            ).flatten()
        else:
            return scaled_values


def main():
    """Test preprocessing pipeline."""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        outlier_method='iqr',
        outlier_threshold=3.0
    )
    
    # Run pipeline
    data_path = '../New_Dataset/kolkata_unified_dataset.csv'
    processed_data = preprocessor.preprocess_pipeline(
        filepath=data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Display sample
    print("\nSample of processed training data:")
    print(processed_data['train'].head())
    
    # Save processed data
    processed_data['train'].to_csv('../data/processed/train_data.csv', index=False)
    processed_data['val'].to_csv('../data/processed/val_data.csv', index=False)
    processed_data['test'].to_csv('../data/processed/test_data.csv', index=False)
    
    print("\n✓ Processed data saved to ../data/processed/")


if __name__ == "__main__":
    main()
