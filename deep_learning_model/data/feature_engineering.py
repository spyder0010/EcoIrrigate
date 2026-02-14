"""
Feature Engineering Module for Soil Moisture Prediction
=========================================================

Creates advanced features including:
1. Lag features (historical values)
2. Rolling statistics (moving averages, std)
3. Rate of change features
4. Interaction features
5. Sequence windows for LSTM/Transformer

Author: Research Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Advanced feature engineering for time series soil moisture prediction.
    """
    
    def __init__(self, lag_hours: List[int] = [1, 3, 6, 12, 24]):
        """
        Initialize feature engineer.
        
        Args:
            lag_hours: List of lag hours for historical features
        """
        self.lag_hours = lag_hours
        self.lag_steps = [h * 4 for h in lag_hours]  # 15-min intervals
        
    def create_lag_features(self, 
                           df: pd.DataFrame, 
                           columns: List[str]) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Args:
            df: Input DataFrame (must be sorted by Timestamp)
            columns: Columns to create lags for
            
        Returns:
            DataFrame with lag features
        """
        df_with_lags = df.copy()
        
        # Create lags for each farm separately
        for farm_id in df['Farm_ID'].unique():
            farm_mask = df['Farm_ID'] == farm_id
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                for lag_steps in self.lag_steps:
                    lag_hours = lag_steps / 4
                    lag_col_name = f'{col}_lag_{int(lag_hours)}h'
                    
                    # Create lag within farm group
                    df_with_lags.loc[farm_mask, lag_col_name] = \
                        df.loc[farm_mask, col].shift(lag_steps)
        
        return df_with_lags
    
    def create_rolling_features(self, 
                               df: pd.DataFrame, 
                               columns: List[str],
                               windows: List[int] = [4, 12, 24, 96]) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute rolling stats for
            windows: Window sizes in 15-min intervals (4=1h, 24=6h, 96=24h)
            
        Returns:
            DataFrame with rolling features
        """
        df_with_rolling = df.copy()
        
        for farm_id in df['Farm_ID'].unique():
            farm_mask = df['Farm_ID'] == farm_id
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                farm_series = df.loc[farm_mask, col]
                
                for window in windows:
                    window_hours = window / 4
                    
                    # Rolling mean
                    roll_mean = farm_series.rolling(
                        window=window, min_periods=1
                    ).mean()
                    df_with_rolling.loc[farm_mask, f'{col}_roll_mean_{int(window_hours)}h'] = roll_mean
                    
                    # Rolling std
                    roll_std = farm_series.rolling(
                        window=window, min_periods=1
                    ).std()
                    df_with_rolling.loc[farm_mask, f'{col}_roll_std_{int(window_hours)}h'] = roll_std
                    
                    # Rolling min/max
                    roll_min = farm_series.rolling(
                        window=window, min_periods=1
                    ).min()
                    df_with_rolling.loc[farm_mask, f'{col}_roll_min_{int(window_hours)}h'] = roll_min
                    
                    roll_max = farm_series.rolling(
                        window=window, min_periods=1
                    ).max()
                    df_with_rolling.loc[farm_mask, f'{col}_roll_max_{int(window_hours)}h'] = roll_max
        
        return df_with_rolling
    
    def create_rate_of_change_features(self, 
                                      df: pd.DataFrame, 
                                      columns: List[str]) -> pd.DataFrame:
        """
        Create rate of change (derivative) features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute derivatives for
            
        Returns:
            DataFrame with rate of change features
        """
        df_with_roc = df.copy()
        
        for farm_id in df['Farm_ID'].unique():
            farm_mask = df['Farm_ID'] == farm_id
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                farm_series = df.loc[farm_mask, col]
                
                # 1-hour change rate (4 steps = 1 hour)
                roc_1h = farm_series.diff(4)
                df_with_roc.loc[farm_mask, f'{col}_roc_1h'] = roc_1h
                
                # 3-hour change rate
                roc_3h = farm_series.diff(12)
                df_with_roc.loc[farm_mask, f'{col}_roc_3h'] = roc_3h
                
                # 6-hour change rate
                roc_6h = farm_series.diff(24)
                df_with_roc.loc[farm_mask, f'{col}_roc_6h'] = roc_6h
        
        return df_with_roc
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_with_interactions = df.copy()
    
        # Temperature * Pressure (weather patterns)
        if 'Atm_Temperature_C' in df.columns and 'Atm_Pressure_inHg' in df.columns:
            df_with_interactions['Temp_Pressure_interaction'] = \
                df['Atm_Temperature_C'] * df['Atm_Pressure_inHg']
        
        # Soil temp * Moisture (evaporation proxy)
        if 'Soil_Temperature_C' in df.columns and 'Volumetric_Moisture_Pct' in df.columns:
            df_with_interactions['SoilTemp_Moisture_interaction'] = \
                df['Soil_Temperature_C'] * df['Volumetric_Moisture_Pct']
        
        # ADC * Temperature (temperature compensation)
        if 'Raw_Capacitive_ADC' in df.columns and 'Soil_Temperature_C' in df.columns:
            df_with_interactions['ADC_Temp_interaction'] = \
                df['Raw_Capacitive_ADC'] * df['Soil_Temperature_C']
        
        # Thermal lag feature (key novelty: captures evapotranspiration signal)
        if 'Soil_Temperature_C' in df.columns and 'Atm_Temperature_C' in df.columns:
            df_with_interactions['Thermal_Lag'] = \
                df['Atm_Temperature_C'] - df['Soil_Temperature_C']
        
        # Sensor board self-heating indicator
        if 'Sensor_Board_Temperature_C' in df.columns and 'Atm_Temperature_C' in df.columns:
            df_with_interactions['Sensor_Heating'] = \
                df['Sensor_Board_Temperature_C'] - df['Atm_Temperature_C']
        
        # Pressure change rate (weather system indicator)
        if 'Atm_Pressure_inHg' in df.columns:
            for farm_id in df['Farm_ID'].unique():
                farm_mask = df['Farm_ID'] == farm_id
                # 4-hour pressure tendency (replaces rule-based rain prediction)
                pressure_diff_4h = df.loc[farm_mask, 'Atm_Pressure_inHg'].diff(16)  # 4-hour change
                df_with_interactions.loc[farm_mask, 'Pressure_Tendency_4h'] = pressure_diff_4h
                # 6-hour pressure change
                pressure_diff_6h = df.loc[farm_mask, 'Atm_Pressure_inHg'].diff(24)  # 6-hour change
                df_with_interactions.loc[farm_mask, 'Pressure_change_6h'] = pressure_diff_6h
        
        return df_with_interactions
    
    def create_sequence_windows(self, 
                               df: pd.DataFrame, 
                               feature_cols: List[str],
                               target_col: str,
                               sequence_length: int = 96,
                               forecast_horizons: List[int] = [4, 24, 48, 96]) -> Tuple:
        """
        Create sequence windows for LSTM/Transformer models.
        
        Args:
            df: Input DataFrame (sorted by timestamp)
            feature_cols: Input feature columns
            target_col: Target variable column
            sequence_length: Input sequence length (96 = 24 hours at 15-min intervals)
            forecast_horizons: Prediction horizons in steps [1h, 6h, 12h, 24h]
            
        Returns:
            Tuple of (X_sequences, y_targets, timestamps)
        """
        X_sequences = []
        y_targets = []
        timestamps = []
        farm_ids = []
        
        # Process each farm separately
        for farm_id in df['Farm_ID'].unique():
            farm_df = df[df['Farm_ID'] == farm_id].reset_index(drop=True)
            
            # Get feature and target arrays
            X = farm_df[feature_cols].values
            y = farm_df[target_col].values
            ts = farm_df['Timestamp'].values
            
            # Create sequences
            max_horizon = max(forecast_horizons)
            for i in range(sequence_length, len(X) - max_horizon):
                # Input sequence (past 24 hours)
                X_seq = X[i-sequence_length:i]
                
                # Targets at multiple horizons
                y_multi = []
                for horizon in forecast_horizons:
                    y_multi.append(y[i + horizon])
                
                X_sequences.append(X_seq)
                y_targets.append(y_multi)
                timestamps.append(ts[i])
                farm_ids.append(farm_id)
        
        X_sequences = np.array(X_sequences)  # Shape: (N, sequence_length, n_features)
        y_targets = np.array(y_targets)      # Shape: (N, n_horizons)
        
        return X_sequences, y_targets, timestamps, farm_ids
    
    def engineer_all_features(self, 
                             df: pd.DataFrame,
                             create_lags: bool = True,
                             create_rolling: bool = True,
                             create_roc: bool = True,
                             create_interactions: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            create_lags: Create lag features
            create_rolling: Create rolling statistics
            create_roc: Create rate of change features
            create_interactions: Create interaction features
            
        Returns:
            DataFrame with all engineered features
        """
        df_engineered = df.copy()
        
        print("\n### FEATURE ENGINEERING ###")
        initial_features = len(df.columns)
        
        # Core features to engineer
        core_features = [
            'Volumetric_Moisture_Pct',
            'Soil_Temperature_C',
            'Atm_Temperature_C',
            'Atm_Pressure_inHg',
            'Raw_Capacitive_ADC',
            'Sensor_Board_Temperature_C'
        ]
        
        if create_lags:
            print(f"Creating lag features for {self.lag_hours} hours...")
            df_engineered = self.create_lag_features(df_engineered, core_features)
        
        if create_rolling:
            print("Creating rolling window statistics...")
            df_engineered = self.create_rolling_features(
                df_engineered, 
                ['Volumetric_Moisture_Pct', 'Soil_Temperature_C', 'Atm_Pressure_inHg'],
                windows=[12, 24, 96]  # 3h, 6h, 24h
            )
        
        if create_roc:
            print("Creating rate of change features...")
            df_engineered = self.create_rate_of_change_features(
                df_engineered, 
                ['Volumetric_Moisture_Pct', 'Atm_Pressure_inHg', 'Soil_Temperature_C']
            )
        
        if create_interactions:
            print("Creating interaction features...")
            df_engineered = self.create_interaction_features(df_engineered)
        
        final_features = len(df_engineered.columns)
        new_features = final_features - initial_features
        
        print(f"✓ Created {new_features} new features")
        print(f"✓ Total features: {final_features}")
        
        # Drop rows with NaN from lag/rolling operations
        original_len = len(df_engineered)
        df_engineered = df_engineered.dropna().reset_index(drop=True)
        dropped = original_len - len(df_engineered)
        
        if dropped > 0:
            print(f"✓ Dropped {dropped} rows with NaN from feature engineering")
        
        return df_engineered


def main():
    """Test feature engineering."""
    
    # Load preprocessed data
    train_df = pd.read_csv('../data/processed/train_data.csv')
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    
    # Initialize feature engineer
    engineer = FeatureEngineer(lag_hours=[1, 3, 6, 12, 24])
    
    # Apply feature engineering
    train_engineered = engineer.engineer_all_features(
        train_df,
        create_lags=True,
        create_rolling=True,
        create_roc=True,
        create_interactions=True
    )
    
    print("\nSample of engineered features:")
    print(train_engineered.columns.tolist()[:30])
    print(f"\nShape: {train_engineered.shape}")
    
    # Save
    train_engineered.to_csv('../data/processed/train_engineered.csv', index=False)
    print("\n✓ Engineered data saved")


if __name__ == "__main__":
    main()
