"""
Baseline Comparison: Traditional ML vs Deep Learning
=====================================================

Compares 8+ traditional ML baselines against our DL models
for soil moisture calibration task.

Baselines:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. SVR (RBF kernel)
6. Random Forest
7. Gradient Boosting (XGBoost-style)
8. K-Nearest Neighbors
9. Multi-layer Perceptron (sklearn)

Usage:
    python experiments/baseline_comparison.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import DataPreprocessor


def run_baseline_comparison(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Run all baseline models and compare."""
    
    print("=" * 80)
    print("BASELINE COMPARISON: Traditional ML vs Deep Learning")
    print("=" * 80)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(scaling_method='standard')
    processed = preprocessor.preprocess_pipeline(filepath=data_path)
    
    feature_groups = processed['feature_groups']
    # CAVEAT: This script uses ALL 15 features (calibration + temporal).
    # The manuscript's primary comparison uses only A2 features (ADC + Voltage).
    # For the A2-feature baselines, see: results/experiments/a2_recomputation_*.json
    feature_cols = feature_groups['calibration_features'] + feature_groups['temporal_features']
    target_col = feature_groups['target']
    
    # Prepare data
    X_train = processed['train_raw'][feature_cols].values
    y_train = processed['train_raw'][target_col].values
    X_val = processed['val_raw'][feature_cols].values
    y_val = processed['val_raw'][target_col].values
    X_test = processed['test_raw'][feature_cols].values
    y_test = processed['test_raw'][target_col].values
    
    # Scale features for baselines
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Define baselines
    baselines = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'SVR (RBF)': SVR(kernel='rbf', C=10.0, epsilon=0.1),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            n_jobs=-1, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
        'MLP (sklearn)': MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            solver='adam', max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42
        ),
    }
    
    results = {}
    
    for name, model in baselines.items():
        print(f"\n{'─' * 60}")
        print(f"Training: {name}")
        print(f"{'─' * 60}")
        
        start_time = time.time()
        
        try:
            model.fit(X_train_s, y_train)
            train_time = time.time() - start_time
            
            # Predictions
            y_pred_val = model.predict(X_val_s)
            y_pred_test = model.predict(X_test_s)
            
            # Metrics
            val_r2 = r2_score(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_mae = mean_absolute_error(y_val, y_pred_val)
            
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'val_r2': float(val_r2),
                'val_rmse': float(val_rmse),
                'val_mae': float(val_mae),
                'test_r2': float(test_r2),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'train_time_seconds': float(train_time),
                'status': 'success'
            }
            
            print(f"  Val  → R²={val_r2:.4f} | RMSE={val_rmse:.4f} | MAE={val_mae:.4f}")
            print(f"  Test → R²={test_r2:.4f} | RMSE={test_rmse:.4f} | MAE={test_mae:.4f}")
            print(f"  Time: {train_time:.2f}s")
            
        except Exception as e:
            results[name] = {'status': 'failed', 'error': str(e)}
            print(f"  ✗ FAILED: {e}")
    
    # Add DL results (from previous training runs)
    dl_results_dir = 'results/logs'
    if os.path.exists(dl_results_dir):
        for f in os.listdir(dl_results_dir):
            if f.endswith('.json'):
                with open(os.path.join(dl_results_dir, f)) as fp:
                    dl_data = json.load(fp)
                    model_name = f"DL: {dl_data.get('model', 'unknown')}"
                    if 'r2' in dl_data:
                        results[model_name] = {
                            'test_r2': dl_data['r2'],
                            'test_rmse': dl_data['rmse'],
                            'test_mae': dl_data['mae'],
                            'status': 'success'
                        }
    
    # Summary table
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Test R²':>10} {'Test RMSE':>10} {'Test MAE':>10} {'Time (s)':>10}")
    print("─" * 65)
    
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.get('status') == 'success' and 'test_r2' in v],
        key=lambda x: x[1]['test_r2'],
        reverse=True
    )
    
    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['test_r2']:>10.4f} {metrics['test_rmse']:>10.4f} "
              f"{metrics['test_mae']:>10.4f} {metrics.get('train_time_seconds', 0):>10.2f}")
    
    # Save results
    os.makedirs('results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/experiments/baseline_comparison_{timestamp}.json'
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_baseline_comparison()
