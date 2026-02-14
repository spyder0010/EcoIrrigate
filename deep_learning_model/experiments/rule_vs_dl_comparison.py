"""
Rule-based vs Deep Learning Comparison
=======================================

Compares the original manuscript's rule-based rain prediction/irrigation
logic (simple humidity/temperature thresholds from Gutierrez-Lopez et al.)
against the DL model for moisture forecasting.

This directly demonstrates the upgrade path from IoT rule-based control
to deep learning intelligence.

Rule-based approaches:
1. Threshold-based: If moisture < threshold → irrigate
2. Moving average: If moisture drops below rolling mean - k*std → irrigate  
3. Weather-rule: If pressure drops (rain coming) → skip irrigation
4. Combined heuristic: Threshold + weather awareness

These are compared against the DL ForecastingNet for ability to
predict future soil moisture.

Usage:
    python experiments/rule_vs_dl_comparison.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import DataPreprocessor


def persistence_forecast(y, horizons=[4, 24, 48, 96]):
    """Persistence model: predict current value for all horizons."""
    max_h = max(horizons)
    n = len(y) - max_h
    predictions = {h: np.zeros(n) for h in horizons}
    actuals = {h: np.zeros(n) for h in horizons}
    
    for i in range(n):
        for h in horizons:
            predictions[h][i] = y[i]  # Current value as prediction
            actuals[h][i] = y[i + h]  # Actual future value
    
    return predictions, actuals


def moving_average_forecast(y, window=24, horizons=[4, 24, 48, 96]):
    """Moving average model: predict rolling mean as future value."""
    max_h = max(horizons)
    start = max(window, 96)  # Need enough history
    n = len(y) - max_h - start
    predictions = {h: np.zeros(n) for h in horizons}
    actuals = {h: np.zeros(n) for h in horizons}
    
    for i in range(n):
        idx = i + start
        ma = np.mean(y[idx-window:idx])
        for h in horizons:
            predictions[h][i] = ma
            actuals[h][i] = y[idx + h]
    
    return predictions, actuals


def linear_trend_forecast(y, window=24, horizons=[4, 24, 48, 96]):
    """Linear trend: extrapolate from recent slope."""
    max_h = max(horizons)
    start = max(window, 96)
    n = len(y) - max_h - start
    predictions = {h: np.zeros(n) for h in horizons}
    actuals = {h: np.zeros(n) for h in horizons}
    
    for i in range(n):
        idx = i + start
        recent = y[idx-window:idx]
        x = np.arange(window)
        slope = np.polyfit(x, recent, 1)[0]
        
        for h in horizons:
            predictions[h][i] = y[idx] + slope * h
            actuals[h][i] = y[idx + h]
    
    return predictions, actuals


def threshold_irrigation_rule(y, pressure, temperature, 
                               moisture_threshold=25.0,
                               pressure_drop_threshold=-0.02,
                               horizons=[4, 24, 48, 96]):
    """
    Combined rule-based prediction inspired by original EcoIrrigate manuscript.
    
    Rules:
    1. If current moisture is stable: predict persistence
    2. If pressure is dropping (rain coming): predict moisture increase
    3. If temperature is high and moisture low: predict dry-down
    """
    max_h = max(horizons)
    window = 24  # 6 hours of 15-min data
    start = max(window, 96)
    n = len(y) - max_h - start
    predictions = {h: np.zeros(n) for h in horizons}
    actuals = {h: np.zeros(n) for h in horizons}
    
    for i in range(n):
        idx = i + start
        current_moisture = y[idx]
        
        # Calculate recent moisture trend
        recent_moisture = y[idx-window:idx]
        moisture_trend = (recent_moisture[-1] - recent_moisture[0]) / window
        
        # Pressure trend (4h = 16 steps)
        if idx >= 16:
            pressure_trend = pressure[idx] - pressure[idx-16]
        else:
            pressure_trend = 0
        
        for h in horizons:
            # Base prediction: persistence with trend
            pred = current_moisture + moisture_trend * h
            
            # Rule 1: If pressure dropping significantly → expect rain → moisture increase
            if pressure_trend < pressure_drop_threshold:
                rain_boost = abs(pressure_trend) * 50 * min(h, 24) / 24
                pred += rain_boost
            
            # Rule 2: If high temperature and low moisture → faster dry-down
            if temperature[idx] > 30 and current_moisture < moisture_threshold:
                evap_factor = (temperature[idx] - 25) * 0.01 * h
                pred -= evap_factor
            
            # Clamp to reasonable range
            pred = np.clip(pred, 0, 100)
            
            predictions[h][i] = pred
            actuals[h][i] = y[idx + h]
    
    return predictions, actuals


def run_rule_vs_dl_comparison(data_path='../New_Dataset/kolkata_unified_dataset.csv'):
    """Compare rule-based methods against DL for moisture forecasting."""
    
    print("=" * 80)
    print("RULE-BASED vs DEEP LEARNING COMPARISON")
    print("=" * 80)
    
    # Load data
    preprocessor = DataPreprocessor(scaling_method='standard')
    processed = preprocessor.preprocess_pipeline(filepath=data_path)
    
    # Use test set for comparison
    test_df = processed['test_raw'].copy()
    
    y = test_df['Volumetric_Moisture_Pct'].values
    pressure = test_df['Atm_Pressure_inHg'].values
    temperature = test_df['Atm_Temperature_C'].values
    
    horizons = [4, 24, 48, 96]  # 1h, 6h, 12h, 24h
    horizon_labels = {4: '1h', 24: '6h', 48: '12h', 96: '24h'}
    
    methods = {
        'Persistence': lambda: persistence_forecast(y, horizons),
        'Moving Average (6h)': lambda: moving_average_forecast(y, window=24, horizons=horizons),
        'Linear Trend': lambda: linear_trend_forecast(y, window=24, horizons=horizons),
        'Rule-based (EcoIrrigate)': lambda: threshold_irrigation_rule(
            y, pressure, temperature, horizons=horizons
        ),
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n{'─' * 60}")
        print(f"Method: {method_name}")
        print(f"{'─' * 60}")
        
        predictions, actuals = method_func()
        
        method_results = {}
        for h in horizons:
            r2 = r2_score(actuals[h], predictions[h])
            rmse = np.sqrt(mean_squared_error(actuals[h], predictions[h]))
            mae = mean_absolute_error(actuals[h], predictions[h])
            
            method_results[horizon_labels[h]] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'n_samples': len(actuals[h])
            }
            
            print(f"  {horizon_labels[h]:>4}: R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")
        
        results[method_name] = method_results
    
    # Add DL results if available
    dl_dir = 'results/logs'
    if os.path.exists(dl_dir):
        for f in os.listdir(dl_dir):
            if 'ForecastingNet' in f and f.endswith('.json'):
                with open(os.path.join(dl_dir, f)) as fp:
                    dl_data = json.load(fp)
                    if 'horizons' in dl_data:
                        dl_results = {}
                        for h_name, h_metrics in dl_data['horizons'].items():
                            dl_results[h_name] = {
                                'rmse': h_metrics['rmse'],
                                'mae': h_metrics['mae']
                            }
                        results['DL: ForecastingNet'] = dl_results
                        print(f"\n{'─' * 60}")
                        print(f"Method: DL ForecastingNet (from training logs)")
                        print(f"{'─' * 60}")
                        for h_name, h_metrics in dl_results.items():
                            print(f"  {h_name:>4}: RMSE={h_metrics['rmse']:.4f} | MAE={h_metrics['mae']:.4f}")
                break
    
    # Summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (RMSE by horizon)")
    print("=" * 80)
    print(f"\n{'Method':<30} {'1h':>8} {'6h':>8} {'12h':>8} {'24h':>8}")
    print("─" * 62)
    
    for method_name, method_results in results.items():
        row = f"{method_name:<30}"
        for h in ['1h', '6h', '12h', '24h']:
            if h in method_results:
                row += f" {method_results[h]['rmse']:>8.4f}"
            else:
                row += f" {'N/A':>8}"
        print(row)
    
    # Save
    os.makedirs('results/experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/experiments/rule_vs_dl_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_rule_vs_dl_comparison()
