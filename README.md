# EcoIrrigate: Multi-Task Deep Learning for IoT Soil Moisture Calibration and Forecasting

> **Companion repository** for the manuscript:
> *"Multi-Task Deep Learning Framework for IoT Soil Moisture Calibration and Forecasting in Urban Agriculture"*
>
> **Authors:** Soham Saha · Sneha Paul · Ahona Ghosh · Shalini Shaw · Arpan Deyasi  
> **Affiliation:** Department of Computer Science and Engineering (IoT, CS, BT), Institute of Engineering and Management, Kolkata 700160, India  
> **Corresponding Author:** Soham Saha ([sohamsaha568@gmail.com](mailto:sohamsaha568@gmail.com))  
> **Repository:** [https://github.com/spyder0010/EcoIrrigate](https://github.com/spyder0010/EcoIrrigate)

---

## Abstract

This repository provides the complete source code, dataset, trained model weights, and experiment scripts to reproduce all results reported in the accompanying manuscript. The framework implements a multi-task BiLSTM-Attention architecture for joint sensor calibration and multi-horizon soil moisture forecasting using IoT sensor data from two urban agricultural sites in Kolkata, India. Novel contributions include the integration of atmospheric pressure tendency and thermal lag features, farm-specific embeddings for inter-sensor variability, and uncertainty-aware multi-task learning with automatic task weighting.

---

## Repository Structure

```
EcoIrrigate/
├── .gitignore
├── README.md                                  ← This file
│
├── New_Dataset/
│   ├── kolkata_unified_dataset.csv            ← Primary dataset (21,312 records)
│   ├── daily_pressure_extracted.csv           ← Atmospheric pressure observations
│   ├── daily_temperature_extracted.csv        ← Atmospheric temperature observations
│   └── QUALITY_REPORT.json                    ← Data quality audit results
│
└── deep_learning_model/
    ├── main.py                                ← Unified training entry point
    ├── requirements.txt                       ← Python dependencies
    │
    ├── data/
    │   ├── preprocessing.py                   ← Data loading, cleaning, scaling, splitting
    │   ├── feature_engineering.py             ← Lag, rolling, rate-of-change, interaction features
    │   └── data_loader.py                     ← PyTorch Dataset/DataLoader classes
    │
    ├── models/
    │   ├── architectures.py                   ← CalibrationNet, ForecastingNet, MultiTaskNet
    │   └── losses.py                          ← Huber, WeightedMSE, MultiTask, Focal, Quantile
    │
    ├── experiments/
    │   ├── __init__.py
    │   ├── ablation_study.py                  ← 7-configuration feature ablation
    │   ├── baseline_comparison.py             ← 9 traditional ML baselines
    │   ├── cross_farm_validation.py           ← Leave-one-farm-out generalization
    │   ├── rule_vs_dl_comparison.py           ← Rule-based vs. DL forecasting
    │   └── generate_figures.py                ← All 10 publication figures + SHAP + stats
    │
    └── results/
        ├── experiments/                       ← JSON experiment outputs
        ├── figures/                           ← 10 publication-quality PNG figures
        ├── logs/                              ← Training logs (JSON)
        └── models/                            ← Trained model checkpoints (.pt)
```

---

## Dataset

| Property | Value |
|----------|-------|
| **Records** | 21,312 |
| **Farms** | 2 (`Kolkata_Farm_1`, `Kolkata_Farm_2`) |
| **Period** | January 5 – April 25, 2025 (110 days) |
| **Temporal resolution** | 15-minute intervals |
| **Features** | 22 columns (see below) |
| **Target variable** | `Volumetric_Moisture_Pct` |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `Raw_Capacitive_ADC` | int | 10-bit ADC reading from capacitive sensor |
| `Sensor_Voltage_V` | float | Analog voltage output |
| `Sensor_Board_Temperature_C` | float | On-board temperature |
| `Atm_Temperature_C` | float | Atmospheric temperature (°C) |
| `Soil_Temperature_C` | float | Soil temperature at 15 cm depth (°C) |
| `Atm_Pressure_inHg` | float | Barometric pressure (inHg) |
| `Volumetric_Moisture_Pct` | float | Ground-truth volumetric soil moisture (%) |
| `SM_Level_15cm`, `SM_Volume_15cm`, `SM_Aggregate_Pct`, `SM_Volume_Pct` | float | Supplementary soil moisture indicators |
| `Hour`, `Day`, `Month`, `DayOfWeek`, `DayOfYear` | int | Calendar features |
| `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos` | float | Cyclic temporal encodings |
| `Farm_ID` | str | Farm identifier |

### Data Split (Temporal, No Shuffle)

| Split | Ratio | Purpose |
|-------|-------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (reported in manuscript) |

---

## Model Architectures

### CalibrationNet
Point-wise sensor calibration: Raw ADC + environmental features → calibrated moisture percentage.

- Multi-layer perceptron with residual connections
- Farm-specific learned embeddings (16-dim)
- Batch normalization + dropout (0.3)
- Huber loss (δ = 1.0)
- **Parameters:** ~14,000

### ForecastingNet
Multi-horizon moisture forecasting from 24-hour historical sequences.

- 2-layer bidirectional LSTM (128 hidden units)
- Temporal attention mechanism over sequence outputs
- Separate prediction heads for each horizon (+1h, +6h, +12h, +24h)
- Weighted MSE loss (near-term weighted higher)
- **Parameters:** ~1.1M

### MultiTaskNet
Joint calibration + forecasting via shared encoder with task-specific heads.

- Shared BiLSTM encoder
- CalibrationHead and ForecastingHead decoders
- Uncertainty-based automatic task weighting (Kendall et al., 2018)
- **Parameters:** ~1.2M

---

## Installation

### Requirements

- Python ≥ 3.8
- CUDA-compatible GPU (optional but recommended)

```bash
# Clone repository
git clone https://github.com/spyder0010/EcoIrrigate.git
cd EcoIrrigate/deep_learning_model

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) For GPU acceleration with CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Reproducing Results

All commands should be executed from `deep_learning_model/`.

### 1. Train Models

```bash
# Calibration model
python main.py --task calibration --model CalibrationNet --epochs 100 --lr 0.001

# Forecasting model
python main.py --task forecasting --model ForecastingNet --epochs 100 --lr 0.001

# Multi-task model
python main.py --task multi-task --model MultiTaskNet --epochs 100 --lr 0.001
```

Training uses AdamW optimizer with cosine annealing learning rate schedule and early stopping (patience = 15 epochs). Model checkpoints are saved to `results/models/`.

### 2. Run Experiments

```bash
# 7-configuration ablation study (Table III in manuscript)
python experiments/ablation_study.py

# 9-baseline comparison (Table II in manuscript)
python experiments/baseline_comparison.py

# Cross-farm generalization test (Table IV in manuscript)
python experiments/cross_farm_validation.py

# Rule-based vs. DL forecasting comparison (Table V in manuscript)
python experiments/rule_vs_dl_comparison.py
```

Results are saved as JSON files in `results/experiments/`.

### 3. Generate Figures

```bash
# Generate all 10 publication figures + SHAP analysis + statistical tests
python experiments/generate_figures.py
```

Figures are saved at 300 DPI to `results/figures/`.

---

## Key Results

### Calibration Performance (Test Set)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| CalibrationNet (A2: ADC + Voltage) | 0.947 | 0.483 | 0.392 |
| MultiTaskNet | 0.872 | 0.542 | 0.459 |
| Random Forest (best baseline) | 0.885 | 0.712 | 0.548 |
| Gradient Boosting | 0.333 | 1.716 | 1.602 |
| Linear Regression | −5.646 | 5.416 | 5.253 |

### Multi-Horizon Forecasting (ForecastingNet)

| Horizon | RMSE | MAE |
|---------|------|-----|
| +1 hour | 1.484 | 1.309 |
| +6 hours | 1.865 | 1.626 |
| +12 hours | 2.164 | 1.891 |
| +24 hours | 2.561 | 2.275 |

### Cross-Farm Generalization

| Experiment | R² | RMSE | MAE |
|------------|-----|------|-----|
| Train Farm 1 → Test Farm 2 | 0.934 | 0.540 | 0.356 |
| Train Farm 2 → Test Farm 1 | 0.917 | 0.559 | 0.402 |

---

## Publication Figures

All figures in the manuscript are generated by `experiments/generate_figures.py`:

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | `fig1_calibration_scatter.png` | Predicted vs. actual moisture scatter |
| Fig. 2 | `fig2_multi_horizon_forecast.png` | Multi-horizon forecasting performance |
| Fig. 3 | `fig3_ablation_study.png` | 7-configuration ablation results |
| Fig. 4 | `fig4_shap_importance.png` | SHAP feature importance analysis |
| Fig. 5 | `fig5_timeseries_overlay.png` | 7-day prediction vs. ground truth overlay |
| Fig. 6 | `fig6_cross_farm.png` | Cross-farm generalization results |
| Fig. 7 | `fig7_baseline_comparison.png` | Baseline model comparison |
| Fig. 8 | `fig8_thermal_lag.png` | Atmospheric–soil thermal lag visualization |
| Fig. 9 | `fig9_rule_vs_dl.png` | Rule-based vs. DL forecasting comparison |
| Fig. 10 | `fig10_training_convergence.png` | Training/validation convergence curves |

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{saha2026ecoirrigate,
  title     = {Multi-Task Deep Learning Framework for {IoT} Soil Moisture 
               Calibration and Forecasting in Urban Agriculture},
  author    = {Saha, Soham and Paul, Sneha and Ghosh, Ahona and Shaw, Shalini 
               and Deyasi, Arpan},
  year      = {2026},
  note      = {Under Review}
}
```

---

## License

This repository is released for **academic and non-commercial research purposes only**. Commercial use requires prior written permission from the authors.

---

## Acknowledgements

- Dataset collected from IoT sensor installations at two urban agricultural sites in Kolkata, West Bengal, India.
- Research conducted at the Department of Computer Science and Engineering (IoT, CS, BT), Institute of Engineering and Management, Kolkata.
