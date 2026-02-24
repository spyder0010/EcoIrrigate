# EcoIrrigate: Multi-Task Deep Learning for IoT Soil Moisture Calibration and Forecasting

> **Companion repository** for the manuscript:
> *"Multi-Task Deep Learning Framework for IoT Soil Moisture Calibration and Forecasting in Urban Agriculture"*
>
> **Authors:** Soham Saha · Subhojit Kar · Arijeet Ghosh · Sohan Bhattacharjee · Somes Sanyal  
> **Affiliation:** Department of Computer Science and Engineering (IoT, CS, BT), Institute of Engineering and Management, Kolkata 700160, India  
> **Corresponding Author:** Arijeet Ghosh ([arijeet.mtece.12@gmail.com](mailto:arijeet.mtece.12@gmail.com))  
> **Repository:** [https://github.com/spyder0010/EcoIrrigate](https://github.com/spyder0010/EcoIrrigate)

---

## What This Project Does (Non-Technical Summary)

**EcoIrrigate** is a smart irrigation system that uses low-cost soil sensors and artificial intelligence to:

1. **Measure soil moisture accurately** — A cheap capacitive sensor reads raw electrical signals from the soil. Our AI model converts these raw numbers into precise soil moisture percentages, correcting for temperature and other environmental effects.

2. **Predict future soil moisture** — Using the last 24 hours of sensor data, the model forecasts how wet or dry the soil will be in 1 hour, 6 hours, 12 hours, and 24 hours from now.

3. **Work across different farms** — The model was trained on data from two real farms in Kolkata, India, and can transfer what it learned from one farm to another without retraining.

### Key Highlights

- **96% accuracy** in converting raw sensor readings to moisture percentages (R² = 0.956)
- **24-hour forecasting** with graceful accuracy degradation over longer time horizons
- **Low-cost hardware** — Arduino Uno R4 WiFi + one capacitive soil sensor + one DHT22 temperature sensor (total cost under ₹2,000 / ~$25)
- **No barometer or soil thermometer needed** — Atmospheric pressure comes from public weather reports (METAR), and soil temperature is estimated using a physics formula
- **Fully reproducible** — All code, data, and trained models are included below

---

## Repository Structure

```
EcoIrrigate/
├── README.md                                  ← This file
│
├── New_Dataset/
│   ├── kolkata_unified_dataset.csv            ← Primary dataset (21,312 records)
│   ├── daily_pressure_extracted.csv           ← Atmospheric pressure observations
│   ├── daily_temperature_extracted.csv        ← Atmospheric temperature observations
│   ├── QUALITY_REPORT.json                    ← Data quality audit results
│   └── DATA_DICTIONARY.md                     ← Feature descriptions and metadata
│
├── graphical_abstract/                        ← Graphical abstract generator
│   ├── generate_graphical_abstract.py         ← Python script (matplotlib + numpy)
│   └── results/
│       ├── graphical_abstract.png             ← Graphical abstract (PNG, 300 DPI)
│       ├── graphical_abstract.tiff            ← Graphical abstract (TIFF, 300 DPI)
│       └── graphical_abstract.pdf             ← Graphical abstract (PDF, 300 DPI)
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
    │   ├── architectures.py                   ← CalibrationNet, ForecastingNet, MultiModalFusionNet, MultiTaskNet
    │   └── losses.py                          ← Huber, WeightedMSE, MultiTask (+ experimental: Focal, Quantile)
    │
    ├── experiments/
    │   ├── __init__.py
    │   ├── ablation_study.py                  ← 7-configuration feature ablation
    │   ├── baseline_comparison.py             ← 9 traditional ML baselines
    │   ├── cross_farm_validation.py           ← Leave-one-farm-out generalization
    │   ├── rule_vs_dl_comparison.py           ← Rule-based vs. DL forecasting
    │   └── generate_figures.py                ← All 14 publication figures + SHAP + stats
    │
    ├── hardware/
    │   └── data_logging/
    │       └── data_logging.ino               ← Arduino firmware for sensor data collection
    │
    └── results/
        ├── experiments/                       ← JSON experiment outputs
        ├── figures/                           ← 14 publication-quality PNG figures (10 main + 4 supplementary)
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
| `Sensor_Board_Temperature_C` | float | On-board temperature from DHT22. Reads ~2.5°C above true ambient due to self-heating |
| `Atm_Temperature_C` | float | Atmospheric temperature (°C), sourced from DHT22 ambient reading |
| `Soil_Temperature_C` | float | Soil temperature at 15 cm depth, estimated via Fourier thermal diffusion model (°C) |
| `Atm_Pressure_inHg` | float | Barometric pressure interpolated from METAR reports at VECC Airport (inHg) |
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

### Mendeley Data Repository

The complete dataset is archived on Mendeley Data for long-term preservation and citability:

| Property | Value |
|----------|-------|
| **DOI** | [10.17632/8x4v7yy3ds.1](https://doi.org/10.17632/8x4v7yy3ds.1) |
| **Title** | EcoIrrigate: Multi-Sensor Soil Moisture Monitoring Dataset from Kolkata, India (Jan–Apr 2025) |
| **License** | CC BY 4.0 |

---

## Model Architectures

### CalibrationNet
Point-wise sensor calibration: Raw ADC + environmental features → calibrated moisture percentage.

- Multi-layer perceptron with batch normalization and dropout (0.3)
- Farm-specific learned embeddings (16-dim)
- Huber loss (δ = 1.0)
- **Parameters:** ~47,000 (A2 config with 2 input features)

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

## Step-by-Step Replication Guide

Follow these steps **exactly** to reproduce every result and figure in the paper.

### Step 0: Prerequisites

You need the following installed on your computer:

| Software | Version | Purpose |
|----------|---------|---------|
| Python | ≥ 3.8 | Running the code |
| pip | latest | Installing Python packages |
| Git | any | Cloning the repository |
| (Optional) NVIDIA GPU + CUDA | 11.8+ | Faster training (~10× speedup) |

### Step 1: Clone and Set Up

```bash
# 1a. Clone the repository
git clone https://github.com/spyder0010/EcoIrrigate.git
cd EcoIrrigate

# 1b. Create an isolated Python environment
python -m venv venv

# 1c. Activate the environment
#     On Linux/macOS:
source venv/bin/activate
#     On Windows (Command Prompt):
venv\Scripts\activate
#     On Windows (PowerShell):
venv\Scripts\Activate.ps1

# 1d. Install all required packages
cd deep_learning_model
pip install -r requirements.txt
```

> **GPU users:** If you have an NVIDIA GPU, also run:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 2: Verify the Dataset

Before training, confirm the dataset is present and intact:

```bash
# Should print: 21313 (21,312 data rows + 1 header row)
#     On Linux/macOS:
wc -l ../New_Dataset/kolkata_unified_dataset.csv
#     On Windows (PowerShell):
(Get-Content ..\New_Dataset\kolkata_unified_dataset.csv | Measure-Object -Line).Lines
```

### Step 3: Train the Models

All commands below should be run from the `deep_learning_model/` directory.

```bash
# 3a. Train the Calibration model (~2-5 min on GPU, ~15-30 min on CPU)
python main.py --task calibration --model CalibrationNet --epochs 100 --lr 0.001

# 3b. Train the Forecasting model (~10-20 min on GPU, ~1-2 hours on CPU)
python main.py --task forecasting --model ForecastingNet --epochs 100 --lr 0.001

# 3c. Train the Multi-Task model (~15-30 min on GPU, ~2-3 hours on CPU)
python main.py --task multi-task --model MultiTaskNet --epochs 100 --lr 0.001
```

**What to expect:**
- Training progress prints every 5 epochs
- Best model checkpoint is saved to `results/models/` whenever validation loss improves
- Training stops automatically when validation loss stops improving (early stopping, patience = 15 epochs)
- Final test metrics (R², RMSE, MAE) are printed and saved to `results/logs/`

### Step 4: Run All Experiments

```bash
# 4a. Ablation study — tests 7 feature configurations (Table III, ~30-60 min)
python experiments/ablation_study.py

# 4b. Baseline comparison — trains 9 ML models (Table II, ~5-10 min)
python experiments/baseline_comparison.py

# 4c. Cross-farm validation — trains on Farm 1 and tests on Farm 2, and vice versa (Table IV, ~10-20 min)
python experiments/cross_farm_validation.py

# 4d. Rule-based vs. DL forecasting comparison (Table V, ~5 min)
python experiments/rule_vs_dl_comparison.py
```

Each experiment saves its results as a JSON file in `results/experiments/`.

### Step 5: Generate All Figures

```bash
# Generates all 14 publication figures + SHAP analysis + statistical tests
python experiments/generate_figures.py
```

Figures are saved at 300 DPI to `results/figures/`. This step also runs SHAP analysis and bootstrap statistical tests.

### Step 6: Verify Your Results

Compare your generated files against the values in the tables below. Due to stochastic training, single-seed results may vary slightly. The multi-seed ablation study (Step 4a) averages over 5 random seeds to reduce variance.

---

## Key Results

### Calibration Performance (Test Set, A2 Features)

All models below are trained on the optimal A2 feature set (Raw ADC + Sensor Voltage) identified by ablation study. CalibrationNet values report the 5-seed mean ± std.

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| MLP (sklearn, 256-128-64) | **0.965** | 0.307 | 0.244 |
| CalibrationNet (A2, 5-seed mean) | 0.956 ± 0.007 | 0.341 | 0.268 |
| Gradient Boosting | 0.954 | 0.351 | 0.266 |
| Random Forest | 0.953 | 0.354 | 0.269 |
| KNN (k=10) | 0.949 | 0.370 | 0.267 |
| SVR (RBF) | 0.916 | 0.473 | 0.300 |
| Linear Regression | 0.713 | 0.876 | 0.654 |

> **Note:** The sklearn MLP achieves a marginally higher point-estimate R² than CalibrationNet on A2 features. CalibrationNet is preferred in this framework for its farm-specific embeddings, multi-task integration capability, and compatibility with the joint training architecture.

### Multi-Horizon Forecasting (ForecastingNet)

| Horizon | RMSE | MAE |
|---------|------|-----|
| +1 hour | 1.484 | 1.309 |
| +6 hours | 1.865 | 1.626 |
| +12 hours | 2.164 | 1.891 |
| +24 hours | 2.561 | 2.275 |

> These values are from the best single-run checkpoint logged in `results/logs/`.

### Cross-Farm Generalization

| Experiment | R² | RMSE | MAE |
|------------|-----|------|-----|
| Train Farm 1 → Test Farm 2 | 0.934 | 0.540 | 0.356 |
| Train Farm 2 → Test Farm 1 | 0.917 | 0.559 | 0.402 |
| Combined (reference) | 0.812 | 1.060 | 0.840 |

### Ablation Study Summary (5 seeds per config)

| Config | Features | R² (mean ± std) | 95% CI |
|--------|----------|------------------|--------|
| A1 | ADC only | 0.946 ± 0.019 | [0.914, 0.963] |
| **A2** | **+ Voltage** | **0.956 ± 0.007** | **[0.943, 0.963]** |
| A3 | + Board Temp | 0.936 ± 0.022 | [0.898, 0.955] |
| A4 | + Atm Temp | 0.947 ± 0.013 | [0.926, 0.962] |
| A5 | + Soil Temp | 0.924 ± 0.022 | [0.890, 0.949] |
| A6 | + Pressure | 0.906 ± 0.018 | [0.880, 0.933] |
| A7 | All features | 0.596 ± 0.091 | [0.450, 0.693] |

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
| Supp. S1 | `fig_ablation_confidence_intervals.png` | Ablation R² with 95% confidence intervals |
| Supp. S2 | `fig_ablation_training_curves.png` | Training loss/R² convergence per ablation config |
| Supp. S3 | `fig_sensitivity_lambda.png` | Multi-task loss weight (λ) sensitivity analysis |
| Supp. S4 | `fig_mtl_feature_importance.png` | MTL vs. standalone feature importance ratios |

---

## Hardware Setup

The IoT data collection uses the following components:

| Component | Model | Purpose |
|-----------|-------|---------|
| Microcontroller | Arduino Uno R4 WiFi | Data acquisition and logging |
| Soil Moisture Sensor | Capacitive v1.2 (Analog A0) | Raw ADC + voltage measurement |
| Temperature/Humidity | DHT22 (Digital Pin 4) | Ambient temperature (used as board temp proxy) |
| Actuator | Relay Module (Digital Pin 3) | Water pump control |

> **Important:** Atmospheric pressure and soil temperature are **not** measured by the hardware. They are obtained from public METAR weather reports and a Fourier thermal diffusion model, respectively.

The Arduino firmware is located at `deep_learning_model/hardware/data_logging/data_logging.ino`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` from `deep_learning_model/` |
| `FileNotFoundError: kolkata_unified_dataset.csv` | Make sure you're running from `deep_learning_model/` (the dataset is at `../New_Dataset/`) |
| Training is very slow | Use a GPU. If unavailable, reduce epochs: `--epochs 30` |
| Results differ from paper | Single-seed runs vary. Run the ablation study (Step 4a) for multi-seed averages |
| CUDA out of memory | Reduce batch size: `--batch-size 32` or `--batch-size 16` |

---

## Graphical Abstract Replication

The graphical abstract can be regenerated from the Python script included in the repository:

```bash
cd graphical_abstract
pip install matplotlib numpy pillow
python generate_graphical_abstract.py
```

This produces `results/graphical_abstract.png`, `results/graphical_abstract.tiff`, and `results/graphical_abstract.pdf` at 300 DPI (≥1328×531 px, 2.5:1 aspect ratio) compliant with Elsevier specifications.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{saha2026ecoirrigate,
  title     = {Multi-Task Deep Learning Framework for {IoT} Soil Moisture 
               Calibration and Forecasting in Urban Agriculture},
  author    = {Saha, Soham and Kar, Subhojit and Ghosh, Arijeet and Bhattacharjee, Sohan 
               and Sanyal, Somes},
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
