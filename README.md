# EcoIrrigate: Multi-Task Deep Learning for IoT Soil Moisture Calibration and Forecasting

> **Companion repository** for the manuscript:
> *"A Multi-Task Deep Learning Framework for Sensor Calibration and Multi-Horizon Soil Moisture Prediction in IoT Irrigation Systems"*
>
> **Authors:** Soham Saha · Subhojit Kar · Arijeet Ghosh · Sohan Bhattacharjee · Somes Sanyal · Avik Kumar Das  
> **Affiliation:** Department of Computer Science and Engineering (IoT, CS, BT), Institute of Engineering and Management, Kolkata 700160, India  
> **Corresponding Author:** Soham Saha ([sohamsaha568@gmail.com](mailto:sohamsaha568@gmail.com))  
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
    │   ├── architectures.py                   ← CalibrationNet, ForecastingNet, MultiTaskNet
    │   ├── baseline_architectures.py          ← TCN, ConvLSTM, Transformer variants
    │   └── losses.py                          ← Huber, WeightedMSE, MultiTask losses
    │
    ├── experiments/                           ← All experiment and figure generation scripts
    │   ├── __init__.py
    │   ├── ablation_study.py                  ← 7-configuration feature ablation
    │   ├── a2_recomputation.py                ← A2 statistics + supplementary figures
    │   ├── architecture_variants.py           ← A7 architecture variant sweep
    │   ├── baseline_comparison.py             ← 9 traditional ML baselines
    │   ├── comprehensive_ablation.py          ← Multi-seed ablation with training curves
    │   ├── cross_farm_validation.py           ← Leave-one-farm-out generalization
    │   ├── dl_architecture_comparison.py      ← TCN vs ConvLSTM vs Transformer vs BiLSTM
    │   ├── generate_figures.py                ← 8 main publication figures + SHAP + stats
    │   ├── generate_callibrationnet_arch.py   ← CalibrationNet 3D architecture diagram
    │   ├── generate_multitasknet_3D.py        ← MultiTaskNet 3D architecture diagram
    │   ├── generate_system_overview_3D.py     ← System overview pipeline diagram
    │   ├── mtl_analysis.py                    ← Multi-task representation/gradient analysis
    │   ├── rule_vs_dl_comparison.py           ← Rule-based vs. DL forecasting
    │   ├── scheduling_simulation.py           ← Retrospective irrigation scheduling
    │   ├── sensitivity_analysis.py            ← Multi-task λ sensitivity sweep
    │   └── statistical_analysis.py            ← Statistical significance tests
    │
    ├── hardware/
    │   └── data_logging/
    │       └── data_logging.ino               ← Arduino firmware for sensor data collection
    │
    └── results/
        ├── experiments/                       ← JSON experiment outputs
        ├── figures/                           ← Publication-quality PNG figures (300 DPI)
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

## Experiment Scripts

### Complete Script Inventory

All experiment scripts are in `deep_learning_model/experiments/`. Run from the `deep_learning_model/` directory.

| Script | Purpose | Output Type | Manuscript Reference |
|--------|---------|-------------|---------------------|
| `ablation_study.py` | 7-configuration feature ablation (single seed) | JSON | Table 7 |
| `comprehensive_ablation.py` | Multi-seed ablation (5 seeds × 7 configs = 35 runs) with training curves | JSON + Figure | Tables 7–8 |
| `a2_recomputation.py` | A2-feature baselines + supplementary figures (CIs, training curves, λ sensitivity, MTL importance) | JSON + Figures | Tables 2–3, Figs S1–S4 |
| `baseline_comparison.py` | 9 traditional ML baselines for calibration | JSON | Tables 2–3 |
| `cross_farm_validation.py` | Leave-one-farm-out transfer experiments | JSON | Table 5 |
| `rule_vs_dl_comparison.py` | Rule-based vs. DL forecasting across horizons | JSON | Table 4 |
| `dl_architecture_comparison.py` | TCN vs ConvLSTM vs Transformer vs BiLSTM-Attention | JSON | Table 6 |
| `architecture_variants.py` | A7 dropout/hidden-dim/warmup sweep | JSON | Table 8 |
| `sensitivity_analysis.py` | Multi-task loss weight λ sensitivity | JSON | Table 9 |
| `mtl_analysis.py` | Gradient flow + representation (CKA) analysis | JSON | §5.5 Discussion |
| `statistical_analysis.py` | Wilcoxon, Cohen's d, bootstrap CIs, Mann-Whitney U | JSON | §4.6 |
| `scheduling_simulation.py` | Retrospective irrigation scheduling (5 strategies) | JSON + Figures | Table 10, §4.7 |
| `generate_figures.py` | 8 main publication result figures + SHAP + statistical tests | Figures | Figs 1–4, 6–9 |
| `generate_callibrationnet_arch.py` | CalibrationNet 3D architecture diagram | Figure | Architecture figure |
| `generate_multitasknet_3D.py` | MultiTaskNet 3D architecture diagram | Figure | Architecture figure |
| `generate_system_overview_3D.py` | End-to-end system pipeline diagram | Figure | System overview figure |

### Figure-Producing Scripts

The following table maps each figure file in `results/figures/` to the script that generates it.

| Figure File | Generating Script | Description |
|-------------|-------------------|-------------|
| `fig1_calibration_scatter.png` | `generate_figures.py` | Predicted vs. actual moisture scatter plot |
| `fig2_multi_horizon_forecast.png` | `generate_figures.py` | Multi-horizon forecasting comparison |
| `fig3_ablation_study.png` | `generate_figures.py` | 7-configuration ablation bar chart |
| `fig4_shap_importance.png` | `generate_figures.py` | SHAP beeswarm feature importance |
| `fig6_cross_farm.png` | `generate_figures.py` | Cross-farm generalization results |
| `fig7_baseline_comparison.png` | `generate_figures.py` | Baseline model comparison |
| `fig8_thermal_lag.png` | `generate_figures.py` | Atmospheric–soil thermal lag |
| `fig9_rule_vs_dl.png` | `generate_figures.py` | Rule-based vs. DL degradation |
| `fig_ablation_confidence_intervals.png` | `a2_recomputation.py` | Ablation R² with 95% CIs |
| `fig_ablation_training_curves.png` | `a2_recomputation.py` | Training loss/R² per ablation config |
| `fig_sensitivity_lambda.png` | `a2_recomputation.py` | λ sensitivity Pareto plot |
| `fig_mtl_feature_importance.png` | `a2_recomputation.py` | MTL vs. standalone feature importance |
| `fig_scheduling_timeline.png` | `scheduling_simulation.py` | SM trace + irrigation triggers |
| `fig_scheduling_comparison.png` | `scheduling_simulation.py` | Strategy precision/recall/F1 bars |
| `fig_calibrationnet_3d.png` | `generate_callibrationnet_arch.py` | CalibrationNet architecture diagram |
| `fig_multitasknet_3d.png` | `generate_multitasknet_3D.py` | MultiTaskNet architecture diagram |
| `fig_system_overview_3d.png` | `generate_system_overview_3D.py` | System pipeline overview |
| `prototype_setup.jpg` | *(photograph)* | Hardware deployment photo |

---

## Step-by-Step Replication Guide

Follow these steps **exactly** to reproduce every result and figure in the paper.

### Step 0: Prerequisites

You need the following installed on your computer:

| Software | Version | Purpose |
|----------|---------|---------||
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
# 4a. Ablation study — tests 7 feature configurations (Table 7, ~30-60 min)
python experiments/ablation_study.py

# 4b. Comprehensive ablation — multi-seed (5 seeds × 7 configs, ~3-5 hours)
python experiments/comprehensive_ablation.py

# 4c. A2 recomputation — baselines on optimal features + supplementary figures (~15-30 min)
python experiments/a2_recomputation.py

# 4d. Baseline comparison — trains 9 ML models (~5-10 min)
python experiments/baseline_comparison.py

# 4e. Cross-farm validation — bidirectional farm transfer (~10-20 min)
python experiments/cross_farm_validation.py

# 4f. Rule-based vs. DL forecasting comparison (~5 min)
python experiments/rule_vs_dl_comparison.py

# 4g. DL architecture comparison — TCN, ConvLSTM, Transformer, BiLSTM-Att (~1-2 hours)
python experiments/dl_architecture_comparison.py

# 4h. Architecture variants — A7 dropout/hidden-dim sweep (~30-45 min)
python experiments/architecture_variants.py

# 4i. Sensitivity analysis — multi-task λ sweep (~30-45 min)
python experiments/sensitivity_analysis.py

# 4j. MTL representation analysis — gradient flow and CKA (~10-15 min)
python experiments/mtl_analysis.py

# 4k. Statistical analysis — significance tests, effect sizes, CIs (~5 min)
python experiments/statistical_analysis.py

# 4l. Scheduling simulation — retrospective irrigation strategies (~2 min)
python experiments/scheduling_simulation.py
```

Each experiment saves its results as a JSON file in `results/experiments/`.

### Step 5: Generate All Figures

All figure-generating scripts should be run from `deep_learning_model/`:

```bash
# 5a. Main publication figures (Figs 1–4, 6–9) + SHAP + statistical tests
python experiments/generate_figures.py

# 5b. Supplementary figures (CIs, training curves, λ sensitivity, MTL importance)
python experiments/a2_recomputation.py

# 5c. Scheduling simulation figures (timeline + strategy comparison)
python experiments/scheduling_simulation.py

# 5d. Architecture diagrams
python experiments/generate_system_overview_3D.py
python experiments/generate_callibrationnet_arch.py
python experiments/generate_multitasknet_3D.py
```

All figures are saved at 300 DPI to `results/figures/`.

### Step 6: Generate the Graphical Abstract

```bash
cd graphical_abstract
pip install matplotlib numpy pillow
python generate_graphical_abstract.py
```

This produces `results/graphical_abstract.png`, `results/graphical_abstract.tiff`, and `results/graphical_abstract.pdf` at 300 DPI (≥1328×531 px, 2.5:1 aspect ratio) compliant with Elsevier specifications.

### Step 7: Verify Your Results

Compare your generated files against the values in the tables below. Due to stochastic training, single-seed results may vary slightly. The multi-seed ablation study (Step 4b) averages over 5 random seeds to reduce variance.

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

All 18 figures used in the manuscript are listed below, mapped to their generating scripts:

| Figure | File | Script | Description |
|--------|------|--------|-------------|
| Fig. 1 | `fig1_calibration_scatter.png` | `generate_figures.py` | Predicted vs. actual moisture scatter |
| Fig. 2 | `fig2_multi_horizon_forecast.png` | `generate_figures.py` | Multi-horizon forecasting performance |
| Fig. 3 | `fig3_ablation_study.png` | `generate_figures.py` | 7-configuration ablation results |
| Fig. 4 | `fig4_shap_importance.png` | `generate_figures.py` | SHAP feature importance analysis |
| Fig. 6 | `fig6_cross_farm.png` | `generate_figures.py` | Cross-farm generalization results |
| Fig. 7 | `fig7_baseline_comparison.png` | `generate_figures.py` | Baseline model comparison |
| Fig. 8 | `fig8_thermal_lag.png` | `generate_figures.py` | Atmospheric–soil thermal lag |
| Fig. 9 | `fig9_rule_vs_dl.png` | `generate_figures.py` | Rule-based vs. DL comparison |
| Supp. | `fig_ablation_confidence_intervals.png` | `a2_recomputation.py` | Ablation R² with 95% CIs |
| Supp. | `fig_ablation_training_curves.png` | `a2_recomputation.py` | Training convergence per config |
| Supp. | `fig_sensitivity_lambda.png` | `a2_recomputation.py` | λ sensitivity analysis |
| Supp. | `fig_mtl_feature_importance.png` | `a2_recomputation.py` | MTL feature importance ratios |
| Supp. | `fig_scheduling_timeline.png` | `scheduling_simulation.py` | Irrigation scheduling timeline |
| Supp. | `fig_scheduling_comparison.png` | `scheduling_simulation.py` | Strategy comparison bars |
| Arch. | `fig_calibrationnet_3d.png` | `generate_callibrationnet_arch.py` | CalibrationNet architecture |
| Arch. | `fig_multitasknet_3d.png` | `generate_multitasknet_3D.py` | MultiTaskNet architecture |
| Arch. | `fig_system_overview_3d.png` | `generate_system_overview_3D.py` | System pipeline overview |
| Photo | `prototype_setup.jpg` | *(photograph)* | Hardware deployment |

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
| Results differ from paper | Single-seed runs vary. Run the ablation study (Step 4b) for multi-seed averages |
| CUDA out of memory | Reduce batch size: `--batch-size 32` or `--batch-size 16` |

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{saha2026ecoirrigate,
  title     = {A Multi-Task Deep Learning Framework for Sensor Calibration
               and Multi-Horizon Soil Moisture Prediction in {IoT}
               Irrigation Systems},
  author    = {Saha, Soham and Kar, Subhojit and Ghosh, Arijeet and Bhattacharjee, Sohan 
               and Sanyal, Somes and Das, Avik Kumar},
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
- Supported by the IEM UEM Trust Research Grant (Project ID: IEMT(N)2024/A/08 G44).
