# EcoIrrigate

Reproducibility materials for **Robustness of Soil Moisture Forecast Rankings under Nonlinear Sensor Calibration**. The analysis compares each learned forecast with persistence on the same forecast origins, then checks whether the resulting mean squared-error ranking survives a range of nonlinear transformations applied equally to forecasts and observations.

The two-site EcoIrrigate CSV has 21,312 processed observations covering 5 January through 25 April 2025. Its VWC column has no documented, independent per-observation calibration reference. A separate NOAA USCRN experiment uses seven stations for model development and 35 unseen stations for evaluation. A third check uses published calibration curves and two probe series from the GN1 archive.

## Setup and verification

Use Python 3.11 and install the pinned dependencies from the repository root.

```text
python -m pip install -r requirements.txt
python -m unittest discover -s revision -p test_reproducibility.py -v
python revision/audit_neural_ledger_against_source.py
python revision/verify_noaa_ledger_against_raw.py
python revision/verify_published_calibration_check.py
```

The tests cover the numerical bounds, off-grid map containment, a constructed rank reversal, resampling deviations and exact forecast timing near gaps and year changes. The source checks reconcile forecast ledgers with the input measurements and recompute the GN1 scores.

## Reproduce the ranking results

```text
python revision/calibration_ranking_envelope.py
python revision/continuous_scale_bounds.py
python revision/scale_sampling_sensitivity.py
python revision/published_calibration_check.py
```

The first and third commands may take several minutes. The scale and station resampling procedures are described in [the continuous bounds protocol](revision/CONTINUOUS_SCALE_BOUNDS_PROTOCOL.md) and [the scale and sampling protocol](revision/SCALE_AND_SAMPLING_PROTOCOL.md). The published-curve check and its source data are documented in [the GN1 protocol](revision/PUBLISHED_CALIBRATION_CHECK_PROTOCOL.md).

## Refit and compare forecasts

```text
python revision/audit_dataset_and_baselines.py
python revision/audit_target_mapping.py
python revision/evaluate_residual_ridge.py
python revision/noaa_site_blend.py --year 2026 --expanded
python revision/noaa_zero_shot.py --evaluate-only
```

The NOAA refits use the frozen data in [external_noaa](revision/external_noaa). The [unseen-station protocol](revision/NOAA_ZERO_SHOT_PROTOCOL.md) specifies station eligibility, features and score aggregation. Use the stored raw files to recreate reported scores. The live archive may have changed since the snapshot.

[The neural benchmark](revision/forecast_architecture_benchmark.py) supports new fits and seed-specific output directories. The saved ledgers under [architecture_benchmark](revision/architecture_benchmark) preserve the comparison inputs. The [archived checkpoint evaluation script](revision/evaluate_saved_multitask.py) requires the historical checkpoint, held by the authors. Its scores and predictions are in [baseline_audit](revision/baseline_audit).

## Data sources

- [EcoIrrigate dataset, Mendeley Data version 1](https://doi.org/10.17632/8x4v7yy3ds.1) and [CSV data dictionary](New_Dataset/DATA_DICTIONARY.md).
- [NOAA U.S. Climate Reference Network hourly product](https://www.ncei.noaa.gov/access/crn/products.html). The station manifests record source URLs, dates and hashes.
- [GN1 archive by Strypsteen, Persson, Rakonjac, and Miroshnychenko](https://doi.org/10.5281/zenodo.16911016). The original workbook is included under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
