# Supplied ADC and moisture-target relationship

**Question.** Does the processed EcoIrrigate CSV show that its `Volumetric_Moisture_Pct` target is a direct, exact lookup of the integer `Raw_Capacitive_ADC` value? This is a diagnostic of the supplied file, not an audit of the unavailable raw acquisition or gravimetric procedure.

**Source and grain.** `EcoIrrigate Codes/New_Dataset/kolkata_unified_dataset.csv`, SHA-256 `1f12bac2a35f78b8f2c6af0fe56142f2301471d9ef2203a3983a5a60444d6319`; one processed row per farm and 15-minute timestamp, 21,312 rows. The source has no duplicate farm–timestamp key or null ADC/VWC cells. Reproduce with `python revision/audit_target_mapping.py`; detailed output is `revision/baseline_audit/target_mapping_diagnostic.json`.

| Farm | Distinct ADC levels | Levels seen at least five times | Such levels with multiple VWC values | Median within-level VWC range | ADC–VWC correlation |
|---|---:|---:|---:|---:|---:|
| Farm 1 | 326 | 218 | 218 | 1.36 percentage points | −0.959 |
| Farm 2 | 352 | 264 | 264 | 1.07 percentage points | −0.968 |
| Both | 359 | 302 | 302 | 1.17 percentage points | −0.963 |

An identical *recorded integer ADC* value can accompany multiple supplied VWC values, including within each farm. The target therefore is **not an exact single-valued function of that ADC field alone**. A chronological, ADC-only Ridge fit on each farm's first 70% and scored on its final 15% yields pooled RMSE 1.126 VWC percentage points and R² 0.713. This fit is descriptive and is not an independent calibration error; the model predicts the supplied column, whose physical reference provenance remains unknown.

**Analytical risk (high for calibration claims, high confidence in the file-level observation).** A large ADC–VWC correlation cannot establish gravimetric accuracy, and an auxiliary model that reproduces supplied VWC cannot be described as independently calibrated. The within-ADC variation also means the specific hypothesis of a direct integer-ADC-only lookup is unsupported by this CSV. Smoothing, additional covariates, a higher-resolution unrecorded probe value, a reference procedure, and other transformations remain possible; the file cannot distinguish them. The manuscript should state the observed relation and retain the provenance limit without asserting a particular unseen transformation.
