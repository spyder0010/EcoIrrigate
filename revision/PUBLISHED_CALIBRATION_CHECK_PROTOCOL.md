# External check with published sensor calibration curves

The EcoIrrigate analysis uses hypothetical monotone scales because the original calibration cannot be traced to physical reference readings. This independent data example checks the same evaluation method on two low-cost probes from the Swedish GN1 column experiment, using the universal and soil-specific laboratory calibration functions published by [Strypsteen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13259304/) and their [archived raw probe readings](https://doi.org/10.5281/zenodo.16911016). The calibration paper evaluated agreement with reference moisture. This check asks whether laboratory calibration choice changes the relative accuracy of fixed, short-horizon forecasts of the probe signal.

## Data eligibility

GN1 probes 2 and 4 are documented at 10 cm in the paper. Use both and report them individually. Read the workbook's explicit clock fields, sort them and reject duplicate stamps. Record the mismatch between the workbook's dates and the published experiment dates. Retain raw hourly readings between 185 and 467.5 inclusive. The lower limit follows the reported wet-end detection limit. The upper limit is the zero-water-content point of the universal curve. Preserve missing hours and require continuous histories. The field WET values are sparse secondary sensor measurements and available from only one location. They are outside the forecast target and are counted for provenance. GN2 has converted VWC without round-trippable raw values and is excluded. The TH workbook has four numbered channels but its depth-to-channel assignment is unverified and its texture label differs from the laboratory site soil name. Restrict the experiment to GN1.

## Forecast comparison fixed before fitting

Score horizons 1, 6, 12 and 24 hours. Give each forecast the UCL-mapped reading at its origin and the past changes over 1, 3, 6 and 12 exact hours. Require the complete 13-hour history, a finite target and origin and target in the same chronological partition. Set the test cutoff to 80% of the complete workbook timestamp grid, before any probe-specific filtering. Fit separate standardized ridge regressions with α = 1 to predict changes from the current value at each probe and horizon. The baseline copies the current observation. Clip predictions to the UCL range corresponding to the prespecified ADC limits. No field reference reading is used for fitting, and no forecast is refitted when the evaluation scale changes.

## Curve stress test

Use Table 3 coefficients (a, b, c) for the universal laboratory calibration (170.7, 296.8, 0.100) and the matched sandy clay loam curve (180.20, 364.30, 0.062). Both use VWC = −c log[(ADC − a)/b]. Invert UCL and compose with the published soil-specific curve. Affinely align the resulting curve to the UCL scale endpoints for the scored range of each probe across horizons. This preserves the within-probe ranking and isolates nonlinear spacing from offsets and unit changes. Report all raw and normalized scores separately.

Use the eight-bin envelopes at δ = 0.10, 0.25 and 0.50. The mean value theorem ensures all interpolation slopes are within the normalized smooth derivative range. For a bin of width h, the normalized smooth curve differs from its chord by at most h² times the maximum absolute curvature divided by 8. Apply this deterministic interpolation error to the MSE difference using the fact that both the interpolant and the smooth map have Lipschitz constant at most the maximum slope. Independently score the exact smooth and interpolated curves and confirm containment whenever both maps qualify. Do not interpret a bound containing zero as evidence of a reversal without actual feasible forecasts with opposing score signs.

Let z(x) = a₀ + b₀ exp(−x/c₀) be the analogue reading at universal-scale moisture x. For the alternative map g(x) = −c log[(z(x) − a)/b], the derivative is (c/c₀)(z − a₀)/(z − a). The second derivative is (c/c₀²)(a − a₀)(z − a₀)/(z − a)². Here a > a₀, so both are largest at the upper endpoint of the scored domain. Endpoint alignment multiplies both derivative bounds by the same positive gain. If the interpolation error is at most ε and the normalized slope is at most L, the change in paired MSE difference is bounded by 4εL times the mean of |ridge − target| + |persistence − target| on the universal scale. Each forecast–target error changes by at most 2ε, and the sum of its two absolute errors is bounded by twice L times its universal-scale distance. This reasoning applies to smooth curves with stated derivative and curvature limits.

No finding in this trial calibrates the EcoIrrigate probe or establishes the forecast error against independently sampled VWC. The use of an archived raw series and instrument-specific published calibration functions tests transfer of the evaluation method. The outcomes of this experiment do not select the methods, split, horizons or thresholds.

## Results and checks

The raw GN1 workbook contains 962 timestamped hours from 10 October to 19 November 2024. Table 2 of the paper lists 10 July to 19 October 2024. We retain the explicit workbook timestamps without guessing which dates match the campaign. Each probe has 874 numeric readings and 88 spreadsheet division-error values, which remain missing. Both equations transform the archived hourly analogue averages. Individual readings before hourly averaging are unavailable in this workbook. The 68 WET readings are excluded from model fitting and scoring. The cutoff is 11 November 2024 at 08:00 in the workbook's clock. Both probes have 192, 187, 181 and 169 eligible test origins at 1, 6, 12 and 24 hours.

Ridge-minus-persistence MSE differences in squared percentage points are shown below. The second scale applies the published sandy clay loam formula directly, before endpoint alignment. A negative value favors Ridge.

| Probe | Horizon | Universal lab curve | Soil-specific lab curve |
|---|---:|---:|---:|
| 2 | 1 h | −0.00011596 | −0.00010656 |
| 2 | 6 h | +0.00493428 | +0.00343214 |
| 2 | 12 h | −0.00212364 | −0.00093734 |
| 2 | 24 h | +0.00597987 | +0.01047413 |
| 4 | 1 h | −0.00000497 | +0.00002416 |
| 4 | 6 h | −0.00762728 | −0.00635688 |
| 4 | 12 h | −0.02252980 | −0.01728038 |
| 4 | 24 h | −0.07241140 | −0.05091540 |

All normalized curve derivatives were within 0.10 of one across the scored ranges. The relaxed bounds with the interpolation slack contained all eight exact normalized scores. For the near tie at probe 4 and one hour the smooth-curve range is about [−0.001870, +0.001857]. At 6, 12 and 24 hours on probe 4, even the widened 0.10 intervals exclude zero. This example shows a real published scale pair can flip a close comparison while other differences remain stable within the specified scale class. The one-hour RMSE difference is under 0.00005 percentage points on both curves. It is a numerical reversal with negligible practical size. The repository contains only a short archive for this check and does not establish a physical calibration for the EcoIrrigate sensor.

Run `python revision/published_calibration_check.py` and `python revision/verify_published_calibration_check.py`. The second script reads the workbook XML independently of the analysis, reconstructs all 1,458 exact forecast pairs, refits Ridge with direct linear algebra, and verifies the exact and interpolated scores and their containment. The unmodified source workbook is attributed to Strypsteen, Persson, Rakonjac, and Miroshnychenko (CC BY 4.0). Its download and checksums are in `external_calibration/strypsteen/SOURCE_MANIFEST.json`.
