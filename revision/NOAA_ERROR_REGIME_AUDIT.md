# Exploratory NOAA error-regime decomposition

Run `python revision\noaa_error_regime_audit.py` from the workspace root to regenerate `external_noaa/zero_shot/error_regime_audit.json` and `.csv` from the fixed 35-station prediction ledger. This check was designed **after** inspecting the pooled RMSE/MAE tradeoff. It conditions on the observed future change `|VWC(t+h) − VWC(t)|`, which is unavailable at forecast time. The bins are descriptive and cannot be used as an operational model gate or as independently confirmed evidence of an event detector.

For every 1-, 6-, 12- and 24-hour horizon, the script reports all five fixed magnitude bins: `<0.5`, `0.5–<1`, `1–<2`, `2–<4` and `≥4` VWC percentage points. It retains the same eligible stations and paired origins as Table 7. The CSV contains counts, within-bin RMSE/MAE for persistence and Ridge, and each bin's contribution to the pooled mean squared and absolute error differences.

At six hours, 149,289 of 209,662 origins (71.2%) changed by less than 0.5 points. Persistence MAE was 0.161 and Ridge MAE 0.318 in this bin. The 4,514 origins (2.2%) with a change of at least 4 points had persistence RMSE 8.776 and Ridge RMSE 8.533. The smallest-change bin contributed **+0.0898 points²** per pooled origin to Ridge-minus-persistence mean squared error; the largest-change bin contributed **−0.0906 points²**. Intermediate bins account for the remaining pooled difference. Thus a small reduction in pooled 6-hour RMSE can coexist with worse overall MAE.

The direction is not identical at every horizon. At one hour, Ridge's pooled RMSE is worse, and even the largest-change bin contributes positive pooled squared-error difference despite its lower within-bin MAE. At 12 and 24 hours, large-change bins partly offset error added on routine origins. All horizons are retained in the output; no threshold was selected to maximize a displayed gain.

This analysis explains how the fixed forecast was scored on the observed cohort. It does not establish that future large changes can be identified before they occur, that the changes were caused by irrigation or rainfall, or that the model improves irrigation decisions.
