# Frozen external forecast check (before 2026 outcome download)

Created after inspecting 2025 NOAA development results and before downloading
2026 NOAA records. This is a computational replication of the forecasting
procedure, not evidence for EcoIrrigate calibration, irrigation savings, or
transfer of model weights between countries and sensor systems.

## Source and station rule

Use the public NOAA USCRN `hourly02` files at
https://www.ncei.noaa.gov/access/crn/products.html, with the format and missing
value definitions in
https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/readme.txt.
The four initially selected stations were fixed by distinct geography before
model-score inspection: Alabama Brewton, Arizona Tucson, California Merced and
Florida Sebring. Alabama has no 5 cm VWC values in 2023–24, so it cannot enter
the 2023 training/2024 selection protocol. Retain the other three, including
the Florida 2024 outage, and report usable origin counts. Do not replace a
station on the basis of forecasting score.

## Information boundary

At hour `t`, use the current observed 5 cm VWC and trailing 24 hours of
measured VWC, plus weather observations through `t`. Forecast the measured
VWC at `t+h` for `h = 1, 6, 12, 24` hours. Require 25 consecutive current and
past VWC readings and an observed target; never fill missing VWC labels.
Impute missing weather features with medians fitted on the training partition.
Express errors in VWC percentage points (`100 × m³/m³`).

## Frozen learner and selection

For each horizon, fit a pooled Ridge model for the future change from current
VWC, using the fixed features in `noaa_site_blend.py` and
`noaa_residual_baseline.py`. Select one Ridge alpha from
`{0.1, 1, 10, 100, 1000}` by pooled 2024 RMSE after fitting on 2023. Using
that 2023-fitted Ridge, select one blend weight per station and horizon from
`{0, 0.25, 0.5, 0.75, 1}` by that station's 2024 RMSE. The final prediction is
`current VWC + weight × predicted change`.

The 2025 evaluation in `site_blend_2025_development.json` is exploratory; its
outcomes motivated the site blend, including the observation that pooled
Ridge lost to persistence at some sites. For the untouched 2026 check, retain
the 2024-selected alpha and weights, refit Ridge on 2023–25 observations, and
score available 2026 origins once. Compare on identical origins with
persistence and unblended Ridge. Report pooled and each-site RMSE and MAE,
paired uncertainty intervals with day-level blocks, missingness, and failures
without revising the model after seeing 2026 outcomes.

The data are updated by NOAA, so the raw files, download time, source URL and
SHA-256 hashes must be retained in `external_noaa/manifest.json`. The first
2025 development pull and 2026 held-out pull occur at different times; the
manifest reflects the latest downloaded bytes for each file.

## Expanded station replication, fixed before downloading new station outcomes

The first 2026 check has been inspected and shows a pooled gain largely driven
by Arizona. To test whether this reflects a broader station pattern, add four
geographically distinct stations before inspecting their data or scores:
`IA_Des_Moines_17_E`, `IL_Champaign_9_SW`, `SC_Blackville_3_W`, and
`TX_Austin_33_NW`. These names were selected from NOAA's 2024 directory by
state and station identity, not forecast performance. A station enters the
expanded model only if both 2023 and 2024 have at least 1,000 nonmissing 5-cm
VWC readings. Exclude by this coverage rule only; report every exclusion and
2026 coverage. Run the same fixed feature set, alpha grid, blend grid,
selection years, origin rules and refitting procedure as above. Save the
expanded result separately from the original three-station check. The new
stations' 2026 outcomes are unseen at the time this extension is specified;
the original stations' 2026 outcomes are already known and should be labeled
accordingly in the interpretation.

## Timestamp implementation correction, 14 November 2026

An audit after the initial seven-station analysis found one absent UTC hour
in each of the Iowa and Texas 2026 source files, at 03:00 UTC on 8 March.
The original code shifted the target by row position. It therefore included
1, 6, 12 and 24 incorrectly timed origins per affected station at the
corresponding horizons. This violated the protocol's explicit `t+h` target
definition. The corrected code requires
`target_time - origin_time == h hours` before fitting or scoring. It does not
change the feature set, alpha grid, blend weights, station eligibility rule
or year split. The first results and ledgers are retained under
`external_noaa/pre_timestamp_correction/` for audit; all reported manuscript
numbers use the corrected exact-hour evaluation.
