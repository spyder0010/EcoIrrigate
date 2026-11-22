# Post hoc seven-station NOAA station-mix sensitivity

**Computed 15 November 2026 from the saved 2026 prediction ledger.** This check was added after the seven-station results were examined. It addresses whether the pooled blend-versus-persistence score is highly dependent on the selected station mix. It is distinct from the existing 24- and 72-hour calendar-block intervals, which resample time *within* stations. The seven sites were selected in two sequential cohorts, and their available 2026 coverage is uneven; neither form of resampling makes them a random USCRN sample or predicts another year's results.

`noaa_seven_station_sensitivity.py` aggregates each station's paired squared and absolute forecast errors directly from `external_noaa/site_blend_2026_expanded_heldout_ledger.csv`. It reproduces the pooled JSON scores, resamples seven whole stations with replacement 10,000 times using NumPy seed 20261115, and calculates all seven leave-one-station-out pooled differences. The output is `external_noaa/seven_station_mix_sensitivity.json`. All differences below are blend minus persistence in VWC percentage points; negative favors the blend.

| Horizon | Pooled RMSE difference | Station-mix RMSE 2.5–97.5% range | Station RMSE wins | Leave-one-out RMSE range | Pooled MAE difference | Station-mix MAE 2.5–97.5% range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 h | −0.02548 | [−0.04362, −0.00273] | 3/7 | [−0.03213, −0.01271] | +0.00458 | [−0.01045, +0.01779] |
| 6 h | −0.07409 | [−0.12500, −0.02372] | 5/7 | [−0.08518, −0.05176] | −0.04499 | [−0.12562, +0.02918] |
| 12 h | −0.11239 | [−0.17635, −0.04125] | 6/7 | [−0.13070, −0.08757] | −0.06706 | [−0.19345, +0.05804] |
| 24 h | −0.11252 | [−0.16693, −0.05094] | 5/7 | [−0.12454, −0.07430] | +0.04927 | [−0.07095, +0.14285] |

The observed pooled RMSE advantage retains its sign in these resamples and after each single-station omission; the corresponding MAE ranges include zero at all horizons. This strengthens the description of a *selected-cohort RMSE result*, while leaving the uneven site effects, sequential selection, incomplete 2026 records, and absence of an EcoIrrigate model transfer intact. With only seven selected stations, the percentile ranges are descriptive sensitivity summaries rather than confidence intervals for the USCRN network or a new year. The 35-new-station fixed-model check in `NOAA_ZERO_SHOT_PROTOCOL.md` remains the more direct transport test; its small six-hour pooled RMSE gain is more sensitive to station mix and comes with worse MAE.
