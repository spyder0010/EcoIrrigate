# NOAA prediction-ledger cross-check against raw measurements

**15 November 2026.** `verify_noaa_ledger_against_raw.py` uses Python's standard-library CSV, gzip and text parsing rather than the modelling pipeline's pandas loader. It verifies the archived SHA-256 hashes, reads the `SOIL_MOISTURE_5` field from each selected `hourly02` source line, and compares **every** saved 2026 ledger row's persistence value and target with the raw observation at its origin and target UTC timestamps. It also checks each exact forecast horizon.

| Ledger | Saved forecast rows matching source origin and target VWC |
|---|---:|
| Seven-station blend | 161,142 |
| Fixed Ridge on new stations | 837,338 |
| **Total** | **998,480** |

The check verified 50 archived source files covering 43 distinct stations. The seven-station ledger can use the 2025 file's last UTC observation as an early-2026 forecast origin, so both 2025 and 2026 files for those seven stations were checked. The 36 newly selected stations use their archived 2026 files. The additional 2025 source rows are historical context, not a new evaluation period.

Run from the workspace root:

```powershell
python revision\verify_noaa_ledger_against_raw.py
```

This confirms the saved ledgers' measured target values, persistence inputs and temporal alignment against archived source bytes. It does **not** validate the Ridge feature construction, station-selection rationale, uncertainty interpretation, EcoIrrigate calibration, or the unpublished physical study. It is an internal second-parser check by Codex, not independent human review.
