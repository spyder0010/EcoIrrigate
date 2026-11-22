# Fixed unseen-station transport check

**Written 14 November 2026 before downloading or opening any of the selected stations' 2026 measurement files.** This is a separate transport check of the NOAA pooled residual Ridge recipe. It is not an EcoIrrigate sensor validation, an irrigation intervention, or a new neural model. The seven development stations' 2026 results are already known, so this extension is prospective only with respect to the 36 station files named below.

## Station rule

Use the filenames in the official NOAA USCRN 2026 `hourly02` directory index at https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2026/. Exclude Alaska, Hawaii and Ontario and the states of the seven existing stations (AZ, CA, FL, IA, IL, SC, TX). For every remaining state in that index, take its alphabetically first station filename. This deterministic rule produces the following 36 stations, fixed from the directory listing before measurement download:

`AL_Brewton_3_NNE`, `AR_Batesville_8_WNW`, `CO_Boulder_14_W`, `GA_Brunswick_23_S`, `ID_Arco_17_SW`, `IN_Bedford_5_WNW`, `KS_Manhattan_6_SSW`, `KY_Bowling_Green_21_NNE`, `LA_Lafayette_13_SE`, `MA_Blue_Hill_0_W`, `ME_Limestone_4_NNW`, `MI_Chatham_1_SE`, `MN_Goodridge_12_NNW`, `MO_Chillicothe_22_ENE`, `MS_Holly_Springs_4_N`, `MT_Dillon_18_WSW`, `NC_Asheville_13_S`, `ND_Jamestown_38_WSW`, `NE_Harrison_20_SSE`, `NH_Durham_2_N`, `NM_Las_Cruces_20_N`, `NV_Baker_5_W`, `NY_Ithaca_13_E`, `OH_Wooster_3_SSE`, `OK_Goodwell_2_E`, `OR_Coos_Bay_8_SW`, `PA_Avondale_2_N`, `RI_Kingston_1_NW`, `SD_Aberdeen_35_WNW`, `TN_Crossville_7_NW`, `UT_Brigham_City_28_WNW`, `VA_Cape_Charles_5_ENE`, `WA_Darrington_21_NNE`, `WI_Necedah_5_WNW`, `WV_Elkins_21_ENE`, `WY_Lander_11_SSE`.

Store the fetched directory listing, its retrieval time and hash along with every station file URL, retrieval time, row count and SHA-256 hash. If a listed file cannot be downloaded, retain that failure in the report rather than substitute another station.

## Frozen model and forecast origins

For each horizon of 1, 6, 12 and 24 hours, fit the pooled residual Ridge from `noaa_site_blend.py` on the original seven stations. Choose alpha from `{0.1, 1, 10, 100, 1000}` on 2024 after fitting on 2023, then refit on eligible 2023–2025 records. Do **not** train or select any parameter using a new station. Apply the fixed AZ, CA and FL station indicators; every new station has zeros in these columns. Do not use site blend weights, because no station-specific validation was performed for new sites.

For a new 2026 station, require 25 consecutive observed hourly 5-cm VWC values through origin `t` and a measured target at exactly `t+h` UTC hours. Use only weather observed through `t`; impute missing predictors from the original seven-station fitting partition. Express VWC in percentage points. The predictions are current VWC plus the Ridge-predicted change. Persistence copies current VWC. Both use exactly the same origins. Do not fill missing VWC.

## Locked reporting

Report usable origin counts, missingness and RMSE/MAE at every selected station and horizon. A station contributes to pooled metrics and site-level summaries for a horizon only if it has at least 1,000 eligible 2026 origins at that horizon; retain all smaller or zero-coverage stations in the availability table. Report the eligible station count, pooled RMSE and MAE, median station RMSE difference, and the number of stations where Ridge beats persistence. Present all four horizons, not only the best. The 2026 archive is incomplete and the directory sample is a deterministic convenience cohort, so intervals or site summaries must not be described as population-wide generalization. Do not alter the model after seeing these outcomes.
