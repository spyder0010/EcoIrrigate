# EcoIrrigate released CSV: data dictionary and provenance limits

**File:** `kolkata_unified_dataset.csv`

**Associated archive:** [Mendeley Data version 1](https://doi.org/10.17632/8x4v7yy3ds.1)
**Purpose of this dictionary:** identify the supplied fields without treating undocumented transformations as independently verified measurements.

## Coverage and grain

The file has **21,312 rows and 22 columns**: one record per `Farm_ID` and 15-minute `Timestamp`, with 10,656 rows for each of two site IDs. The inclusive range is **5 January 2025 00:00 through 25 April 2025 23:45**, or **111 calendar days per site**. The released file contains no null cells, duplicate site–timestamp pairs or gaps in its 15-minute grid. This is a property of the processed file, not proof that acquisition was uninterrupted.

The earlier description said that 128 records were interpolated. An original unfilled timestamp log, gap mask and interpolation procedure are not present in the available repository, so that count and any gap-length distribution are **unverified**. Do not use the processed grid alone to calculate a pre-processing missingness rate.

## Columns

| Column | Supplied representation | Interpretation and qualification |
|---|---|---|
| `Timestamp` | Date and time | 15-minute record time; the file does not document the time zone |
| `Farm_ID` | Text | Two site identifiers, `Kolkata_Farm_1` and `Kolkata_Farm_2`; exact physical locations and field conditions are not independently documented here |
| `Volumetric_Moisture_Pct` | Numeric, nominal % VWC | Supplied forecast and auxiliary reconstruction target; its per-record gravimetric reference or conversion procedure is unavailable |
| `Sensor_Board_Temperature_C` | Numeric, nominal °C | Supplied board-temperature field; sensor make, placement and accuracy are unverified from the CSV |
| `Raw_Capacitive_ADC` | Integer counts | Supplied probe-related ADC field; bit depth and acquisition hardware are unverified from the archived firmware |
| `Sensor_Voltage_V` | Numeric, nominal V | Closely tracks `Raw_Capacitive_ADC`; the observed relation is approximately `3.3 × ADC / 1023` |
| `Atm_Pressure_inHg` | Numeric, nominal inHg | Supplied pressure field; original weather-source and interpolation lineage need source records for verification |
| `Atm_Temperature_C` | Numeric, nominal °C | Supplied atmospheric-temperature field; original measurement and weather-source lineage are not documented per row |
| `Soil_Temperature_C` | Numeric, nominal °C | Supplied soil-temperature field; whether it was directly measured or calculated cannot be established per row here |
| `SM_Level_15cm` | Numeric, nominal cm | Supplementary moisture-related field; derivation and physical interpretation are not independently documented |
| `SM_Volume_15cm` | Numeric, nominal cm³ | Supplementary moisture-related field; derivation is not independently documented |
| `SM_Aggregate_Pct` | Numeric, nominal % | Supplementary moisture-related field; derivation is not independently documented |
| `SM_Volume_Pct` | Numeric, nominal % | Supplementary moisture-related field; derivation is not independently documented |
| `Hour`, `Day`, `Month`, `DayOfWeek`, `DayOfYear` | Integers | Calendar fields associated with the record timestamp |
| `Hour_sin`, `Hour_cos`, `Day_sin`, `Day_cos` | Numeric | Cyclic calendar encodings supplied in the file |

Repeated integer `Raw_Capacitive_ADC` values can accompany different supplied `Volumetric_Moisture_Pct` values. Therefore, the latter is not an exact function of the recorded integer ADC field alone. This observation does not reveal the actual transformation or establish independent calibration accuracy.

## Other files and hardware descriptions

`daily_pressure_extracted.csv`, `daily_temperature_extracted.csv` and `QUALITY_REPORT.json` are associated derived files or older reports; their text does not replace missing row-level acquisition or calibration records. Older repository documents disagree about the board and cloud platform. The available firmware is dated after the 2025 period and does not establish which device produced these rows. The supplied series reports one sensing depth and contains no measured irrigation volumes, pump logs or crop outcomes.

Use the [Mendeley Data page](https://doi.org/10.17632/8x4v7yy3ds.1) for the published dataset version and governing data license. Code and model files require their own licensing and release decisions.
