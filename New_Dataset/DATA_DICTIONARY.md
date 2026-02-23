# EcoIrrigate Dataset — Data Dictionary

**Dataset title:** EcoIrrigate: Multi-Sensor Soil Moisture Monitoring Dataset from Kolkata, India (Jan–Apr 2025)

**Associated manuscript:** From Rule-Based to Deep Learning: Multi-Task Sensor Calibration and Soil Moisture Forecasting for Precision Irrigation

---

## File: `kolkata_unified_dataset.csv`

| # | Column | Type | Unit | Description | Range |
|---|--------|------|------|-------------|-------|
| 1 | `Timestamp` | datetime | ISO 8601 | Measurement timestamp (15-min intervals) | 2025-01-05 00:00 – 2025-04-25 23:45 |
| 2 | `Farm_ID` | string | — | Farm identifier | Kolkata_Farm_1, Kolkata_Farm_2 |
| 3 | `Volumetric_Moisture_Pct` | float | % VWC | Calibrated volumetric water content | 8.91 – 19.63 |
| 4 | `Sensor_Board_Temperature_C` | float | °C | On-board temperature of capacitive sensor | varies |
| 5 | `Raw_Capacitive_ADC` | int | counts | Raw 10-bit ADC reading from capacitive sensor | ~600 – 780 |
| 6 | `Sensor_Voltage_V` | float | V | Supply voltage to sensor | ~2.0 – 3.3 |
| 7 | `Atm_Pressure_inHg` | float | inHg | Atmospheric pressure from METAR data | ~29.8 – 30.1 |
| 8 | `Atm_Temperature_C` | float | °C | Atmospheric temperature from METAR data | ~14 – 37 |
| 9 | `Soil_Temperature_C` | float | °C | Soil temperature at sensor depth | ~17 – 34 |
| 10 | `SM_Level_15cm` | float | cm | Soil moisture level at 15 cm depth | varies |
| 11 | `SM_Volume_15cm` | float | cm³ | Soil moisture volume at 15 cm depth | varies |
| 12 | `SM_Aggregate_Pct` | float | % | Aggregate soil moisture percentage | varies |
| 13 | `SM_Volume_Pct` | float | % | Soil moisture volume percentage | varies |
| 14 | `Hour` | int | — | Hour of day (0–23) | 0 – 23 |
| 15 | `Day` | int | — | Day of month | 1 – 30 |
| 16 | `Month` | int | — | Month (1–12) | 1 – 4 |
| 17 | `DayOfWeek` | int | — | Day of week (0=Monday) | 0 – 6 |
| 18 | `DayOfYear` | int | — | Day of year | 5 – 115 |
| 19 | `Hour_sin` | float | — | Cyclical hour encoding (sine) | -1.0 – 1.0 |
| 20 | `Hour_cos` | float | — | Cyclical hour encoding (cosine) | -1.0 – 1.0 |
| 21 | `Day_sin` | float | — | Cyclical day-of-year encoding (sine) | -1.0 – 1.0 |
| 22 | `Day_cos` | float | — | Cyclical day-of-year encoding (cosine) | -1.0 – 1.0 |

**Total records:** 21,312 (10,656 per farm)
**Temporal resolution:** 15-minute intervals
**Study period:** 110 days (5 Jan – 25 Apr 2025)
**Missing data:** Gap-filled using linear interpolation (128 gaps, < 1.1%)
**Location:** Kolkata, West Bengal, India (22.57°N, 88.36°E)

---

## File: `daily_pressure_extracted.csv`

Barometric pressure extracted from METAR aviation weather reports (Kolkata VECC station).

---

## File: `daily_temperature_extracted.csv`

Temperature data extracted from METAR aviation weather reports (Kolkata VECC station).

---

## File: `QUALITY_REPORT.json`

Automated data quality assessment including completeness, range checks, and gap statistics.

---

## Hardware

- **Sensor:** HW-390 capacitive soil moisture sensor (v1.2)
- **Microcontroller:** Arduino Mega 2560 + ESP8266 WiFi module
- **Power:** Solar-powered (6V panel + 18650 Li-ion battery)
- **Sampling:** 15-minute intervals via ThingSpeak IoT platform

---

## License

CC BY 4.0 International
