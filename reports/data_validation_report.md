# Milestone 3 Data Quality Validation Report: German Power Market (2023–2024)

**Project**: Portfolio Project 3 — Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Study Window**: 2023-01-01 00:00:00 to 2024-12-31 23:00:00 Europe/Berlin (17,544 delivery hours)  
**Evaluated Datasets**:
1. `day_ahead_price_2023_2024.parquet` (Filter 4169, Region DE-LU)
2. `load_forecast_2023_2024.parquet` (Filter 411, Region DE)
3. `load_actual_2023_2024.parquet` (Filter 410, Region DE)

---

## A. Structural Integrity

Each series was independently ingested across 106 weekly raw SMARD JSON files and processed into standard Parquet format.

| Dataset | Row Count | Columns | Timestamp Dtypes | Value Column & Dtype | Null Count | Duplicate Count | Missing Timestamps | Non-Numeric Count | Monotonic UTC |
|---|---|---|---|---|---|---|---|---|---|
| **Day-Ahead Price** | 17,544 | `['timestamp_utc', 'timestamp_local', 'day_ahead_price_eur_mwh']` | `datetime64[ns, UTC]`, `datetime64[ns, Europe/Berlin]` | `day_ahead_price_eur_mwh` (`float64`) | 0 | 0 | 0 | 0 | True |
| **Load Forecast** | 17,544 | `['timestamp_utc', 'timestamp_local', 'load_forecast_mw']` | `datetime64[ns, UTC]`, `datetime64[ns, Europe/Berlin]` | `load_forecast_mw` (`float64`) | 0 | 0 | 0 | 0 | True |
| **Load Actual** | 17,544 | `['timestamp_utc', 'timestamp_local', 'load_actual_mw']` | `datetime64[ns, UTC]`, `datetime64[ns, Europe/Berlin]` | `load_actual_mw` (`float64`) | 0 | 0 | 0 | 0 | True |

### Exact Temporal Range
- **First Delivery Hour (Local)**: `2023-01-01 00:00:00+01:00` (`2022-12-31 23:00:00 UTC`)
- **Last Delivery Hour (Local)**: `2024-12-31 23:00:00+01:00` (`2024-12-31 22:00:00 UTC`)
- **Timezone Integrity**: Both UTC (`datetime64[ns, UTC]`) and German market time (`datetime64[ns, Europe/Berlin]`) are explicitly preserved.

---

## B. Yearly Coverage

| Year | Calendar Type | Expected Delivery Hours | Returned Price Hours | Returned Forecast Hours | Returned Actual Hours | Coverage % |
|---|---|---|---|---|---|---|
| **2023** | Standard (365 days) | 8,760 | 8,760 | 8,760 | 8,760 | 100.00% |
| **2024** | Leap Year (366 days) | 8,784 | 8,784 | 8,784 | 8,784 | 100.00% |
| **Total** | **Full 2-Year Period** | **17,544** | **17,544** | **17,544** | **17,544** | **100.00%** |

Both 2023 (8,760 hours) and 2024 (8,784 hours due to Feb 29) match the exact legal electricity delivery requirements in Germany.

---

## C. Monthly Coverage (All 24 Months)

Every single month accounts strictly for daylight-saving transitions (March 23h, October 25h) and the 2024 leap year.

| Month | Expected Hours | Returned Hours | Missing Hours | Notes / DST Accounting |
|---|---|---|---|---|
| 2023-01 | 744 | 744 | 0 |  |
| 2023-02 | 672 | 672 | 0 |  |
| 2023-03 | 743 | 743 | 0 | Spring DST (-1h -> 23h on transition Sunday) |
| 2023-04 | 720 | 720 | 0 |  |
| 2023-05 | 744 | 744 | 0 |  |
| 2023-06 | 720 | 720 | 0 |  |
| 2023-07 | 744 | 744 | 0 |  |
| 2023-08 | 744 | 744 | 0 |  |
| 2023-09 | 720 | 720 | 0 |  |
| 2023-10 | 745 | 745 | 0 | Autumn DST (+1h -> 25h on transition Sunday) |
| 2023-11 | 720 | 720 | 0 |  |
| 2023-12 | 744 | 744 | 0 |  |
| 2024-01 | 744 | 744 | 0 |  |
| 2024-02 | 696 | 696 | 0 | Leap year (29 days = 696h) |
| 2024-03 | 743 | 743 | 0 | Spring DST (-1h -> 23h on transition Sunday) |
| 2024-04 | 720 | 720 | 0 |  |
| 2024-05 | 744 | 744 | 0 |  |
| 2024-06 | 720 | 720 | 0 |  |
| 2024-07 | 744 | 744 | 0 |  |
| 2024-08 | 744 | 744 | 0 |  |
| 2024-09 | 720 | 720 | 0 |  |
| 2024-10 | 745 | 745 | 0 | Autumn DST (+1h -> 25h on transition Sunday) |
| 2024-11 | 720 | 720 | 0 |  |
| 2024-12 | 744 | 744 | 0 |  |

- **DST Verification Result**: 
  - March 2023 and March 2024 each returned exactly **743 hours** (31 days × 24 = 744, minus 1 spring transition hour at 02:00 CET $ightarrow$ 03:00 CEST).
  - October 2023 and October 2024 each returned exactly **745 hours** (31 days × 24 = 744, plus 1 autumn transition hour repeated at 02:00 CEST and 02:00 CET).
  - February 2024 returned exactly **696 hours** (29 days × 24h = 696h), validating the leap year.
  - Total returned across all 24 months: **17,544 hours** (0 missing hours).

---

## D. Day-Ahead Price Quality

### Descriptive Statistics and Percentiles (EUR/MWh)
- **Observations Count**: 17,544
- **Mean**: 86.83 EUR/MWh
- **Median**: 88.87 EUR/MWh
- **Standard Deviation**: 50.91 EUR/MWh
- **Minimum**: -500.00 EUR/MWh
- **Maximum**: 936.28 EUR/MWh
- **1st Percentile (P1)**: -11.34 EUR/MWh
- **5th Percentile (P5)**: 0.03 EUR/MWh
- **25th Percentile (P25)**: 62.92 EUR/MWh
- **75th Percentile (P75)**: 112.34 EUR/MWh
- **95th Percentile (P95)**: 158.40 EUR/MWh
- **99th Percentile (P99)**: 214.10 EUR/MWh

### Threshold Counts and Market Extremes
- **Negative-Price Hours ($P < 0.00$)**: 758 hours (4.32% of all market hours)
- **Zero-Price Hours ($P == 0.00$)**: 86 hours
- **Price > 200 EUR/MWh**: 241 hours
- **Price > 300 EUR/MWh**: 44 hours
- **Price $\le$ -100 EUR/MWh**: 13 hours

### Investigation of Minimum Price (-500.00 EUR/MWh)
- **Total occurrences of -500.00 EUR/MWh**: Exactly **1 observation**.
- **Exact Timestamp**: 
  - **Local**: `2023-07-02 14:00:00+02:00`
  - **UTC**: `2023-07-02 12:00:00+00:00`
- **Context Analysis**: The observation occurred on **Sunday, July 2, 2023 at 14:00 CEST** (solar peak on a low-demand weekend).
- **Nearby Prices**:
| Local Timestamp | UTC Timestamp | Day-Ahead Price (EUR/MWh) |
|---|---|---|
| 2023-07-02 11:00:00+02:00 | 2023-07-02 09:00:00+00:00 | -98.11 |
| 2023-07-02 12:00:00+02:00 | 2023-07-02 10:00:00+00:00 | -167.96 |
| 2023-07-02 13:00:00+02:00 | 2023-07-02 11:00:00+00:00 | -266.92 |
| 2023-07-02 14:00:00+02:00 | 2023-07-02 12:00:00+00:00 | **-500.00** |
| 2023-07-02 15:00:00+02:00 | 2023-07-02 13:00:00+00:00 | -399.00 |
| 2023-07-02 16:00:00+02:00 | 2023-07-02 14:00:00+00:00 | -124.21 |
| 2023-07-02 17:00:00+02:00 | 2023-07-02 15:00:00+00:00 | -35.18 |

- **Market Context & Validity Determination**:
  The observed minimum was -500 EUR/MWh. Neighboring hours were also strongly negative, providing no evidence of an ingestion or isolated telemetry error.
  **Conclusion**: This observation is retained as valid historical settlement data. It must **NOT** be filtered, removed, clipped, or treated as an artifact.

---

## E. Negative Price Analysis

### Breakdown by Temporal Dimensions
- **By Year**:
  - **2023**: 301 hours (3.44% of 2023)
  - **2024**: 457 hours (5.20% of 2024)
  *(Noticeable +51.8% increase in negative price incidence from 2023 to 2024).*
- **By Day Type**:
  - **Weekdays (Mon–Fri)**: 293 negative hours (2.34% of weekday hours)
  - **Weekends (Sat–Sun)**: 465 negative hours (9.27% of weekend hours)
  *(Weekend negative-price incidence was about 4.0x the weekday rate over this dataset).*

### Distribution by Hour of Day (CET/CEST)
Negative prices concentrate heavily between 11:00 and 16:00 and secondarily around 02:00–05:00.

| Hour of Day | Negative Hours Count |
|---|---|
| 00:00 | 11 |
| 01:00 | 21 |
| 02:00 | 23 |
| 03:00 | 25 |
| 04:00 | 27 |
| 05:00 | 22 |
| 06:00 | 15 |
| 07:00 | 16 |
| 08:00 | 13 |
| 09:00 | 17 |
| 10:00 | 35 |
| 11:00 | 58 |
| 12:00 | 84 |
| 13:00 | 108 |
| 14:00 | 110 |
| 15:00 | 83 |
| 16:00 | 54 |
| 17:00 | 26 |
| 18:00 | 3 |
| 19:00 | 1 |
| 20:00 | 1 |
| 21:00 | 1 |
| 22:00 | 1 |
| 23:00 | 3 |


### Longest Consecutive Negative-Price Event
- **Start Delivery Hour**: `2023-12-24 01:00:00+01:00`
- **End Delivery Hour**: `2023-12-25 12:00:00+01:00`
- **Duration**: **36 consecutive hours** (1.5 full days)
- **Minimum Price during Event**: **-13.37 EUR/MWh**
- **Mean Price during Event**: **-4.36 EUR/MWh**
- **Historical Context**: The longest negative-price run coincided with the Christmas 2023 period (December 24–25, 2023).

---

## F. Load Forecast Quality

> **CRITICAL REGULATORY & PROVENANCE QUALIFICATION**:  
> **"PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED"**  
> Under EU Regulation 543/2013, the day-ahead total load forecast is mandated for publication by 12:00 CET on D-1. However, SMARD retrospective archives may reflect updated historical vintages. This series is analyzed here strictly as a data-quality and system-level diagnostic, NOT as a final modeling feature.

### Statistics for `load_forecast_mw` (Filter 411, DE)
- **Count**: 17,544
- **Mean**: 52,546.08 MW
- **Standard Deviation**: 9,032.43 MW
- **Minimum**: 30,544.75 MW (Occurred on 2023-05-29 05:00 CEST, Whit Monday holiday)
- **Median**: 52,284.12 MW
- **Maximum**: 73,298.25 MW (Occurred on 2024-01-10 18:00 CET, winter peak evening)
- **Values $\le$ 0**: 0 (No zero or negative load values)
- **1st Percentile**: 35,000.33 MW
- **99th Percentile**: 69,960.74 MW

---

## G. Actual Load Quality

### Statistics for `load_actual_mw` (Filter 410, DE)
- **Count**: 17,544
- **Mean**: 52,661.00 MW
- **Standard Deviation**: 9,122.74 MW
- **Minimum**: 30,902.75 MW (Occurred on 2023-05-29 06:00 CEST, Whit Monday holiday)
- **Median**: 52,628.75 MW
- **Maximum**: 75,508.25 MW (Occurred on 2024-01-10 18:00 CET, winter peak evening)
- **Values $\le$ 0**: 0 (No zero or negative load values)
- **1st Percentile**: 35,099.15 MW
- **99th Percentile**: 70,974.82 MW

---

## H. Forecast vs Actual Load Diagnostic

Joining `load_forecast_mw` and `load_actual_mw` strictly on `timestamp_utc`:
- **Matched Observations**: **17,544 / 17,544** (100.00% match)
- **Unmatched Observations**: **0**
- **Mean Absolute Error (MAE)**: **1,952.72 MW** (~3.71% mean relative error)
- **Mean Error (Bias)**: **-114.92 MW** (Slight under-forecasting on average)
- **Root Mean Squared Error (RMSE)**: **2,472.48 MW**
- **Pearson Correlation ($r$)**: **0.9630** (Extremely high fidelity)

### Top 10 Largest Absolute Forecast Differences
| Rank | Delivery Timestamp (Local) | Load Forecast (MW) | Load Actual (MW) | Forecast Error (MW) | Absolute Error (MW) | Notable Market Context |
|---|---|---|---|---|---|---|
| 1 | 2024-05-10 11:00:00+02:00 | 60,232.00 | 50,904.00 | +9,328.00 | 9,328.00 |
| 2 | 2024-01-13 17:00:00+01:00 | 56,123.00 | 65,444.50 | -9,321.50 | 9,321.50 |
| 3 | 2024-05-10 09:00:00+02:00 | 59,357.25 | 50,142.25 | +9,215.00 | 9,215.00 |
| 4 | 2024-05-10 10:00:00+02:00 | 60,189.75 | 50,980.75 | +9,209.00 | 9,209.00 |
| 5 | 2024-03-31 19:00:00+02:00 | 48,130.50 | 38,962.75 | +9,167.75 | 9,167.75 |
| 6 | 2024-01-13 16:00:00+01:00 | 54,094.25 | 63,238.75 | -9,144.50 | 9,144.50 |
| 7 | 2024-03-31 16:00:00+02:00 | 44,284.00 | 35,175.00 | +9,109.00 | 9,109.00 |
| 8 | 2024-01-13 15:00:00+01:00 | 53,690.00 | 62,737.75 | -9,047.75 | 9,047.75 |
| 9 | 2024-11-20 16:00:00+01:00 | 60,203.50 | 69,228.50 | -9,025.00 | 9,025.00 |
| 10 | 2024-01-13 14:00:00+01:00 | 54,404.50 | 63,226.00 | -8,821.50 | 8,821.50 |

*Observation on Largest Differences*: The largest discrepancies coincided with non-standard German calendar periods:
- **2024-05-10**: The large 2024-05-10 load forecast error coincided with the working day following Ascension Day (*Christi Himmelfahrt* / *Brückentag*).
- **2024-03-31**: This difference coincided with Easter Sunday and the spring Daylight Saving Time clock shift.

---

## I. Cross-Dataset Alignment

Validation of UTC and Europe/Berlin timestamp parity across all three datasets:
- **`timestamp_utc` Identical Across All 3 Series**: **True**
- **`timestamp_local` Identical Across All 3 Series**: **True**
- **Total Three-Way Merged Rows**: **17,544**
- **Unmatched Day-Ahead Price Rows**: **0**
- **Unmatched Load Forecast Rows**: **0**
- **Unmatched Load Actual Rows**: **0**
- **Total Unmatched Rows**: **0**

**Conclusion**: All three datasets are 100% time-aligned, contiguous, and share the exact same 17,544 delivery hours.

---

## J. Output Figures

The following publication-grade diagnostic figures have been generated in `reports/figures/`:

1. [Figure 1: Monthly Day-Ahead Price Distribution](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/01_monthly_price_distribution.png)  
   *Boxplot of hourly prices across all 24 months, showing median, interquartile range, outliers, and the zero-price threshold.*
2. [Figure 2: Negative-Price Hours by Month](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/02_negative_price_hours_by_month.png)  
   *Bar chart of monthly negative-price incidence illustrating the 51.8% year-over-year surge from 2023 to 2024.*
3. [Figure 3: Average Price Profile by Hour of Day](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/03_average_price_by_hour.png)  
   *Diurnal curve highlighting morning/evening peaks and the midday solar depression ("duck curve").*
4. [Figure 4: Representative Week Load Profile](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/04_load_forecast_vs_actual_week.png)  
   *High-resolution time series of forecasted vs actual load during a winter high-demand week (Jan 16–22, 2023).*
5. [Figure 5: Load Forecast Error Distribution](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/05_load_forecast_error_distribution.png)  
   *Residual error histogram showing that the forecast-error distribution is centered close to zero, with mean bias -114.9 MW.*

---

## K. Automated Validation Tests

Automated pytest validation tests have been created in `tests/test_validation.py` asserting:
1. Exact row count equals 17,544 for all three datasets.
2. Complete timestamp uniqueness (0 duplicate UTC timestamps).
3. Zero null values in timestamp and metric value columns.
4. Schema consistency (`['timestamp_utc', 'timestamp_local', '<metric_col>']` with correct types).
5. 100% three-way cross-dataset UTC timestamp equality.
6. Full hourly continuity without a single missing hour across both years and all four DST transitions.
