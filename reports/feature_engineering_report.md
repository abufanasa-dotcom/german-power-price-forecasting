# Milestone 4 Feature Engineering & Temporal Split Report

**Project**: Portfolio Project 3 — German Day-Ahead Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Study Horizon**: 2023-01-01 00:00:00 to 2024-12-31 23:00:00 Europe/Berlin (17,544 delivery hours)  
**Initial Rows Merged**: 17,544  
**Warm-Up Period Dropped**: 168 hours (7 days required for 168h lags)  
**Final Usable Rows**: **17,376 hours**  

---

## 1. Feature Architecture Overview

Two distinct feature sets are constructed to guarantee strict leakage control while isolating the impact of official TSO load forecasts:

### A. Core Strict Pre-Auction Features (23 Predictors)
- **Targets (2)**:
  - `day_ahead_price_eur_mwh` (Continuous spot price in EUR/MWh, regression target)
  - `is_negative_price` (Binary indicator: 1 if spot price < 0.00 EUR/MWh, 0 otherwise)
- **Price History (9)**:
  - 7 daily lags: `price_lag_24h`, `price_lag_48h`, `price_lag_72h`, `price_lag_96h`, `price_lag_120h`, `price_lag_144h`, `price_lag_168h`
  - 2 derived statistics: `price_daily_lag_mean_7d`, `price_daily_lag_std_7d`
- **Actual Load History (2)**:
  - `load_actual_lag_48h` (48h lookback respecting retrospective settlement)
  - `load_actual_lag_168h` (same delivery hour last week)
- **Calendar Features in Europe/Berlin Time (11)**:
  - Diurnal: `hour_of_day`, `hour_sin`, `hour_cos`
  - Weekly: `day_of_week`, `is_weekend`
  - Annual: `month`, `month_sin`, `month_cos`
  - Market Regimes: `is_dst` (Daylight Saving Time indicator)
  - Statutory: `is_public_holiday_de`, `is_bridge_day`
- **Solar / Seasonal Physical Proxy (1)**:
  - `solar_elevation_proxy` (Deterministic astronomical solar elevation factor)

### B. Official Forecast Extension (26 Predictors)
Contains all 23 Core Strict features plus 3 load forecast predictors:
- `load_forecast_mw`: Filter 411 total load forecast published for delivery day $D$.
- `load_forecast_diff_24h`: Day-over-day change in forecasted load (`load_forecast_mw - shift(24)`).
- `load_forecast_daily_peak_ratio`: Ratio of forecasted load to the maximum forecasted load of that local delivery day.

> **CRITICAL REGULATORY & ARCHIVE QUALIFICATION**:  
> **"PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED"**  
> Under EU Regulation 543/2013, Day-Ahead total load forecasts must be published by 12:00 CET on D-1. However, retrospective SMARD archives do not guarantee immutable pre-auction point-in-time bid vintage. This group is strictly segregated from the Core Strict dataset.

---

## 2. Forbidden Features & Leakage Audit

The following information sources are **strictly prohibited** and excluded from all feature matrices:
- **SMARD Day-Ahead Wind & Solar Forecasts**: Mandated by EU Regulation 543/2013 for publication by 18:00 CET on D-1 (6 hours *after* EPEX Day-Ahead auction gate closure at 12:00 CET). Using them constitutes **UNSAFE / LOOK-AHEAD LEAKAGE**.
- **Contemporary Actual Generation & Load**: Real-time physical flows are unmetered during pre-auction bidding.
- **Current Delivery-Day Prices as Predictors**: Delivery-day clearing prices are the prediction target.
- **Price Lags Shorter Than 24 Hours**: Delivery day $D$ is cleared simultaneously in a single uniform-price auction at 12:00 on $D-1$.
- **Centered or Forward-Looking Rolling Windows**: All rolling windows and aggregations are strictly backward-looking.

---

## 3. Lag Semantics and Daylight Saving Time (DST)

- **UTC Base Timeline**: All lag operations (`shift(k)`) are calculated strictly on the sorted, uninterrupted UTC timeline.
- **Physical Meaning**: A 24-hour lag represents exactly **24 elapsed UTC hours** ($24 \times 3,600$ seconds of physical time).
- **DST Implications**:
  - Across the **Spring DST transition** (March, -1h clock shift): 24 UTC hours prior to `12:00 CEST` on transition Sunday corresponds to `13:00 CET` on Saturday.
  - Across the **Autumn DST transition** (October, +1h clock repeat): 24 UTC hours prior to `02:00 CET` corresponds to `02:00 CEST` 24 elapsed physical hours prior.
  - Using UTC prevents duplicate or omitted clock labels and guarantees that every lag interval is contiguous and physically invariant.

---

## 4. Bridge-Day (*Brückentag*) Rule

In German industry, bridge days feature extensive scheduled facility shutdowns and workforce leave, which dramatically suppress power demand:
- **Rule Definition**: A regular working day (Monday through Friday) that falls between a statutory public holiday and a weekend:
  1. A **Friday** where the preceding **Thursday** is a nationwide public holiday (e.g. the Friday after Ascension Day / *Christi Himmelfahrt*).
  2. A **Monday** where the following **Tuesday** is a nationwide public holiday (e.g. Monday preceding German Unity Day when Oct 3 is a Tuesday).
  3. The day itself must NOT be a statutory public holiday.

---

## 5. Deterministic Solar Physical Proxy

- **Concept**: Electricity spot prices are heavily depressed by midday solar photovoltaic generation ("duck curve" and negative prices). Rather than using look-ahead weather forecasts or historical telemetry, we engineer an astronomical proxy.
- **Coordinates**: Geographic center of Germany (51.1657° N, 10.4515° E, Niederdorla, Thuringia).
- **Mathematical Formulation**:
  Using the Spencer (1971) solar position model:
  1. Fractional year $\gamma = \frac{2\pi}{365.25} (d - 1 + \frac{\text{hour}_{\text{UTC}} - 12}{24})$
  2. Solar declination $\delta$ and Equation of Time $E_t$.
  3. True solar time $T_{\text{solar}} = \text{hour}_{\text{UTC}} \times 60 + 4 \times \lambda + E_t$.
  4. Solar elevation angle $\alpha$:
     $$\sin(\alpha) = \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\cos(H)$$
  5. Proxy Factor: `solar_elevation_proxy` $= \max(0, \sin(\alpha))$.
- **Properties**: 0.0 at night, rising smoothly to ~0.88 at solar noon on the summer solstice and ~0.27 at solar noon on the winter solstice. 100% deterministic, zero weather data, zero lookahead.

---

## 6. Warm-Up Period and Dropped Observations

- **Maximum Lookback Required**: 168 hours (7 full days).
- **Rows Dropped**: Exactly **168 rows** (from `2023-01-01 00:00:00+01:00` to `2023-01-07 23:00:00+01:00`).
- **First Usable Delivery Timestamp**:
  - Local: `2023-01-08 00:00:00+01:00`
  - UTC: `2023-01-07 23:00:00+00:00`
- **Last Usable Delivery Timestamp**:
  - Local: `2024-12-31 23:00:00+01:00`
  - UTC: `2024-12-31 22:00:00+00:00`
- **Final Valid Row Count**: **17,376 rows** (out of 17,544 initial rows).

---

## 7. Temporal Model Split Architecture

To prevent look-ahead bias and respect market evolution, data is split strictly along chronological German market time boundaries:

| Split | Local Timestamp Window (Europe/Berlin) | UTC Timestamp Window | Elapsed Days | Hourly Rows | Share | Purpose & Usage Rule |
|---|---|---|---|---|---|---|
| **TRAIN** | `2023-01-08 00:00:00+01:00` to `2023-12-31 23:00:00+01:00` | `2023-01-07 23:00:00+00:00` to `2023-12-31 22:00:00+00:00` | ~358 days | **8,592** | 49.4% | Model estimation, parameter fitting, lag baseline fitting. |
| **VALIDATION** | `2024-01-01 00:00:00+01:00` to `2024-06-30 23:00:00+02:00` | `2023-12-31 23:00:00+00:00` to `2024-06-30 21:00:00+00:00` | 182 days (H1 2024) | **4,367** | 25.1% | Hyperparameter tuning, feature selection, probability threshold tuning. (Includes March DST 23h: 182×24 - 1 = 4,367h). |
| **FINAL HOLDOUT** | `2024-07-01 00:00:00+02:00` to `2024-12-31 23:00:00+01:00` | `2024-06-30 22:00:00+00:00` to `2024-12-31 22:00:00+00:00` | 184 days (H2 2024) | **4,417** | 25.4% | Final out-of-sample benchmark evaluation only. (Includes October DST 25h: 184×24 + 1 = 4,417h). |

> **MANDATORY PROTOCOL ON FINAL MODEL HOLDOUT**:  
> *"Final model holdout — excluded from model fitting, hyperparameter tuning, feature selection, and threshold selection. It was previously inspected only for data-quality validation."*  
> From this point onward, no holdout target behavior shall be inspected for model-development decisions.

---

## 8. Data Quality & Null Verification

- **Total Missing / Null Predictors in Usable Core Strict Matrix**: `0`
- **Total Missing / Null Predictors in Usable Forecast Extension Matrix**: `0`
- **Unexpected Nulls Detected**: **False**

Both feature sets are 100% contiguous and complete without requiring any interpolation or imputation.
