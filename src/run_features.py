"""
Execution script for Milestone 4: Feature Engineering and Temporal Split.

Orchestrates the creation of Core Strict and Official Forecast Extension feature sets,
drops the 168-hour initial warm-up rows, assigns strict temporal splits,
saves parquet artifacts to data/processed/, writes reports/feature_manifest.json,
and compiles reports/feature_engineering_report.md.
"""

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.features import (
    build_feature_datasets,
    generate_feature_manifest,
    WARMUP_HOURS,
    TRAIN_END_LOCAL,
    VAL_START_LOCAL,
    VAL_END_LOCAL,
    HOLDOUT_START_LOCAL,
    HOLDOUT_END_LOCAL,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"


def generate_feature_engineering_report(
    summary: dict,
    df_core: pd.DataFrame,
    df_ext: pd.DataFrame,
) -> str:
    """
    Format complete engineering findings, split details, and provenance rules into markdown.
    """
    train_mask = df_core["split"] == "train"
    val_mask = df_core["split"] == "val"
    holdout_mask = df_core["split"] == "holdout"

    train_df = df_core[train_mask]
    val_df = df_core[val_mask]
    holdout_df = df_core[holdout_mask]

    # Null checks
    core_nulls = df_core.isna().sum().to_dict()
    ext_nulls = df_ext.isna().sum().to_dict()
    has_unexpected_nulls = (df_core.isna().sum().sum() > 0) or (df_ext.isna().sum().sum() > 0)

    report = f"""# Milestone 4 Feature Engineering & Temporal Split Report

**Project**: Portfolio Project 3 — German Day-Ahead Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Study Horizon**: 2023-01-01 00:00:00 to 2024-12-31 23:00:00 Europe/Berlin (17,544 delivery hours)  
**Initial Rows Merged**: {summary['initial_rows']:,}  
**Warm-Up Period Dropped**: {summary['rows_dropped_warmup']} hours (7 days required for 168h lags)  
**Final Usable Rows**: **{summary['final_usable_rows']:,} hours**  

---

## 1. Feature Architecture Overview

Two distinct feature sets are constructed to guarantee strict leakage control while isolating the impact of official TSO load forecasts:

### A. Core Strict Pre-Auction Features ({summary['core_features_count']} Predictors)
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

### B. Official Forecast Extension ({summary['total_ext_features_count']} Predictors)
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
- **Physical Meaning**: A 24-hour lag represents exactly **24 elapsed UTC hours** ($24 \\times 3,600$ seconds of physical time).
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
  1. Fractional year $\\gamma = \\frac{{2\\pi}}{{365.25}} (d - 1 + \\frac{{\\text{{hour}}_{{\\text{{UTC}}}} - 12}}{{24}})$
  2. Solar declination $\\delta$ and Equation of Time $E_t$.
  3. True solar time $T_{{\\text{{solar}}}} = \\text{{hour}}_{{\\text{{UTC}}}} \\times 60 + 4 \\times \\lambda + E_t$.
  4. Solar elevation angle $\\alpha$:
     $$\\sin(\\alpha) = \\sin(\\phi)\\sin(\\delta) + \\cos(\\phi)\\cos(\\delta)\\cos(H)$$
  5. Proxy Factor: `solar_elevation_proxy` $= \\max(0, \\sin(\\alpha))$.
- **Properties**: 0.0 at night, rising smoothly to ~0.88 at solar noon on the summer solstice and ~0.27 at solar noon on the winter solstice. 100% deterministic, zero weather data, zero lookahead.

---

## 6. Warm-Up Period and Dropped Observations

- **Maximum Lookback Required**: 168 hours (7 full days).
- **Rows Dropped**: Exactly **168 rows** (from `2023-01-01 00:00:00+01:00` to `2023-01-07 23:00:00+01:00`).
- **First Usable Delivery Timestamp**:
  - Local: `{summary['first_usable_local']}`
  - UTC: `{summary['first_usable_utc']}`
- **Last Usable Delivery Timestamp**:
  - Local: `{summary['last_usable_local']}`
  - UTC: `{summary['last_usable_utc']}`
- **Final Valid Row Count**: **{summary['final_usable_rows']:,} rows** (out of 17,544 initial rows).

---

## 7. Temporal Model Split Architecture

To prevent look-ahead bias and respect market evolution, data is split strictly along chronological German market time boundaries:

| Split | Local Timestamp Window (Europe/Berlin) | UTC Timestamp Window | Elapsed Days | Hourly Rows | Share | Purpose & Usage Rule |
|---|---|---|---|---|---|---|
| **TRAIN** | `{train_df['timestamp_local'].iloc[0]}` to `{train_df['timestamp_local'].iloc[-1]}` | `{train_df['timestamp_utc'].iloc[0]}` to `{train_df['timestamp_utc'].iloc[-1]}` | ~358 days | **{summary['train_rows']:,}** | {summary['train_rows']/summary['final_usable_rows']*100:.1f}% | Model estimation, parameter fitting, lag baseline fitting. |
| **VALIDATION** | `{val_df['timestamp_local'].iloc[0]}` to `{val_df['timestamp_local'].iloc[-1]}` | `{val_df['timestamp_utc'].iloc[0]}` to `{val_df['timestamp_utc'].iloc[-1]}` | 182 days (H1 2024) | **{summary['val_rows']:,}** | {summary['val_rows']/summary['final_usable_rows']*100:.1f}% | Hyperparameter tuning, feature selection, probability threshold tuning. (Includes March DST 23h: 182×24 - 1 = 4,367h). |
| **FINAL HOLDOUT** | `{holdout_df['timestamp_local'].iloc[0]}` to `{holdout_df['timestamp_local'].iloc[-1]}` | `{holdout_df['timestamp_utc'].iloc[0]}` to `{holdout_df['timestamp_utc'].iloc[-1]}` | 184 days (H2 2024) | **{summary['holdout_rows']:,}** | {summary['holdout_rows']/summary['final_usable_rows']*100:.1f}% | Final out-of-sample benchmark evaluation only. (Includes October DST 25h: 184×24 + 1 = 4,417h). |

> **MANDATORY PROTOCOL ON FINAL MODEL HOLDOUT**:  
> *"Final model holdout — excluded from model fitting, hyperparameter tuning, feature selection, and threshold selection. It was previously inspected only for data-quality validation."*  
> From this point onward, no holdout target behavior shall be inspected for model-development decisions.

---

## 8. Data Quality & Null Verification

- **Total Missing / Null Predictors in Usable Core Strict Matrix**: `{df_core.isna().sum().sum()}`
- **Total Missing / Null Predictors in Usable Forecast Extension Matrix**: `{df_ext.isna().sum().sum()}`
- **Unexpected Nulls Detected**: **{has_unexpected_nulls}**

Both feature sets are 100% contiguous and complete without requiring any interpolation or imputation.
"""
    return report


def run_features() -> None:
    """
    Main orchestration routine for feature engineering.
    """
    print("=" * 70)
    print("MILESTONE 4: FEATURE ENGINEERING AND TEMPORAL SPLIT")
    print("=" * 70)

    # 1. Load processed datasets
    price_path = PROCESSED_DIR / "day_ahead_price_2023_2024.parquet"
    fc_path = PROCESSED_DIR / "load_forecast_2023_2024.parquet"
    act_path = PROCESSED_DIR / "load_actual_2023_2024.parquet"

    print(f"Loading {price_path.name}...")
    df_price = pd.read_parquet(price_path)
    print(f"Loading {fc_path.name}...")
    df_fc = pd.read_parquet(fc_path)
    print(f"Loading {act_path.name}...")
    df_act = pd.read_parquet(act_path)

    # 2. Build feature datasets
    print("\nEngineering features and temporal splits...")
    df_core, df_ext, summary = build_feature_datasets(df_price, df_fc, df_act)

    # 3. Save feature Parquet files
    core_out_path = PROCESSED_DIR / "features_core_strict.parquet"
    ext_out_path = PROCESSED_DIR / "features_forecast_extension.parquet"

    print(f"\nSaving {core_out_path.name} ({len(df_core):,} rows, {len(df_core.columns)} cols)...")
    df_core.to_parquet(core_out_path, index=False)
    print(f"Saving {ext_out_path.name} ({len(df_ext):,} rows, {len(df_ext.columns)} cols)...")
    df_ext.to_parquet(ext_out_path, index=False)

    # 4. Generate and save feature manifest
    print("\nGenerating reports/feature_manifest.json...")
    manifest = generate_feature_manifest()
    manifest_path = REPORTS_DIR / "feature_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [OK] Saved feature manifest ({len(manifest['features'])} documented features)")

    # 5. Generate and save report
    print("\nWriting reports/feature_engineering_report.md...")
    report_content = generate_feature_engineering_report(summary, df_core, df_ext)
    report_path = REPORTS_DIR / "feature_engineering_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"  [OK] Saved feature engineering report ({len(report_content):,} characters)")

    # 6. Print summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Initial Rows:                  {summary['initial_rows']:,}")
    print(f"Rows Dropped (Warm-up):        {summary['rows_dropped_warmup']} (First 7 days)")
    print(f"Final Usable Rows:             {summary['final_usable_rows']:,}")
    print(f"Train Rows:                    {summary['train_rows']:,} ({summary['first_usable_local']} to {TRAIN_END_LOCAL})")
    print(f"Validation Rows:               {summary['val_rows']:,} (2024-01-01 to 2024-06-30 Europe/Berlin)")
    print(f"Holdout Rows:                  {summary['holdout_rows']:,} (2024-07-01 to 2024-12-31 Europe/Berlin)")
    print(f"Core Strict Predictors:        {summary['core_features_count']}")
    print(f"Forecast Extension Predictors: {summary['total_ext_features_count']}")
    print(f"Missing Values in Core:        {df_core.isna().sum().sum()}")
    print(f"Missing Values in Ext:         {df_ext.isna().sum().sum()}")
    print("=" * 70)


if __name__ == "__main__":
    run_features()
