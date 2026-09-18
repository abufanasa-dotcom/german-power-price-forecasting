"""
Feature Engineering Module for German Day-Ahead Electricity Market Forecasting.

Builds two distinct, leakage-free feature sets:
1. Core Strict Pre-Auction Features:
   - Price historical daily lags (24h to 168h) and derived 7-day statistics.
   - Actual load historical lags (minimum 48h lookback).
   - Europe/Berlin calendar and deterministic holiday/bridge-day indicators.
   - Deterministic, leakage-free astronomical solar elevation proxy for Germany.
2. Official Forecast Extension:
   - Adds Day-Ahead total load forecast, 24h forecast difference, and daily peak ratio.
   - Qualified under: "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED".

All temporal lags are computed strictly on the sorted UTC timeline.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd

# Geographic coordinates for the center of Germany (Niederdorla, Thuringia)
GERMANY_CENTER_LAT = 51.1657
GERMANY_CENTER_LON = 10.4515

WARMUP_HOURS = 168  # 7 days of 24h history required for 168h lags

TRAIN_END_LOCAL = "2023-12-31 23:00:00+01:00"
VAL_START_LOCAL = "2024-01-01 00:00:00+01:00"
VAL_END_LOCAL = "2024-06-30 23:00:00+02:00"
HOLDOUT_START_LOCAL = "2024-07-01 00:00:00+02:00"
HOLDOUT_END_LOCAL = "2024-12-31 23:00:00+01:00"

PRICE_DAILY_LAGS = [24, 48, 72, 96, 120, 144, 168]
LOAD_ACTUAL_LAGS = [48, 168]


def compute_solar_elevation_proxy(
    timestamps_utc: pd.Series,
    lat: float = GERMANY_CENTER_LAT,
    lon: float = GERMANY_CENTER_LON,
) -> pd.Series:
    """
    Compute a deterministic astronomical solar elevation proxy.
    
    Formula based on standard NOAA / Spencer (1971) solar position equations:
    - Fractional year gamma (radians)
    - Solar declination delta
    - Equation of time (minutes)
    - True solar time and hour angle
    - Solar elevation angle alpha = arcsin(sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(ha))
    - Proxy = max(0, sin(alpha)), representing normalized clear-sky extraterrestrial irradiance.
    
    Contains ZERO observed weather data and ZERO lookahead leakage.
    """
    dt_series = pd.to_datetime(timestamps_utc)
    day_of_year = dt_series.dt.dayofyear.values
    hour_utc = (dt_series.dt.hour + dt_series.dt.minute / 60.0).values
    
    lat_rad = np.radians(lat)
    
    # Fractional year in radians
    gamma = 2.0 * np.pi / 365.25 * (day_of_year - 1 + (hour_utc - 12.0) / 24.0)
    
    # Solar declination (Spencer 1971)
    delta = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    
    # Equation of time in minutes
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    
    # True solar time in minutes
    solar_time_min = hour_utc * 60.0 + 4.0 * lon + eqtime
    hour_angle_deg = (solar_time_min / 4.0) - 180.0
    ha_rad = np.radians(hour_angle_deg)
    
    sin_elev = np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.cos(ha_rad)
    proxy = np.maximum(0.0, sin_elev)
    
    return pd.Series(proxy, index=timestamps_utc.index, name="solar_elevation_proxy")


def get_german_public_holidays(years: List[int] = [2023, 2024]) -> Set[date]:
    """
    Return all statutory nationwide public holidays in Germany for given years.
    
    Nationwide holidays observed across all 16 federal states:
    - Neujahr (New Year's Day): Jan 1
    - Karfreitag (Good Friday): Easter - 2 days
    - Ostermontag (Easter Monday): Easter + 1 day
    - Tag der Arbeit (Labour Day): May 1
    - Christi Himmelfahrt (Ascension Day): Easter + 39 days (always Thursday)
    - Pfingstmontag (Whit Monday): Easter + 50 days (always Monday)
    - Tag der Deutschen Einheit (German Unity Day): Oct 3
    - 1. Weihnachtstag (Christmas Day): Dec 25
    - 2. Weihnachtstag (Boxing Day): Dec 26
    """
    # Easter Sunday dates:
    # 2023: April 9
    # 2024: March 31
    easter_dates = {
        2023: date(2023, 4, 9),
        2024: date(2024, 3, 31),
    }
    
    holidays: Set[date] = set()
    for y in years:
        # Fixed-date nationwide holidays
        holidays.add(date(y, 1, 1))    # New Year
        holidays.add(date(y, 5, 1))    # Labour Day
        holidays.add(date(y, 10, 3))   # German Unity Day
        holidays.add(date(y, 12, 25))  # Christmas Day
        holidays.add(date(y, 12, 26))  # 2nd Christmas Day
        
        # Movable holidays tied to Easter
        if y in easter_dates:
            e = easter_dates[y]
            holidays.add(e - timedelta(days=2))   # Good Friday
            holidays.add(e + timedelta(days=1))   # Easter Monday
            holidays.add(e + timedelta(days=39))  # Ascension Day (Thursday)
            holidays.add(e + timedelta(days=50))  # Whit Monday
            
    return holidays


def get_german_bridge_days(holidays: Set[date]) -> Set[date]:
    """
    Identify bridge days (Brückentage) in Germany.
    
    Explicit Bridge-Day Rule:
    A regular working day (Monday through Friday) that falls between a public holiday
    and a weekend:
    1. A Friday where the preceding Thursday is a public holiday (e.g. Friday after Ascension Day).
    2. A Monday where the following Tuesday is a public holiday (e.g. Monday before Oct 3 if Oct 3 is Tuesday).
    The day itself must NOT be a public holiday.
    """
    bridge_days: Set[date] = set()
    for h in holidays:
        # Rule 1: If holiday is Thursday, the following Friday is a bridge day
        if h.weekday() == 3:  # Thursday
            fri = h + timedelta(days=1)
            if fri not in holidays:
                bridge_days.add(fri)
        # Rule 2: If holiday is Tuesday, the preceding Monday is a bridge day
        if h.weekday() == 1:  # Tuesday
            mon = h - timedelta(days=1)
            if mon not in holidays:
                bridge_days.add(mon)
                
    return bridge_days


def engineer_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Europe/Berlin market calendar indicators and trigonometric encodings.
    """
    out = df.copy()
    local_ts = out["timestamp_local"]
    
    out["hour_of_day"] = local_ts.dt.hour
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour_of_day"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour_of_day"] / 24.0)
    
    out["day_of_week"] = local_ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    
    out["month"] = local_ts.dt.month
    out["month_sin"] = np.sin(2.0 * np.pi * (out["month"] - 1.0) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (out["month"] - 1.0) / 12.0)
    
    # Daylight Saving Time indicator (1 during CEST, 0 during CET)
    out["is_dst"] = local_ts.apply(lambda dt: int(bool(dt.dst()))).values
    
    # Nationwide German holidays and bridge days
    holidays = get_german_public_holidays([2023, 2024])
    bridge_days = get_german_bridge_days(holidays)
    
    dates_series = local_ts.dt.date
    out["is_public_holiday_de"] = dates_series.isin(holidays).astype(int)
    out["is_bridge_day"] = dates_series.isin(bridge_days).astype(int)
    
    return out


def engineer_price_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate historical price daily lags and derived 7-day statistics.
    
    Computed strictly on the sorted UTC timeline.
    All lags >= 24h, preventing any same-day or lookahead leakage.
    """
    out = df.copy()
    
    # Generate 7 daily lags: 24h, 48h, 72h, 96h, 120h, 144h, 168h
    daily_lag_cols = []
    for lag in PRICE_DAILY_LAGS:
        col_name = f"price_lag_{lag}h"
        out[col_name] = out["day_ahead_price_eur_mwh"].shift(lag)
        daily_lag_cols.append(col_name)
        
    # Derived 7-day daily lag summary statistics (strictly from historical lags)
    out["price_daily_lag_mean_7d"] = out[daily_lag_cols].mean(axis=1)
    out["price_daily_lag_std_7d"] = out[daily_lag_cols].std(axis=1, ddof=1)
    
    return out


def engineer_load_actual_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate actual load historical lags on sorted UTC timeline.
    
    Minimum lookback is 48 hours to respect retrospective publication delays.
    Lags shorter than 48h are strictly prohibited.
    """
    out = df.copy()
    for lag in LOAD_ACTUAL_LAGS:
        col_name = f"load_actual_lag_{lag}h"
        out[col_name] = out["load_actual_mw"].shift(lag)
    return out


def engineer_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Official Forecast Extension features.
    
    Includes:
    - load_forecast_mw (Filter 411)
    - load_forecast_diff_24h (forecast vs same hour yesterday)
    - load_forecast_daily_peak_ratio (forecast divided by max forecast of that delivery day)
    
    Retains qualification: "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED".
    """
    out = df.copy()
    
    # 24h difference in forecasted load
    out["load_forecast_diff_24h"] = out["load_forecast_mw"] - out["load_forecast_mw"].shift(24)
    
    # Daily peak ratio: forecast divided by peak forecast for that market delivery day
    daily_max = out.groupby(out["timestamp_local"].dt.date)["load_forecast_mw"].transform("max")
    out["load_forecast_daily_peak_ratio"] = out["load_forecast_mw"] / daily_max
    
    return out


def assign_temporal_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign split labels without overlap:
    - 'train': Usable rows in 2023 through 2023-12-31 23:00 Europe/Berlin
    - 'val': 2024-01-01 00:00 through 2024-06-30 23:00 Europe/Berlin
    - 'holdout': 2024-07-01 00:00 through 2024-12-31 23:00 Europe/Berlin
    """
    out = df.copy()
    local_ts = out["timestamp_local"]
    
    conditions = [
        local_ts <= TRAIN_END_LOCAL,
        (local_ts >= VAL_START_LOCAL) & (local_ts <= VAL_END_LOCAL),
        (local_ts >= HOLDOUT_START_LOCAL) & (local_ts <= HOLDOUT_END_LOCAL),
    ]
    choices = ["train", "val", "holdout"]
    out["split"] = np.select(conditions, choices, default="unknown")
    
    return out


def build_feature_datasets(
    df_price: pd.DataFrame,
    df_load_forecast: pd.DataFrame,
    df_load_actual: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Full pipeline to build both Core Strict and Official Forecast Extension feature sets.
    
    Drops initial WARMUP_HOURS (168 rows) lacking history.
    """
    # 1. Merge all three datasets on timestamp_utc and timestamp_local
    merged = df_price.merge(
        df_load_forecast[["timestamp_utc", "load_forecast_mw"]],
        on="timestamp_utc",
        how="inner",
    ).merge(
        df_load_actual[["timestamp_utc", "load_actual_mw"]],
        on="timestamp_utc",
        how="inner",
    )
    
    # 2. Sort strictly by timestamp_utc
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    initial_rows = len(merged)
    
    # 3. Targets
    merged["is_negative_price"] = (merged["day_ahead_price_eur_mwh"] < 0.0).astype(int)
    
    # 4. Feature engineering
    merged = engineer_price_lag_features(merged)
    merged = engineer_load_actual_lag_features(merged)
    merged = engineer_calendar_features(merged)
    merged["solar_elevation_proxy"] = compute_solar_elevation_proxy(merged["timestamp_utc"])
    merged = engineer_forecast_features(merged)
    
    # 5. Drop initial warm-up rows (168 rows)
    usable_df = merged.iloc[WARMUP_HOURS:].copy().reset_index(drop=True)
    rows_dropped = WARMUP_HOURS
    final_rows = len(usable_df)
    
    # 6. Assign temporal splits
    usable_df = assign_temporal_splits(usable_df)
    
    # Column groupings
    meta_cols = ["timestamp_utc", "timestamp_local", "split"]
    target_cols = ["day_ahead_price_eur_mwh", "is_negative_price"]
    
    core_strict_features = [
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_72h",
        "price_lag_96h",
        "price_lag_120h",
        "price_lag_144h",
        "price_lag_168h",
        "price_daily_lag_mean_7d",
        "price_daily_lag_std_7d",
        "load_actual_lag_48h",
        "load_actual_lag_168h",
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "month",
        "month_sin",
        "month_cos",
        "is_dst",
        "is_public_holiday_de",
        "is_bridge_day",
        "solar_elevation_proxy",
    ]
    
    forecast_ext_features = [
        "load_forecast_mw",
        "load_forecast_diff_24h",
        "load_forecast_daily_peak_ratio",
    ]
    
    # Assemble final DataFrames
    df_core = usable_df[meta_cols + target_cols + core_strict_features].copy()
    df_ext = usable_df[meta_cols + target_cols + core_strict_features + forecast_ext_features].copy()
    
    summary = {
        "initial_rows": initial_rows,
        "rows_dropped_warmup": rows_dropped,
        "final_usable_rows": final_rows,
        "first_usable_local": str(usable_df["timestamp_local"].iloc[0]),
        "first_usable_utc": str(usable_df["timestamp_utc"].iloc[0]),
        "last_usable_local": str(usable_df["timestamp_local"].iloc[-1]),
        "last_usable_utc": str(usable_df["timestamp_utc"].iloc[-1]),
        "train_rows": int((usable_df["split"] == "train").sum()),
        "val_rows": int((usable_df["split"] == "val").sum()),
        "holdout_rows": int((usable_df["split"] == "holdout").sum()),
        "core_features_count": len(core_strict_features),
        "forecast_features_count": len(forecast_ext_features),
        "total_ext_features_count": len(core_strict_features) + len(forecast_ext_features),
        "core_feature_names": core_strict_features,
        "forecast_feature_names": forecast_ext_features,
    }
    
    return df_core, df_ext, summary


def generate_feature_manifest() -> Dict[str, Any]:
    """
    Generate machine-readable specification and leakage documentation for every feature.
    """
    manifest = {
        "project": "german-power-price-forecasting",
        "milestone": "Milestone 4 - Feature Engineering & Temporal Split",
        "auction_gate_closure_cet": "12:00 CET/CEST on D-1",
        "targets": {
            "day_ahead_price_eur_mwh": {
                "task": "regression",
                "dtype": "float64",
                "description": "Hourly German Day-Ahead wholesale electricity spot price (EPEX SPOT DE-LU).",
            },
            "is_negative_price": {
                "task": "classification",
                "dtype": "int64",
                "description": "Binary indicator for negative spot price occurrence (day_ahead_price_eur_mwh < 0.00).",
            },
        },
        "features": {
            # --- Price History ---
            "price_lag_24h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(24) on sorted UTC timeline",
                "lookback": "24 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled Day-Ahead price of same delivery hour yesterday (D-1). Published and known prior to gate closure for day D.",
            },
            "price_lag_48h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(48) on sorted UTC timeline",
                "lookback": "48 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled price from D-2. Fully settled and cleared prior to auction gate closure.",
            },
            "price_lag_72h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(72) on sorted UTC timeline",
                "lookback": "72 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled price from D-3. Fully historical.",
            },
            "price_lag_96h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(96) on sorted UTC timeline",
                "lookback": "96 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled price from D-4. Fully historical.",
            },
            "price_lag_120h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(120) on sorted UTC timeline",
                "lookback": "120 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled price from D-5. Fully historical.",
            },
            "price_lag_144h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(144) on sorted UTC timeline",
                "lookback": "144 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled price from D-6. Fully historical.",
            },
            "price_lag_168h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_ahead_price_2023_2024.parquet",
                "transformation": "shift(168) on sorted UTC timeline",
                "lookback": "168 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Settled price of same delivery hour same day last week (D-7). Captures weekly cyclicality.",
            },
            "price_daily_lag_mean_7d": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "Derived from price_lag_24h..168h",
                "transformation": "Row-wise mean across the 7 historical daily lag columns",
                "lookback": "24h to 168h",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Computed strictly from historical daily lags. Never includes current delivery day target.",
            },
            "price_daily_lag_std_7d": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "Derived from price_lag_24h..168h",
                "transformation": "Row-wise sample standard deviation across the 7 historical daily lag columns",
                "lookback": "24h to 168h",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Captures recent price volatility across historical daily lags without target contamination.",
            },
            # --- Actual Load History ---
            "load_actual_lag_48h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "load_actual_2023_2024.parquet",
                "transformation": "shift(48) on sorted UTC timeline",
                "lookback": "48 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Actual total electrical load from D-2. Minimum 48h lag respects SMARD retrospective publication latency.",
            },
            "load_actual_lag_168h": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "load_actual_2023_2024.parquet",
                "transformation": "shift(168) on sorted UTC timeline",
                "lookback": "168 hours",
                "availability_classification": "STRICT_PRE_AUCTION",
                "leakage_rationale": "Actual total electrical load from same hour last week (D-7). Fully historical and settled.",
            },
            # --- Calendar Features ---
            "hour_of_day": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_local",
                "transformation": "Local market hour integer (0-23)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Deterministic delivery hour known infinitely in advance.",
            },
            "hour_sin": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "hour_of_day",
                "transformation": "sin(2*pi*hour / 24)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Trigonometric encoding of diurnal delivery cycle.",
            },
            "hour_cos": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "hour_of_day",
                "transformation": "cos(2*pi*hour / 24)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Trigonometric encoding of diurnal delivery cycle.",
            },
            "day_of_week": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_local",
                "transformation": "Local market day of week (0=Mon, 6=Sun)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Deterministic calendar property known in advance.",
            },
            "is_weekend": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "day_of_week",
                "transformation": "Binary indicator (1 if Sat/Sun, else 0)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Deterministic weekend indicator.",
            },
            "month": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_local",
                "transformation": "Local calendar month integer (1-12)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Deterministic seasonal calendar indicator.",
            },
            "month_sin": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "month",
                "transformation": "sin(2*pi*(month-1) / 12)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Trigonometric encoding of annual seasonality.",
            },
            "month_cos": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "month",
                "transformation": "cos(2*pi*(month-1) / 12)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Trigonometric encoding of annual seasonality.",
            },
            "is_dst": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_local",
                "transformation": "Binary indicator for Daylight Saving Time (1 for CEST, 0 for CET)",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Deterministic legal clock regime in Europe/Berlin.",
            },
            "is_public_holiday_de": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_local",
                "transformation": "Binary indicator for statutory German nationwide public holidays",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Fixed legal calendar dates known years in advance.",
            },
            "is_bridge_day": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_local",
                "transformation": "Binary indicator for Friday following a Thursday holiday or Monday preceding a Tuesday holiday",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_CALENDAR",
                "leakage_rationale": "Deterministic calendar rule. Strongly correlates with German industrial shutdown leave.",
            },
            # --- Solar / Seasonal Physical Proxy ---
            "solar_elevation_proxy": {
                "group": "Core Strict Pre-Auction",
                "source_dataset": "timestamp_utc & coordinates (51.1657N, 10.4515E)",
                "transformation": "max(0, sin(astronomical_solar_elevation))",
                "lookback": "0 hours (Deterministic)",
                "availability_classification": "DETERMINISTIC_PHYSICAL_PROXY",
                "leakage_rationale": "Exact astronomical solar position calculated for geographic center of Germany. 100% deterministic, zero weather data, zero lookahead.",
            },
            # --- Official Forecast Extension ---
            "load_forecast_mw": {
                "group": "Official Forecast Extension",
                "source_dataset": "load_forecast_2023_2024.parquet",
                "transformation": "Raw SMARD Filter 411 total load forecast",
                "lookback": "Published by 12:00 CET on D-1 per EU Regulation 543/2013",
                "availability_classification": "ARCHIVE_VINTAGE_LIMITATION",
                "leakage_rationale": "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED. Retrospective SMARD archive may reflect updated load forecast vintages rather than strictly preserved point-in-time pre-auction bids.",
            },
            "load_forecast_diff_24h": {
                "group": "Official Forecast Extension",
                "source_dataset": "load_forecast_2023_2024.parquet",
                "transformation": "load_forecast_mw - load_forecast_mw.shift(24)",
                "lookback": "24 hours",
                "availability_classification": "ARCHIVE_VINTAGE_LIMITATION",
                "leakage_rationale": "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED. Day-over-day shift in forecasted system demand.",
            },
            "load_forecast_daily_peak_ratio": {
                "group": "Official Forecast Extension",
                "source_dataset": "load_forecast_2023_2024.parquet",
                "transformation": "load_forecast_mw / daily_max_load_forecast",
                "lookback": "Published by 12:00 CET on D-1 for all 24h of day D",
                "availability_classification": "ARCHIVE_VINTAGE_LIMITATION",
                "leakage_rationale": "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED. Normalizes intraday demand shape relative to expected peak.",
            },
        },
    }
    return manifest
