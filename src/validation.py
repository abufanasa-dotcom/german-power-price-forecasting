"""
Validation module for German Power Market Datasets (2023-2024).

Provides structural checks, temporal continuity, price distribution metrics,
negative price streak analysis, and forecast diagnostic calculations.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

EXPECTED_HOURS_2023 = 8760
EXPECTED_HOURS_2024 = 8784  # Leap year
TOTAL_EXPECTED_HOURS = 17544


def validate_structural_integrity(df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    """
    Validate basic structural properties, types, and timestamp continuity.
    """
    row_count = len(df)
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    first_ts_local = str(df["timestamp_local"].iloc[0])
    last_ts_local = str(df["timestamp_local"].iloc[-1])
    first_ts_utc = str(df["timestamp_utc"].iloc[0])
    last_ts_utc = str(df["timestamp_utc"].iloc[-1])

    tz_utc = str(df["timestamp_utc"].dt.tz)
    tz_local = str(df["timestamp_local"].dt.tz)

    is_monotonic_utc = bool(df["timestamp_utc"].is_monotonic_increasing)
    dup_utc = int(df["timestamp_utc"].duplicated().sum())

    expected_utc = pd.date_range(df["timestamp_utc"].iloc[0], df["timestamp_utc"].iloc[-1], freq="h", tz="UTC")
    missing_utc = len(set(expected_utc) - set(df["timestamp_utc"]))

    null_count = int(df[value_col].isna().sum())
    non_numeric_count = int((~df[value_col].apply(lambda x: isinstance(x, (int, float, np.number)))).sum())

    return {
        "row_count": row_count,
        "columns": columns,
        "dtypes": dtypes,
        "first_timestamp_local": first_ts_local,
        "last_timestamp_local": last_ts_local,
        "first_timestamp_utc": first_ts_utc,
        "last_timestamp_utc": last_ts_utc,
        "timezone_utc": tz_utc,
        "timezone_local": tz_local,
        "is_monotonic_utc": is_monotonic_utc,
        "duplicate_timestamps": dup_utc,
        "missing_timestamps": missing_utc,
        "null_values": null_count,
        "non_numeric_values": non_numeric_count,
    }


def compute_yearly_monthly_coverage(df: pd.DataFrame) -> Tuple[Dict[int, int], pd.DataFrame]:
    """
    Compute yearly counts and a detailed 24-month coverage table.
    """
    df_temp = df.copy()
    df_temp["year"] = df_temp["timestamp_local"].dt.year
    df_temp["month"] = df_temp["timestamp_local"].dt.month
    df_temp["year_month"] = df_temp["timestamp_local"].dt.strftime("%Y-%m")

    yearly_counts = df_temp["year"].value_counts().to_dict()

    # Generate exact expected hours per month using Berlin calendar
    records = []
    for ym, group in df_temp.groupby("year_month"):
        returned_h = len(group)
        # Expected hours by month in Europe/Berlin
        year = int(ym.split("-")[0])
        month = int(ym.split("-")[1])
        start_m = pd.Timestamp(f"{year}-{month:02d}-01 00:00:00", tz="Europe/Berlin")
        if month == 12:
            end_m = pd.Timestamp(f"{year+1}-01-01 00:00:00", tz="Europe/Berlin") - pd.Timedelta(hours=1)
        else:
            end_m = pd.Timestamp(f"{year}-{month+1:02d}-01 00:00:00", tz="Europe/Berlin") - pd.Timedelta(hours=1)
        
        expected_grid = pd.date_range(start_m, end_m, freq="h", tz="Europe/Berlin")
        expected_h = len(expected_grid)
        missing_h = expected_h - returned_h

        # DST note
        dst_note = ""
        if month == 3:
            dst_note = "Spring DST (-1h -> 23h on transition Sunday)"
        elif month == 10:
            dst_note = "Autumn DST (+1h -> 25h on transition Sunday)"
        elif month == 2 and year == 2024:
            dst_note = "Leap year (29 days = 696h)"

        records.append({
            "year_month": ym,
            "expected_hours": expected_h,
            "returned_hours": returned_h,
            "missing_hours": missing_h,
            "notes": dst_note,
        })

    monthly_df = pd.DataFrame(records)
    return yearly_counts, monthly_df


def compute_price_metrics(df_price: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive Day-Ahead price statistics, threshold counts, and percentiles.
    """
    p = df_price["day_ahead_price_eur_mwh"].astype(float)

    metrics = {
        "count": len(p),
        "mean": float(p.mean()),
        "std": float(p.std()),
        "min": float(p.min()),
        "median": float(p.median()),
        "max": float(p.max()),
        "p1": float(np.percentile(p, 1)),
        "p5": float(np.percentile(p, 5)),
        "p25": float(np.percentile(p, 25)),
        "p75": float(np.percentile(p, 75)),
        "p95": float(np.percentile(p, 95)),
        "p99": float(np.percentile(p, 99)),
        "negative_hours": int((p < 0.0).sum()),
        "negative_pct": float((p < 0.0).mean() * 100),
        "zero_hours": int((p == 0.0).sum()),
        "price_gt_200_hours": int((p > 200.0).sum()),
        "price_gt_300_hours": int((p > 300.0).sum()),
        "price_le_neg100_hours": int((p <= -100.0).sum()),
    }
    return metrics


def analyze_extreme_min_price(df_price: pd.DataFrame) -> Dict[str, Any]:
    """
    Investigate the minimum price observation (-500 EUR/MWh).
    """
    min_val = df_price["day_ahead_price_eur_mwh"].min()
    min_rows = df_price[df_price["day_ahead_price_eur_mwh"] == min_val]
    count = len(min_rows)

    occurrences = []
    for idx in min_rows.index:
        start_idx = max(0, idx - 3)
        end_idx = min(len(df_price) - 1, idx + 3)
        context = df_price.loc[start_idx:end_idx][
            ["timestamp_utc", "timestamp_local", "day_ahead_price_eur_mwh"]
        ].to_dict(orient="records")

        occurrences.append({
            "index": int(idx),
            "timestamp_local": str(df_price.loc[idx, "timestamp_local"]),
            "timestamp_utc": str(df_price.loc[idx, "timestamp_utc"]),
            "price": float(df_price.loc[idx, "day_ahead_price_eur_mwh"]),
            "context_before_after": context,
        })

    return {
        "min_value": float(min_val),
        "occurrence_count": count,
        "is_isolated": count == 1,
        "occurrences": occurrences,
    }


def analyze_negative_prices(df_price: pd.DataFrame) -> Dict[str, Any]:
    """
    Break down negative prices by year, month, hour of day, weekday/weekend,
    and detect the longest consecutive negative price streak.
    """
    df = df_price.copy()
    df["is_negative"] = df["day_ahead_price_eur_mwh"] < 0.0
    df["year"] = df["timestamp_local"].dt.year
    df["month"] = df["timestamp_local"].dt.month
    df["hour"] = df["timestamp_local"].dt.hour
    df["dayofweek"] = df["timestamp_local"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"] >= 5

    by_year = df.groupby("year")["is_negative"].agg(["count", "sum", "mean"]).to_dict(orient="index")
    by_month = df.groupby(["year", "month"])["is_negative"].sum().to_dict()
    by_hour = df.groupby("hour")["is_negative"].sum().to_dict()
    by_weekend = df.groupby("is_weekend")["is_negative"].agg(["count", "sum", "mean"]).to_dict(orient="index")

    # Longest consecutive negative event
    df["streak_block"] = (df["is_negative"] != df["is_negative"].shift()).cumsum()
    neg_events = df[df["is_negative"]].groupby("streak_block")

    streaks = []
    for block_id, g in neg_events:
        streaks.append({
            "start_local": str(g["timestamp_local"].iloc[0]),
            "end_local": str(g["timestamp_local"].iloc[-1]),
            "duration_hours": len(g),
            "min_price": float(g["day_ahead_price_eur_mwh"].min()),
            "mean_price": float(g["day_ahead_price_eur_mwh"].mean()),
        })

    streaks.sort(key=lambda x: x["duration_hours"], reverse=True)
    longest_event = streaks[0] if streaks else None

    return {
        "by_year": by_year,
        "by_month": {f"{k[0]}-{k[1]:02d}": v for k, v in by_month.items()},
        "by_hour": by_hour,
        "by_weekend": by_weekend,
        "total_negative_streaks": len(streaks),
        "longest_negative_event": longest_event,
        "top_5_longest_events": streaks[:5],
    }


def compute_load_metrics(df_load: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    """
    Compute descriptive statistics, profile aggregations, and non-positive counts for load series.
    """
    vals = df_load[value_col].astype(float)
    df_temp = df_load.copy()
    df_temp["hour"] = df_temp["timestamp_local"].dt.hour
    df_temp["month"] = df_temp["timestamp_local"].dt.month

    hourly_profile = df_temp.groupby("hour")[value_col].mean().to_dict()
    monthly_profile = df_temp.groupby("month")[value_col].mean().to_dict()

    return {
        "count": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "median": float(vals.median()),
        "max": float(vals.max()),
        "p1": float(np.percentile(vals, 1)),
        "p99": float(np.percentile(vals, 99)),
        "values_le_zero": int((vals <= 0.0).sum()),
        "hourly_profile": hourly_profile,
        "monthly_profile": monthly_profile,
    }


def compute_forecast_diagnostic(df_forecast: pd.DataFrame, df_actual: pd.DataFrame) -> Dict[str, Any]:
    """
    Diagnostic comparison between load forecast and actual load joined on timestamp_utc.
    """
    merged = df_forecast[["timestamp_utc", "timestamp_local", "load_forecast_mw"]].merge(
        df_actual[["timestamp_utc", "load_actual_mw"]],
        on="timestamp_utc",
        how="inner",
    )

    unmatched = (len(df_forecast) - len(merged)) + (len(df_actual) - len(merged))

    err = merged["load_forecast_mw"] - merged["load_actual_mw"]
    abs_err = err.abs()

    mae = float(abs_err.mean())
    bias = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    corr = float(merged["load_forecast_mw"].corr(merged["load_actual_mw"]))

    merged["abs_error"] = abs_err
    top10_diff = merged.sort_values("abs_error", ascending=False).head(10)

    top10_records = []
    for _, r in top10_diff.iterrows():
        top10_records.append({
            "timestamp_local": str(r["timestamp_local"]),
            "timestamp_utc": str(r["timestamp_utc"]),
            "load_forecast_mw": float(r["load_forecast_mw"]),
            "load_actual_mw": float(r["load_actual_mw"]),
            "error_mw": float(r["load_forecast_mw"] - r["load_actual_mw"]),
            "abs_error_mw": float(r["abs_error"]),
        })

    return {
        "matched_rows": len(merged),
        "unmatched_rows": unmatched,
        "mae_mw": mae,
        "mean_error_bias_mw": bias,
        "rmse_mw": rmse,
        "correlation": corr,
        "top_10_largest_differences": top10_records,
    }


def validate_cross_dataset_alignment(
    df_price: pd.DataFrame,
    df_forecast: pd.DataFrame,
    df_actual: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Verify timestamp parity and delivery-hour alignment across all three datasets.
    """
    utc_price = df_price["timestamp_utc"]
    utc_fc = df_forecast["timestamp_utc"]
    utc_act = df_actual["timestamp_utc"]

    local_price = df_price["timestamp_local"]
    local_fc = df_forecast["timestamp_local"]
    local_act = df_actual["timestamp_local"]

    utc_match_fc = bool((utc_price == utc_fc).all())
    utc_match_act = bool((utc_price == utc_act).all())
    local_match_fc = bool((local_price == local_fc).all())
    local_match_act = bool((local_price == local_act).all())

    # Full 3-way inner merge
    merged = df_price[["timestamp_utc"]].merge(
        df_forecast[["timestamp_utc"]], on="timestamp_utc", how="inner"
    ).merge(
        df_actual[["timestamp_utc"]], on="timestamp_utc", how="inner"
    )

    unmatched_price = len(df_price) - len(merged)
    unmatched_fc = len(df_forecast) - len(merged)
    unmatched_act = len(df_actual) - len(merged)

    return {
        "all_utc_identical": utc_match_fc and utc_match_act,
        "all_local_identical": local_match_fc and local_match_act,
        "merged_row_count": len(merged),
        "unmatched_price_rows": unmatched_price,
        "unmatched_forecast_rows": unmatched_fc,
        "unmatched_actual_rows": unmatched_act,
        "total_unmatched_rows": unmatched_price + unmatched_fc + unmatched_act,
    }

