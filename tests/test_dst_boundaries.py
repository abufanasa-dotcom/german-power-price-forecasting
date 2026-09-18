"""
Automated tests for German DST transitions (Spring and Autumn 2023)
across all three core SMARD time series:
- 4169 DE-LU: Day-Ahead Wholesale Electricity Price
- 411 DE: Forecasted Total Electricity Consumption
- 410 DE: Actual Total Electricity Consumption
"""

from pathlib import Path
import pytest
import pandas as pd

from src.ingestion import (
    FILTER_DAY_AHEAD_PRICE,
    FILTER_LOAD_FORECAST,
    FILTER_LOAD_ACTUAL,
    fetch_smard_chunk,
    parse_smard_series,
)

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Spring 2023 DST transition: Sunday 2023-03-26 (23-hour day, 02:00 skipped)
SPRING_START_BERLIN = "2023-03-24 00:00:00+01:00"
SPRING_END_BERLIN = "2023-03-28 23:00:00+02:00"
SPRING_CHUNKS = [1679266800000, 1679868000000]  # Mon Mar 20 & Mon Mar 27

# Autumn 2023 DST transition: Sunday 2023-10-29 (25-hour day, 02:00 repeated)
AUTUMN_START_BERLIN = "2023-10-27 00:00:00+02:00"
AUTUMN_END_BERLIN = "2023-10-31 23:00:00+01:00"
AUTUMN_CHUNKS = [1698012000000, 1698620400000]  # Mon Oct 23 & Mon Oct 30

SERIES_CONFIGS = [
    (FILTER_DAY_AHEAD_PRICE, "DE-LU", "day_ahead_price_eur_mwh", "Price"),
    (FILTER_LOAD_FORECAST, "DE", "load_forecast_mw", "Load Forecast"),
    (FILTER_LOAD_ACTUAL, "DE", "load_actual_mw", "Load Actual"),
]


def _fetch_window_dataframe(filter_id: int, region: str, chunks: list, col_name: str, start_b: str, end_b: str) -> pd.DataFrame:
    dfs = []
    for chunk_ts in chunks:
        raw = fetch_smard_chunk(filter_id, region, chunk_ts, raw_dir=RAW_DATA_DIR)
        dfs.append(parse_smard_series(raw, col_name))
    full_df = pd.concat(dfs, ignore_index=True)
    filtered = full_df[(full_df["timestamp_berlin"] >= start_b) & (full_df["timestamp_berlin"] <= end_b)].copy()
    filtered = filtered.sort_values("timestamp_utc").reset_index(drop=True)
    return filtered


@pytest.mark.parametrize("filter_id,region,col_name,label", SERIES_CONFIGS)
def test_spring_dst_transition(filter_id, region, col_name, label):
    """
    Spring DST window: 2023-03-24 00:00 through 2023-03-28 23:00 Europe/Berlin.
    Expected hours: 119 (4 normal 24h days + one 23h DST day).
    """
    df = _fetch_window_dataframe(filter_id, region, SPRING_CHUNKS, col_name, SPRING_START_BERLIN, SPRING_END_BERLIN)

    # 1. Observation count
    assert len(df) == 119, f"{label} expected 119 hours, got {len(df)}"

    # 2. Timestamp uniqueness & continuity in UTC
    assert df["timestamp_utc"].duplicated().sum() == 0, f"{label} contains duplicate UTC timestamps"
    expected_utc = pd.date_range("2023-03-23 23:00:00", "2023-03-28 21:00:00", freq="h", tz="UTC")
    assert len(expected_utc) == 119
    assert list(df["timestamp_utc"]) == list(expected_utc), f"{label} UTC timestamps do not match expected continuous hourly grid"

    # 3. Null values check
    assert df[col_name].isna().sum() == 0, f"{label} contains null values"

    # 4. Boundary timestamps
    assert str(df["timestamp_berlin"].iloc[0]) == "2023-03-24 00:00:00+01:00"
    assert str(df["timestamp_berlin"].iloc[-1]) == "2023-03-28 23:00:00+02:00"
    assert str(df["timestamp_utc"].iloc[0]) == "2023-03-23 23:00:00+00:00"
    assert str(df["timestamp_utc"].iloc[-1]) == "2023-03-28 21:00:00+00:00"

    # 5. Verify that local 02:00 DOES NOT exist on 2023-03-26 (clock jumped from 01:59 CET to 03:00 CEST)
    mar26_hours = df[df["timestamp_berlin"].dt.strftime("%Y-%m-%d") == "2023-03-26"]["timestamp_berlin"]
    assert len(mar26_hours) == 23, f"2023-03-26 should have exactly 23 hours, found {len(mar26_hours)}"
    hour_strings = [t.strftime("%H:%M") for t in mar26_hours]
    assert "02:00" not in hour_strings, "Local 02:00 should not exist on spring DST transition day"
    assert "01:00" in hour_strings
    assert "03:00" in hour_strings


@pytest.mark.parametrize("filter_id,region,col_name,label", SERIES_CONFIGS)
def test_autumn_dst_transition(filter_id, region, col_name, label):
    """
    Autumn DST window: 2023-10-27 00:00 through 2023-10-31 23:00 Europe/Berlin.
    Expected hours: 121 (4 normal 24h days + one 25h DST day).
    """
    df = _fetch_window_dataframe(filter_id, region, AUTUMN_CHUNKS, col_name, AUTUMN_START_BERLIN, AUTUMN_END_BERLIN)

    # 1. Observation count
    assert len(df) == 121, f"{label} expected 121 hours, got {len(df)}"

    # 2. Timestamp uniqueness & continuity in UTC
    assert df["timestamp_utc"].duplicated().sum() == 0, f"{label} contains duplicate UTC timestamps"
    expected_utc = pd.date_range("2023-10-26 22:00:00", "2023-10-31 22:00:00", freq="h", tz="UTC")
    assert len(expected_utc) == 121
    assert list(df["timestamp_utc"]) == list(expected_utc), f"{label} UTC timestamps do not match expected continuous hourly grid"

    # 3. Null values check
    assert df[col_name].isna().sum() == 0, f"{label} contains null values"

    # 4. Boundary timestamps
    assert str(df["timestamp_berlin"].iloc[0]) == "2023-10-27 00:00:00+02:00"
    assert str(df["timestamp_berlin"].iloc[-1]) == "2023-10-31 23:00:00+01:00"
    assert str(df["timestamp_utc"].iloc[0]) == "2023-10-26 22:00:00+00:00"
    assert str(df["timestamp_utc"].iloc[-1]) == "2023-10-31 22:00:00+00:00"

    # 5. Verify that local 02:00 occurs TWICE on 2023-10-29 with different UTC offsets
    oct29_hours = df[df["timestamp_berlin"].dt.strftime("%Y-%m-%d") == "2023-10-29"]["timestamp_berlin"]
    assert len(oct29_hours) == 25, f"2023-10-29 should have exactly 25 hours, found {len(oct29_hours)}"
    h02 = [t for t in oct29_hours if t.strftime("%H:%M") == "02:00"]
    assert len(h02) == 2, f"Local 02:00 must occur exactly twice, found {len(h02)}"

    # First 02:00 occurrence is CEST (UTC+2)
    assert h02[0].strftime("%Z") == "CEST" or h02[0].strftime("%z") == "+0200"
    assert str(h02[0].tz_convert("UTC")) == "2023-10-29 00:00:00+00:00"

    # Second 02:00 occurrence is CET (UTC+1)
    assert h02[1].strftime("%Z") == "CET" or h02[1].strftime("%z") == "+0100"
    assert str(h02[1].tz_convert("UTC")) == "2023-10-29 01:00:00+00:00"
