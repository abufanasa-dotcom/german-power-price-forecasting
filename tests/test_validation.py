"""
Automated validation tests for processed German Power Market datasets (2023-2024).

Validates:
- Full 17,544 delivery-hour row counts (8,760 in 2023, 8,784 in 2024 leap year)
- Timestamp uniqueness (zero duplicate timestamps)
- Zero null values in timestamps or metric values
- Strict cross-dataset timestamp parity across price, forecast, and actual load
- Parquet schema and data types
- DST-safe hourly continuity over the entire 2-year horizon
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.validation import (
    EXPECTED_HOURS_2023,
    EXPECTED_HOURS_2024,
    TOTAL_EXPECTED_HOURS,
    validate_structural_integrity,
    validate_cross_dataset_alignment,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def df_price():
    path = PROCESSED_DIR / "day_ahead_price_2023_2024.parquet"
    assert path.exists(), f"Missing processed file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def df_forecast():
    path = PROCESSED_DIR / "load_forecast_2023_2024.parquet"
    assert path.exists(), f"Missing processed file: {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def df_actual():
    path = PROCESSED_DIR / "load_actual_2023_2024.parquet"
    assert path.exists(), f"Missing processed file: {path}"
    return pd.read_parquet(path)


def test_processed_files_exist():
    """Verify all three expected parquet files exist in data/processed."""
    for fname in [
        "day_ahead_price_2023_2024.parquet",
        "load_forecast_2023_2024.parquet",
        "load_actual_2023_2024.parquet",
    ]:
        assert (PROCESSED_DIR / fname).exists(), f"Expected file {fname} not found."


def test_full_row_counts(df_price, df_forecast, df_actual):
    """Confirm exact row count of 17,544 for all three datasets."""
    for name, df in [("Price", df_price), ("Forecast", df_forecast), ("Actual", df_actual)]:
        assert len(df) == TOTAL_EXPECTED_HOURS, f"{name} count {len(df)} != {TOTAL_EXPECTED_HOURS}"
        
        # Check yearly splits
        y2023 = (df["timestamp_local"].dt.year == 2023).sum()
        y2024 = (df["timestamp_local"].dt.year == 2024).sum()
        assert y2023 == EXPECTED_HOURS_2023, f"{name} 2023 hours {y2023} != {EXPECTED_HOURS_2023}"
        assert y2024 == EXPECTED_HOURS_2024, f"{name} 2024 hours {y2024} != {EXPECTED_HOURS_2024}"


def test_processed_schema(df_price, df_forecast, df_actual):
    """Verify expected columns and dtypes for all processed datasets."""
    datasets = [
        ("Price", df_price, "day_ahead_price_eur_mwh"),
        ("Forecast", df_forecast, "load_forecast_mw"),
        ("Actual", df_actual, "load_actual_mw"),
    ]
    for name, df, val_col in datasets:
        expected_cols = ["timestamp_utc", "timestamp_local", val_col]
        assert list(df.columns) == expected_cols, f"{name} schema mismatch: {list(df.columns)}"
        
        # Check dtypes
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"])
        assert str(df["timestamp_utc"].dt.tz) == "UTC"
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp_local"])
        assert str(df["timestamp_local"].dt.tz) == "Europe/Berlin"
        assert pd.api.types.is_float_dtype(df[val_col])


def test_timestamp_uniqueness(df_price, df_forecast, df_actual):
    """Verify no duplicate timestamps exist in UTC or Europe/Berlin."""
    for name, df in [("Price", df_price), ("Forecast", df_forecast), ("Actual", df_actual)]:
        assert df["timestamp_utc"].duplicated().sum() == 0, f"{name} has duplicate UTC timestamps"
        assert df["timestamp_local"].duplicated().sum() == 0, f"{name} has duplicate local timestamps"


def test_no_unexpected_nulls(df_price, df_forecast, df_actual):
    """Verify zero null or missing values across all columns."""
    for name, df in [("Price", df_price), ("Forecast", df_forecast), ("Actual", df_actual)]:
        assert df.isna().sum().sum() == 0, f"{name} contains unexpected null values"


def test_cross_dataset_alignment(df_price, df_forecast, df_actual):
    """Verify identical delivery-hour coverage and zero unmatched rows."""
    alignment = validate_cross_dataset_alignment(df_price, df_forecast, df_actual)
    assert alignment["all_utc_identical"] is True, "UTC timestamps do not align across datasets"
    assert alignment["all_local_identical"] is True, "Local timestamps do not align across datasets"
    assert alignment["merged_row_count"] == TOTAL_EXPECTED_HOURS, "Merged row count mismatch"
    assert alignment["total_unmatched_rows"] == 0, f"Unmatched rows detected: {alignment}"


def test_dst_safe_full_period_continuity(df_price, df_forecast, df_actual):
    """Verify uninterrupted hourly sequence from start to end in UTC."""
    for name, df in [("Price", df_price), ("Forecast", df_forecast), ("Actual", df_actual)]:
        assert df["timestamp_utc"].is_monotonic_increasing, f"{name} timestamps not strictly increasing"
        
        # Check start and end boundaries
        assert str(df["timestamp_local"].iloc[0]) == "2023-01-01 00:00:00+01:00"
        assert str(df["timestamp_local"].iloc[-1]) == "2024-12-31 23:00:00+01:00"
        assert str(df["timestamp_utc"].iloc[0]) == "2022-12-31 23:00:00+00:00"
        assert str(df["timestamp_utc"].iloc[-1]) == "2024-12-31 22:00:00+00:00"

        # Check difference between consecutive UTC timestamps is strictly 1 hour
        time_diffs = df["timestamp_utc"].diff().dropna()
        assert (time_diffs == pd.Timedelta(hours=1)).all(), f"{name} has non-hourly gaps or steps"
