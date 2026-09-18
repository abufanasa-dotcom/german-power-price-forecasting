"""
Targeted integration test for SMARD sample window:
2023-01-02 00:00 through 2023-01-08 23:00 Europe/Berlin.
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from src.ingestion import (
    FILTER_DAY_AHEAD_PRICE,
    FILTER_LOAD_FORECAST,
    FILTER_LOAD_ACTUAL,
    fetch_smard_chunk,
    parse_smard_series,
    audit_sample_series,
)

SAMPLE_EPOCH_MS = 1672614000000  # 2023-01-02 00:00:00 Europe/Berlin
EXPECTED_START_BERLIN = "2023-01-02 00:00:00"
EXPECTED_END_BERLIN = "2023-01-08 23:00:00"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_smard_sample_day_ahead_price():
    """Test Day-Ahead price sample ingestion and schema verification."""
    # 1. Fetch raw chunk and preserve untouched JSON
    raw_data = fetch_smard_chunk(
        filter_id=FILTER_DAY_AHEAD_PRICE,
        region="DE-LU",
        timestamp_ms=SAMPLE_EPOCH_MS,
        resolution="hour",
        raw_dir=RAW_DATA_DIR,
    )

    # Verify preserved raw file exists and is valid JSON
    saved_file = RAW_DATA_DIR / f"{FILTER_DAY_AHEAD_PRICE}_DE-LU_hour_{SAMPLE_EPOCH_MS}.json"
    assert saved_file.exists(), f"Raw file {saved_file} was not preserved"
    with open(saved_file, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert "series" in reloaded
    assert len(reloaded["series"]) == 168

    # 2. Parse into in-memory standardized DataFrame
    df = parse_smard_series(raw_data, value_column="day_ahead_price_eur_mwh")
    assert len(df) == 168
    assert list(df.columns) == ["timestamp_utc", "timestamp_berlin", "day_ahead_price_eur_mwh"]

    # 3. Run audit against market-time grid
    audit = audit_sample_series(
        df,
        value_column="day_ahead_price_eur_mwh",
        expected_start_berlin=EXPECTED_START_BERLIN,
        expected_end_berlin=EXPECTED_END_BERLIN,
    )

    assert audit["requested_hours"] == 168
    assert audit["returned_observations"] == 168
    assert audit["duplicate_timestamps"] == 0
    assert audit["missing_timestamps"] == 0
    assert audit["null_values"] == 0
    assert audit["first_timestamp_berlin"] == "2023-01-02 00:00:00+01:00"
    assert audit["last_timestamp_berlin"] == "2023-01-08 23:00:00+01:00"
    assert audit["first_timestamp_utc"] == "2023-01-01 23:00:00+00:00"
    assert audit["last_timestamp_utc"] == "2023-01-08 22:00:00+00:00"

    print("\n--- Day-Ahead Price Audit (DE-LU) ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")


def test_smard_sample_load_forecast():
    """Test Day-Ahead total load forecast sample ingestion and schema verification."""
    # 1. Fetch raw chunk and preserve untouched JSON
    raw_data = fetch_smard_chunk(
        filter_id=FILTER_LOAD_FORECAST,
        region="DE",
        timestamp_ms=SAMPLE_EPOCH_MS,
        resolution="hour",
        raw_dir=RAW_DATA_DIR,
    )

    # Verify preserved raw file exists and is valid JSON
    saved_file = RAW_DATA_DIR / f"{FILTER_LOAD_FORECAST}_DE_hour_{SAMPLE_EPOCH_MS}.json"
    assert saved_file.exists(), f"Raw file {saved_file} was not preserved"
    with open(saved_file, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert "series" in reloaded
    assert len(reloaded["series"]) == 168

    # 2. Parse into in-memory standardized DataFrame
    df = parse_smard_series(raw_data, value_column="load_forecast_mw")
    assert len(df) == 168
    assert list(df.columns) == ["timestamp_utc", "timestamp_berlin", "load_forecast_mw"]

    # 3. Run audit against market-time grid
    audit = audit_sample_series(
        df,
        value_column="load_forecast_mw",
        expected_start_berlin=EXPECTED_START_BERLIN,
        expected_end_berlin=EXPECTED_END_BERLIN,
    )

    assert audit["requested_hours"] == 168
    assert audit["returned_observations"] == 168
    assert audit["duplicate_timestamps"] == 0
    assert audit["missing_timestamps"] == 0
    assert audit["null_values"] == 0
    assert audit["first_timestamp_berlin"] == "2023-01-02 00:00:00+01:00"
    assert audit["last_timestamp_berlin"] == "2023-01-08 23:00:00+01:00"
    assert audit["first_timestamp_utc"] == "2023-01-01 23:00:00+00:00"
    assert audit["last_timestamp_utc"] == "2023-01-08 22:00:00+00:00"

    print("\n--- Load Forecast Audit (DE) ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")


def test_smard_sample_load_actual():
    """Test Day-Ahead actual total electricity consumption sample ingestion and schema verification."""
    # 1. Fetch raw chunk and preserve untouched JSON
    raw_data = fetch_smard_chunk(
        filter_id=FILTER_LOAD_ACTUAL,
        region="DE",
        timestamp_ms=SAMPLE_EPOCH_MS,
        resolution="hour",
        raw_dir=RAW_DATA_DIR,
    )

    # Verify preserved raw file exists and is valid JSON
    saved_file = RAW_DATA_DIR / f"{FILTER_LOAD_ACTUAL}_DE_hour_{SAMPLE_EPOCH_MS}.json"
    assert saved_file.exists(), f"Raw file {saved_file} was not preserved"
    with open(saved_file, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert "series" in reloaded
    assert len(reloaded["series"]) == 168

    # 2. Parse into in-memory standardized DataFrame
    df = parse_smard_series(raw_data, value_column="load_actual_mw")
    assert len(df) == 168
    assert list(df.columns) == ["timestamp_utc", "timestamp_berlin", "load_actual_mw"]

    # 3. Run audit against market-time grid
    audit = audit_sample_series(
        df,
        value_column="load_actual_mw",
        expected_start_berlin=EXPECTED_START_BERLIN,
        expected_end_berlin=EXPECTED_END_BERLIN,
    )

    assert audit["requested_hours"] == 168
    assert audit["returned_observations"] == 168
    assert audit["duplicate_timestamps"] == 0
    assert audit["missing_timestamps"] == 0
    assert audit["null_values"] == 0
    assert audit["first_timestamp_berlin"] == "2023-01-02 00:00:00+01:00"
    assert audit["last_timestamp_berlin"] == "2023-01-08 23:00:00+01:00"
    assert audit["first_timestamp_utc"] == "2023-01-01 23:00:00+00:00"
    assert audit["last_timestamp_utc"] == "2023-01-08 22:00:00+00:00"

    print("\n--- Actual Load Audit (DE) ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")
