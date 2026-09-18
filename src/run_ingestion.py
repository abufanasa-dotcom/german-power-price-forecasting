"""
Full Ingestion Pipeline Runner for German Power Market Data (2023-2024).

Fetches, validates, audits, and persists all three core series:
1. Day-Ahead Wholesale Electricity Price (Filter 4169, DE-LU)
2. Total Load Forecast (Filter 411, DE)
3. Actual Total Load (Filter 410, DE)
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd

from src.ingestion import (
    FILTER_DAY_AHEAD_PRICE,
    FILTER_LOAD_FORECAST,
    FILTER_LOAD_ACTUAL,
    PROJECT_START_BERLIN,
    PROJECT_END_BERLIN,
    EXPECTED_TOTAL_HOURS,
    ingest_and_audit_full_series,
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def run_pipeline():
    print("Starting full 2023-2024 ingestion pipeline...")
    print(f"Target window: {PROJECT_START_BERLIN} to {PROJECT_END_BERLIN} (Europe/Berlin)")
    print(f"Expected hours: {EXPECTED_TOTAL_HOURS}\n")

    audits = {}

    # 1. Day-Ahead Wholesale Price
    print("--- Ingesting Day-Ahead Wholesale Electricity Price (4169 DE-LU) ---")
    audits["price"] = ingest_and_audit_full_series(
        filter_id=FILTER_DAY_AHEAD_PRICE,
        region="DE-LU",
        value_column="day_ahead_price_eur_mwh",
        output_filename="day_ahead_price_2023_2024.parquet",
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )
    print(f"Price complete: {audits['price']['returned_hours']} hours processed.")

    # 2. Total Load Forecast
    print("\n--- Ingesting Total Load Forecast (411 DE) ---")
    audits["load_forecast"] = ingest_and_audit_full_series(
        filter_id=FILTER_LOAD_FORECAST,
        region="DE",
        value_column="load_forecast_mw",
        output_filename="load_forecast_2023_2024.parquet",
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )
    print(f"Load forecast complete: {audits['load_forecast']['returned_hours']} hours processed.")

    # 3. Actual Total Load
    print("\n--- Ingesting Actual Total Electricity Consumption (410 DE) ---")
    audits["load_actual"] = ingest_and_audit_full_series(
        filter_id=FILTER_LOAD_ACTUAL,
        region="DE",
        value_column="load_actual_mw",
        output_filename="load_actual_2023_2024.parquet",
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
    )
    print(f"Actual load complete: {audits['load_actual']['returned_hours']} hours processed.")

    # 4. Save Ingestion Manifest
    download_time = datetime.now(timezone.utc).isoformat()
    raw_files_count = len(list(RAW_DIR.glob("*.json")))

    manifest = {
        "metadata": {
            "requested_market_period": f"{PROJECT_START_BERLIN} to {PROJECT_END_BERLIN}",
            "expected_market_hours": EXPECTED_TOTAL_HOURS,
            "download_timestamp_utc": download_time,
            "resolution": "hour",
            "total_raw_json_files": raw_files_count,
        },
        "series": {
            "day_ahead_price": {
                "smard_filter_id": FILTER_DAY_AHEAD_PRICE,
                "region": "DE-LU",
                "processed_parquet": audits["price"]["processed_file"],
                "processed_file_size_bytes": audits["price"]["processed_file_size_bytes"],
                "raw_file_count": audits["price"]["raw_chunks_downloaded"],
                "processed_row_count": audits["price"]["returned_hours"],
                "missing_utc_count": audits["price"]["missing_utc_timestamps"],
                "duplicate_utc_count": audits["price"]["duplicate_utc_timestamps"],
                "null_count": audits["price"]["null_values"],
                "data_qualification": "CLEARED MARKET OUTCOME (TARGET VARIABLE)",
            },
            "load_forecast": {
                "smard_filter_id": FILTER_LOAD_FORECAST,
                "region": "DE",
                "processed_parquet": audits["load_forecast"]["processed_file"],
                "processed_file_size_bytes": audits["load_forecast"]["processed_file_size_bytes"],
                "raw_file_count": audits["load_forecast"]["raw_chunks_downloaded"],
                "processed_row_count": audits["load_forecast"]["returned_hours"],
                "missing_utc_count": audits["load_forecast"]["missing_utc_timestamps"],
                "duplicate_utc_count": audits["load_forecast"]["duplicate_utc_timestamps"],
                "null_count": audits["load_forecast"]["null_values"],
                "data_qualification": "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED",
            },
            "load_actual": {
                "smard_filter_id": FILTER_LOAD_ACTUAL,
                "region": "DE",
                "processed_parquet": audits["load_actual"]["processed_file"],
                "processed_file_size_bytes": audits["load_actual"]["processed_file_size_bytes"],
                "raw_file_count": audits["load_actual"]["raw_chunks_downloaded"],
                "processed_row_count": audits["load_actual"]["returned_hours"],
                "missing_utc_count": audits["load_actual"]["missing_utc_timestamps"],
                "duplicate_utc_count": audits["load_actual"]["duplicate_utc_timestamps"],
                "null_count": audits["load_actual"]["null_values"],
                "data_qualification": "HISTORICAL TELEMETRY (SAFE ONLY WITH LAG >= 48H)",
            },
        },
        "audit_statistics": audits,
    }

    manifest_path = PROCESSED_DIR / "ingestion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nIngestion manifest saved to {manifest_path}")
    return audits


if __name__ == "__main__":
    run_pipeline()
