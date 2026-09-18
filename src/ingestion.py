"""
Minimal SMARD Ingestion Module for German Day-Ahead Electricity Price Forecasting.

Provides targeted API fetching, raw JSON preservation, safe timestamp parsing,
and full 2023-2024 dataset construction with data quality auditing.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import requests
import numpy as np
import pandas as pd

SMARD_BASE_URL = "https://www.smard.de/app/chart_data"

# Standard SMARD filter module IDs
FILTER_DAY_AHEAD_PRICE = 4169  # DE-LU Day-Ahead auction price (EUR/MWh)
FILTER_LOAD_FORECAST = 411     # DE Total load forecast (MWh / average MW)
FILTER_LOAD_ACTUAL = 410       # DE Total actual load (MWh / average MW)

# Full project evaluation period in German market time (Europe/Berlin)
PROJECT_START_BERLIN = "2023-01-01 00:00:00+01:00"
PROJECT_END_BERLIN = "2024-12-31 23:00:00+01:00"
EXPECTED_TOTAL_HOURS = 17544  # 8760 (2023) + 8784 (2024 leap year)


def fetch_smard_index(
    filter_id: int,
    region: str,
    resolution: str = "hour",
    timeout: int = 15,
) -> List[int]:
    """
    Fetch the list of available chunk epoch-millisecond timestamps from the SMARD index.
    """
    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("timestamps", [])


def fetch_smard_chunk(
    filter_id: int,
    region: str,
    timestamp_ms: int,
    resolution: str = "hour",
    raw_dir: Optional[Path] = None,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Fetch a single SMARD data chunk.
    If raw file already exists locally, loads it from disk to avoid redundant downloads.
    If raw_dir is specified and file doesn't exist, saves the untouched raw JSON response.
    """
    filename = f"{filter_id}_{region}_{resolution}_{timestamp_ms}.json"
    
    if raw_dir is not None:
        raw_path = Path(raw_dir)
        raw_path.mkdir(parents=True, exist_ok=True)
        local_file = raw_path / filename
        if local_file.exists():
            with open(local_file, "r", encoding="utf-8") as f:
                return json.load(f)

    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filename}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    raw_text = response.text
    data = json.loads(raw_text)

    if raw_dir is not None:
        raw_path = Path(raw_dir)
        raw_path.mkdir(parents=True, exist_ok=True)
        target_file = raw_path / filename
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

    return data


def parse_smard_series(
    chunk_data: Dict[str, Any],
    value_column: str,
) -> pd.DataFrame:
    """
    Parse a SMARD JSON series into an in-memory standardized DataFrame.
    Timestamps are converted safely to UTC and localized Europe/Berlin market time.
    """
    series = chunk_data.get("series", [])
    if not series:
        return pd.DataFrame(columns=["timestamp_utc", "timestamp_berlin", value_column])

    records = []
    for entry in series:
        if len(entry) >= 2:
            ts_ms, val = entry[0], entry[1]
            records.append({"epoch_ms": ts_ms, value_column: val})

    df = pd.DataFrame(records)
    # Safe UTC conversion from epoch milliseconds
    df["timestamp_utc"] = pd.to_datetime(df["epoch_ms"], unit="ms", utc=True)
    # Convert to German electricity market time (Europe/Berlin)
    df["timestamp_berlin"] = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")

    # Order columns and drop raw epoch
    df = df[["timestamp_utc", "timestamp_berlin", value_column]]
    return df


def audit_sample_series(
    df: pd.DataFrame,
    value_column: str,
    expected_start_berlin: str,
    expected_end_berlin: str,
) -> Dict[str, Any]:
    """
    Audit an in-memory standardized series against a defined market delivery window.
    """
    expected_grid = pd.date_range(
        expected_start_berlin,
        expected_end_berlin,
        freq="h",
        tz="Europe/Berlin",
    )
    requested_count = len(expected_grid)
    returned_count = len(df)

    dup_utc = df["timestamp_utc"].duplicated().sum()
    dup_berlin = df["timestamp_berlin"].duplicated().sum()

    actual_berlin_set = set(df["timestamp_berlin"])
    expected_berlin_set = set(expected_grid)
    missing_timestamps = expected_berlin_set - actual_berlin_set

    null_count = df[value_column].isna().sum()

    values = df[value_column].dropna()
    min_val = float(values.min()) if not values.empty else None
    max_val = float(values.max()) if not values.empty else None
    mean_val = float(values.mean()) if not values.empty else None

    first_utc = str(df["timestamp_utc"].min()) if not df.empty else None
    last_utc = str(df["timestamp_utc"].max()) if not df.empty else None
    first_berlin = str(df["timestamp_berlin"].min()) if not df.empty else None
    last_berlin = str(df["timestamp_berlin"].max()) if not df.empty else None

    return {
        "value_column": value_column,
        "requested_hours": requested_count,
        "returned_observations": returned_count,
        "duplicate_timestamps": int(max(dup_utc, dup_berlin)),
        "missing_timestamps": len(missing_timestamps),
        "null_values": int(null_count),
        "first_timestamp_berlin": first_berlin,
        "last_timestamp_berlin": last_berlin,
        "first_timestamp_utc": first_utc,
        "last_timestamp_utc": last_utc,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
    }


def get_needed_chunk_timestamps(
    all_index_ts: List[int],
    start_berlin: str,
    end_berlin: str,
) -> List[int]:
    """
    Filter SMARD index chunk timestamps to only those overlapping the requested market period.
    """
    start_req = pd.Timestamp(start_berlin)
    end_req = pd.Timestamp(end_berlin)
    needed = []
    for ts_ms in sorted(all_index_ts):
        ts_start = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert("Europe/Berlin")
        ts_end = ts_start + pd.Timedelta(days=7)
        if not (ts_end < start_req or ts_start > end_req):
            needed.append(ts_ms)
    return needed


def ingest_and_audit_full_series(
    filter_id: int,
    region: str,
    value_column: str,
    output_filename: str,
    start_berlin: str = PROJECT_START_BERLIN,
    end_berlin: str = PROJECT_END_BERLIN,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> Dict[str, Any]:
    """
    Execute end-to-end full series ingestion, preserving raw JSON chunks,
    validating timestamp continuity, generating comprehensive audit statistics,
    and saving typed Parquet output.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Query index and determine overlapping chunks
    all_ts = fetch_smard_index(filter_id=filter_id, region=region)
    needed_chunks = get_needed_chunk_timestamps(all_ts, start_berlin, end_berlin)

    # 2. Fetch all required raw chunks (preserving exact source JSON)
    dfs = []
    for ts_ms in needed_chunks:
        raw_data = fetch_smard_chunk(
            filter_id=filter_id,
            region=region,
            timestamp_ms=ts_ms,
            raw_dir=raw_dir,
        )
        dfs.append(parse_smard_series(raw_data, value_column))

    combined_df = pd.concat(dfs, ignore_index=True)

    # 3. Filter strictly to requested German market delivery window
    filtered_df = combined_df[
        (combined_df["timestamp_berlin"] >= start_berlin) &
        (combined_df["timestamp_berlin"] <= end_berlin)
    ].copy()

    # Sort strictly by UTC
    filtered_df = filtered_df.sort_values("timestamp_utc").reset_index(drop=True)

    # 4. Rigorous audit
    expected_hours = EXPECTED_TOTAL_HOURS
    returned_hours = len(filtered_df)

    # UTC continuity & duplicate check
    dup_utc = int(filtered_df["timestamp_utc"].duplicated().sum())
    expected_utc_grid = pd.date_range(
        pd.Timestamp(start_berlin).tz_convert("UTC"),
        pd.Timestamp(end_berlin).tz_convert("UTC"),
        freq="h",
        tz="UTC",
    )
    actual_utc_set = set(filtered_df["timestamp_utc"])
    expected_utc_set = set(expected_utc_grid)
    missing_utc = len(expected_utc_set - actual_utc_set)

    # Null and non-numeric checks
    null_count = int(filtered_df[value_column].isna().sum())
    non_numeric_count = int((~filtered_df[value_column].apply(lambda x: isinstance(x, (int, float, np.number)))).sum())

    values = filtered_df[value_column].dropna().astype(float)
    min_val = float(values.min()) if not values.empty else None
    max_val = float(values.max()) if not values.empty else None
    mean_val = float(values.mean()) if not values.empty else None
    median_val = float(values.median()) if not values.empty else None
    p1 = float(np.percentile(values, 1)) if not values.empty else None
    p99 = float(np.percentile(values, 99)) if not values.empty else None

    # Series-specific metrics
    audit = {
        "filter_id": filter_id,
        "region": region,
        "value_column": value_column,
        "expected_hours": expected_hours,
        "returned_hours": returned_hours,
        "first_timestamp_berlin": str(filtered_df["timestamp_berlin"].iloc[0]),
        "last_timestamp_berlin": str(filtered_df["timestamp_berlin"].iloc[-1]),
        "first_timestamp_utc": str(filtered_df["timestamp_utc"].iloc[0]),
        "last_timestamp_utc": str(filtered_df["timestamp_utc"].iloc[-1]),
        "missing_utc_timestamps": missing_utc,
        "duplicate_utc_timestamps": dup_utc,
        "null_values": null_count,
        "non_numeric_values": non_numeric_count,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "median": median_val,
        "percentile_1": p1,
        "percentile_99": p99,
        "raw_chunks_downloaded": len(needed_chunks),
    }

    if value_column == "day_ahead_price_eur_mwh":
        audit["negative_price_hours"] = int((values < 0.0).sum())
        audit["zero_price_hours"] = int((values == 0.0).sum())
    else:
        audit["values_le_zero"] = int((values <= 0.0).sum())

    # 5. Save standardized processed Parquet file
    clean_processed_df = filtered_df.rename(columns={"timestamp_berlin": "timestamp_local"})[
        ["timestamp_utc", "timestamp_local", value_column]
    ].copy()
    parquet_path = processed_dir / output_filename
    clean_processed_df.to_parquet(parquet_path, index=False)
    audit["processed_file"] = str(parquet_path)
    audit["processed_file_size_bytes"] = parquet_path.stat().st_size

    return audit
