"""
Minimal SMARD Ingestion Module for German Day-Ahead Electricity Price Forecasting.

Provides targeted API fetching, raw JSON preservation, and safe timestamp parsing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import requests
import pandas as pd

SMARD_BASE_URL = "https://www.smard.de/app/chart_data"

# Standard SMARD filter module IDs
FILTER_DAY_AHEAD_PRICE = 4169  # DE-LU Day-Ahead auction price (EUR/MWh)
FILTER_LOAD_FORECAST = 411      # DE Total load forecast (MWh / average MW)
FILTER_LOAD_ACTUAL = 410        # DE Total actual load (MWh / average MW)


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
    If raw_dir is specified, saves the untouched raw JSON response to preserve source data.
    """
    filename = f"{filter_id}_{region}_{resolution}_{timestamp_ms}.json"
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
    # Expected market-time hourly grid
    expected_grid = pd.date_range(
        expected_start_berlin,
        expected_end_berlin,
        freq="h",
        tz="Europe/Berlin",
    )
    requested_count = len(expected_grid)
    returned_count = len(df)

    # Duplicate timestamps check
    dup_utc = df["timestamp_utc"].duplicated().sum()
    dup_berlin = df["timestamp_berlin"].duplicated().sum()

    # Missing timestamps check
    actual_berlin_set = set(df["timestamp_berlin"])
    expected_berlin_set = set(expected_grid)
    missing_timestamps = expected_berlin_set - actual_berlin_set

    # Null values
    null_count = df[value_column].isna().sum()

    # Numeric summary
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
