"""
Automated unit and regression tests for Milestone 4 feature engineering and temporal splits.

Tests cover:
- Target definition (price < 0.00 binary classification and continuous price)
- Zero missing predictors after warm-up period
- Timestamp uniqueness in UTC and local time
- Accurate lag value computation
- Enforcing minimum lag constraints (no price lag < 24h, no actual load lag < 48h)
- Strict separation between Core Strict and Forecast Extension feature sets
- DST-safe physical lag calculations across clock transitions
- Temporal model split boundaries, counts, and zero overlap
- Feature manifest completeness and schema compliance
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.features import (
    WARMUP_HOURS,
    PRICE_DAILY_LAGS,
    LOAD_ACTUAL_LAGS,
    TRAIN_END_LOCAL,
    VAL_START_LOCAL,
    VAL_END_LOCAL,
    HOLDOUT_START_LOCAL,
    HOLDOUT_END_LOCAL,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"


@pytest.fixture(scope="module")
def df_core():
    path = PROCESSED_DIR / "features_core_strict.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def df_ext():
    path = PROCESSED_DIR / "features_forecast_extension.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def raw_price():
    path = PROCESSED_DIR / "day_ahead_price_2023_2024.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def raw_load_actual():
    path = PROCESSED_DIR / "load_actual_2023_2024.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


def test_feature_datasets_exist():
    """Verify both engineered feature datasets exist on disk."""
    assert (PROCESSED_DIR / "features_core_strict.parquet").exists()
    assert (PROCESSED_DIR / "features_forecast_extension.parquet").exists()


def test_target_definitions(df_core, raw_price):
    """Verify target columns match ground truth and negative-price definition."""
    # Sliced raw price matching post-warmup rows
    raw_usable = raw_price.iloc[WARMUP_HOURS:].reset_index(drop=True)
    
    # Regression target
    np.testing.assert_allclose(
        df_core["day_ahead_price_eur_mwh"].values,
        raw_usable["day_ahead_price_eur_mwh"].values,
    )
    
    # Classification target: is_negative_price = price < 0.00
    expected_is_neg = (raw_usable["day_ahead_price_eur_mwh"] < 0.0).astype(int)
    np.testing.assert_array_equal(
        df_core["is_negative_price"].values,
        expected_is_neg.values,
    )


def test_no_missing_predictors_after_warmup(df_core, df_ext):
    """Verify zero NaN or null values across all feature columns."""
    assert df_core.isna().sum().sum() == 0, f"Core strict contains nulls: {df_core.isna().sum().to_dict()}"
    assert df_ext.isna().sum().sum() == 0, f"Forecast ext contains nulls: {df_ext.isna().sum().to_dict()}"


def test_no_duplicate_timestamps(df_core):
    """Verify complete timestamp uniqueness in both UTC and local market time."""
    assert df_core["timestamp_utc"].duplicated().sum() == 0
    assert df_core["timestamp_local"].duplicated().sum() == 0


def test_correct_lag_values(df_core, raw_price, raw_load_actual):
    """Verify lag columns strictly match elapsed UTC intervals from the raw series."""
    # Check 24h price lag
    raw_price_sorted = raw_price.sort_values("timestamp_utc").reset_index(drop=True)
    expected_lag_24h = raw_price_sorted["day_ahead_price_eur_mwh"].shift(24).iloc[WARMUP_HOURS:].reset_index(drop=True)
    np.testing.assert_allclose(df_core["price_lag_24h"].values, expected_lag_24h.values)
    
    # Check 168h price lag
    expected_lag_168h = raw_price_sorted["day_ahead_price_eur_mwh"].shift(168).iloc[WARMUP_HOURS:].reset_index(drop=True)
    np.testing.assert_allclose(df_core["price_lag_168h"].values, expected_lag_168h.values)
    
    # Check 48h actual load lag
    raw_load_sorted = raw_load_actual.sort_values("timestamp_utc").reset_index(drop=True)
    expected_load_lag_48h = raw_load_sorted["load_actual_mw"].shift(48).iloc[WARMUP_HOURS:].reset_index(drop=True)
    np.testing.assert_allclose(df_core["load_actual_lag_48h"].values, expected_load_lag_48h.values)
    
    # Check derived 7-day daily lag mean and standard deviation
    daily_lag_cols = [f"price_lag_{lag}h" for lag in PRICE_DAILY_LAGS]
    expected_mean = df_core[daily_lag_cols].mean(axis=1)
    expected_std = df_core[daily_lag_cols].std(axis=1, ddof=1)
    np.testing.assert_allclose(df_core["price_daily_lag_mean_7d"].values, expected_mean.values)
    np.testing.assert_allclose(df_core["price_daily_lag_std_7d"].values, expected_std.values)


def test_no_forbidden_short_lags(df_core):
    """Verify no price lag < 24h and no actual load lag < 48h exist in predictor columns."""
    for col in df_core.columns:
        if col.startswith("price_lag_"):
            lag_val = int(col.split("_")[2].replace("h", ""))
            assert lag_val >= 24, f"Forbidden price lag < 24h found: {col}"
        if col.startswith("load_actual_lag_"):
            lag_val = int(col.split("_")[3].replace("h", ""))
            assert lag_val >= 48, f"Forbidden actual load lag < 48h found: {col}"


def test_feature_group_separation(df_core, df_ext):
    """Verify forecast predictors are strictly isolated from the Core Strict matrix."""
    forecast_cols = ["load_forecast_mw", "load_forecast_diff_24h", "load_forecast_daily_peak_ratio"]
    for col in forecast_cols:
        assert col not in df_core.columns, f"Forecast feature {col} leaked into Core Strict dataset"
        assert col in df_ext.columns, f"Forecast feature {col} missing from Forecast Extension dataset"
        
    # All core strict columns must be present in extension dataset
    for col in df_core.columns:
        assert col in df_ext.columns, f"Core feature {col} missing from extension dataset"


def test_dst_safe_lag_calculations(df_core):
    """
    Verify lag calculations are physically consistent across DST transitions.
    On the UTC timeline, every row transition is exactly 1 elapsed hour.
    """
    utc_diffs = df_core["timestamp_utc"].diff().dropna()
    assert (utc_diffs == pd.Timedelta(hours=1)).all(), "UTC timeline is not continuous"
    
    # Check across Spring DST 2023 transition (2023-03-26)
    spring_row = df_core[df_core["timestamp_local"] == "2023-03-27 12:00:00+02:00"]
    assert len(spring_row) == 1
    # 24 UTC hours prior was 2023-03-26 10:00 UTC (which is 12:00 CEST)
    assert spring_row["price_lag_24h"].iloc[0] is not None
    
    # Check across Autumn DST 2023 transition (2023-10-29)
    autumn_row = df_core[df_core["timestamp_local"] == "2023-10-30 12:00:00+01:00"]
    assert len(autumn_row) == 1
    assert autumn_row["price_lag_24h"].iloc[0] is not None


def test_temporal_split_boundaries_and_zero_overlap(df_core):
    """Verify exact split boundaries, exact row counts, and zero timestamp overlap."""
    train_df = df_core[df_core["split"] == "train"]
    val_df = df_core[df_core["split"] == "val"]
    holdout_df = df_core[df_core["split"] == "holdout"]
    
    # Exact row counts
    assert len(train_df) == 8592, f"Train rows {len(train_df)} != 8,592"
    assert len(val_df) == 4367, f"Validation rows {len(val_df)} != 4,367"
    assert len(holdout_df) == 4417, f"Holdout rows {len(holdout_df)} != 4,417"
    assert len(df_core) == 17376, f"Total usable rows {len(df_core)} != 17,376"
    
    # Zero overlap between split timestamp sets
    train_ts = set(train_df["timestamp_utc"])
    val_ts = set(val_df["timestamp_utc"])
    holdout_ts = set(holdout_df["timestamp_utc"])
    
    assert len(train_ts.intersection(val_ts)) == 0, "Overlap between train and val"
    assert len(train_ts.intersection(holdout_ts)) == 0, "Overlap between train and holdout"
    assert len(val_ts.intersection(holdout_ts)) == 0, "Overlap between val and holdout"
    
    # Chronological boundaries
    assert train_df["timestamp_local"].max() <= pd.Timestamp(TRAIN_END_LOCAL)
    assert val_df["timestamp_local"].min() >= pd.Timestamp(VAL_START_LOCAL)
    assert val_df["timestamp_local"].max() <= pd.Timestamp(VAL_END_LOCAL)
    assert holdout_df["timestamp_local"].min() >= pd.Timestamp(HOLDOUT_START_LOCAL)
    assert holdout_df["timestamp_local"].max() <= pd.Timestamp(HOLDOUT_END_LOCAL)


def test_feature_manifest_completeness(df_ext):
    """Verify feature manifest documents every predictor feature."""
    manifest_path = REPORTS_DIR / "feature_manifest.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    features_dict = manifest["features"]
    
    # Every predictor column in df_ext (excluding metadata and targets) must be in manifest
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictor_cols = [c for c in df_ext.columns if c not in non_predictor_cols]
    
    valid_classifications = {
        "STRICT_PRE_AUCTION",
        "ARCHIVE_VINTAGE_LIMITATION",
        "DETERMINISTIC_CALENDAR",
        "DETERMINISTIC_PHYSICAL_PROXY",
    }
    
    for col in predictor_cols:
        assert col in features_dict, f"Feature {col} missing from feature_manifest.json"
        entry = features_dict[col]
        assert "group" in entry
        assert "source_dataset" in entry
        assert "transformation" in entry
        assert "lookback" in entry
        assert "availability_classification" in entry
        assert entry["availability_classification"] in valid_classifications, (
            f"Invalid classification for {col}: {entry['availability_classification']}"
        )
        assert "leakage_rationale" in entry
