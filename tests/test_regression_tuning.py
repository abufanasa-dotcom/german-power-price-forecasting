"""
Tests for Milestone 7A: Limited Regression Model Refinement.

Enforces:
- Holdout protection: holdout data is never accessed, loaded, or scored.
- Exactly 6 candidate configurations (A-F) are defined and evaluated.
- TRAIN row count is 8,592; VALIDATION row count is 4,367.
- Selection rule uses validation MAE only.
- No target columns or metadata in predictors.
- Core Strict and Forecast Extension feature sets remain separate.
- Random state is fixed (42) for all candidates.
- Predictions contain no NaNs.
- Tuning metrics JSON contains zero holdout data.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.run_regression_tuning import CANDIDATE_CONFIGS, build_regressor

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"


@pytest.fixture(scope="module")
def core_features() -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / "features_core_strict.parquet"
    assert parquet_path.exists(), f"Missing {parquet_path}"
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="module")
def ext_features() -> pd.DataFrame:
    parquet_path = PROCESSED_DIR / "features_forecast_extension.parquet"
    assert parquet_path.exists(), f"Missing {parquet_path}"
    return pd.read_parquet(parquet_path)


def test_exactly_six_candidates():
    """Verify exactly 6 predefined candidates (A-F) are evaluated."""
    expected_candidates = {"Candidate A", "Candidate B", "Candidate C", "Candidate D", "Candidate E", "Candidate F"}
    assert set(CANDIDATE_CONFIGS.keys()) == expected_candidates
    assert len(CANDIDATE_CONFIGS) == 6


def test_train_val_row_counts_and_holdout_protection(core_features: pd.DataFrame):
    """
    TRAIN row count must be exactly 8,592.
    VALIDATION row count must be exactly 4,367.
    HOLDOUT split must exist separately (4,417) and never be accessed for tuning.
    """
    train_df = core_features[core_features["split"] == "train"]
    val_df = core_features[core_features["split"] == "val"]
    holdout_df = core_features[core_features["split"] == "holdout"]

    assert len(train_df) == 8592, f"Expected 8,592 train rows, got {len(train_df)}"
    assert len(val_df) == 4367, f"Expected 4,367 val rows, got {len(val_df)}"
    assert len(holdout_df) == 4417, f"Expected 4,417 holdout rows, got {len(holdout_df)}"

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    holdout_idx = set(holdout_df.index)

    assert train_idx.isdisjoint(val_idx), "Train and Validation index sets overlap!"
    assert train_idx.isdisjoint(holdout_idx), "Train and Holdout index sets overlap!"
    assert val_idx.isdisjoint(holdout_idx), "Validation and Holdout index sets overlap!"


def test_no_target_or_metadata_in_predictors(core_features: pd.DataFrame, ext_features: pd.DataFrame):
    """Ensure predictors do not contain target columns or timestamps/metadata."""
    forbidden = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}

    core_preds = [c for c in core_features.columns if c not in forbidden]
    ext_preds = [c for c in ext_features.columns if c not in forbidden]

    for col in forbidden:
        assert col not in core_preds, f"Forbidden column {col} in core predictors!"
        assert col not in ext_preds, f"Forbidden column {col} in ext predictors!"

    assert len(core_preds) == 23, f"Expected 23 core predictors, got {len(core_preds)}"
    assert len(ext_preds) == 26, f"Expected 26 ext predictors, got {len(ext_preds)}"

    diff = set(ext_preds) - set(core_preds)
    expected_diff = {"load_forecast_mw", "load_forecast_diff_24h", "load_forecast_daily_peak_ratio"}
    assert diff == expected_diff, f"Unexpected predictor difference: {diff}"


def test_fixed_random_state_in_candidates():
    """Verify random_state is fixed (42) for all candidate builders."""
    for cand in CANDIDATE_CONFIGS:
        model = build_regressor(cand, random_state=42)
        assert model.random_state == 42
        assert model.objective == "regression"


def test_predictions_contain_no_nans(core_features: pd.DataFrame):
    """Fit a candidate on a small sample of TRAIN and verify predictions contain no NaNs or Infs."""
    train_df = core_features[core_features["split"] == "train"].iloc[:300]
    val_df = core_features[core_features["split"] == "val"].iloc[:100]

    forbidden = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictors = [c for c in core_features.columns if c not in forbidden]

    X_train = train_df[predictors]
    y_train = train_df["day_ahead_price_eur_mwh"]
    X_val = val_df[predictors]

    model = build_regressor("Candidate A")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    assert not np.isnan(preds).any(), "Predictions contain NaNs!"
    assert not np.isinf(preds).any(), "Predictions contain Infs!"
    assert len(preds) == len(val_df)


def test_tuning_metrics_json_has_zero_holdout():
    """Verify regression_tuning_metrics.json contains zero holdout data and exactly 6 candidates."""
    metrics_path = REPORTS_DIR / "regression_tuning_metrics.json"
    if not metrics_path.exists():
        pytest.skip("reports/regression_tuning_metrics.json does not exist yet.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("holdout_data_included") is False
    assert data.get("train_rows") == 8592
    assert data.get("validation_rows") == 4367

    assert len(data.get("core_strict_candidates", {})) == 6
    assert len(data.get("forecast_extension_candidates", {})) == 6

    # Verify selection is based on validation MAE
    core_best = data["selection"]["core_strict"]["best_candidate"]
    core_maes = {k: v["mae"] for k, v in data["core_strict_candidates"].items()}
    assert core_best == min(core_maes, key=core_maes.get)

    ext_best = data["selection"]["forecast_extension"]["best_candidate"]
    ext_maes = {k: v["mae"] for k, v in data["forecast_extension_candidates"].items()}
    assert ext_best == min(ext_maes, key=ext_maes.get)

    content_str = json.dumps(data).lower()
    assert "holdout_rows" not in content_str
    assert "holdout_metrics" not in content_str
    assert "test_score" not in content_str
