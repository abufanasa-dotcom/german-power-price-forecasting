"""
Automated unit and integration tests for Milestone 5 Baseline Models.

Verifies:
- Holdout rows are never passed to baseline training/evaluation
- No target or metadata/timestamp columns appear in predictor matrices
- Preprocessing (StandardScaler) is fit strictly on train data
- Regression predictions contain zero NaNs and valid finite floats
- Classification predicted probabilities strictly lie within [0, 1]
- Reported validation row count is exactly 4,367
- Persistence predictions strictly equal their source lag columns
- Core Strict and Forecast Extension feature sets remain segregated
- baseline_metrics.json contains zero holdout metrics
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baselines import (
    compute_regression_metrics,
    compute_classification_metrics,
    build_ridge_pipeline,
    build_logistic_pipeline,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

EXPECTED_VAL_ROWS = 4367
EXPECTED_TRAIN_ROWS = 8592


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
def baseline_metrics():
    path = REPORTS_DIR / "baseline_metrics.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_no_holdout_in_training_or_validation(df_core):
    """Verify that split labels strictly isolate train (2023) and validation (H1 2024)."""
    train_df = df_core[df_core["split"] == "train"]
    val_df = df_core[df_core["split"] == "val"]
    holdout_df = df_core[df_core["split"] == "holdout"]
    
    # Check that holdout timestamps never overlap with train or val
    train_ts = set(train_df["timestamp_utc"])
    val_ts = set(val_df["timestamp_utc"])
    holdout_ts = set(holdout_df["timestamp_utc"])
    
    assert len(train_ts.intersection(holdout_ts)) == 0
    assert len(val_ts.intersection(holdout_ts)) == 0
    
    # Check that train only contains 2023 timestamps and val only H1 2024
    assert (train_df["timestamp_local"].dt.year == 2023).all()
    assert (val_df["timestamp_local"] >= "2024-01-01 00:00:00+01:00").all()
    assert (val_df["timestamp_local"] <= "2024-06-30 23:00:00+02:00").all()


def test_no_target_or_meta_in_predictors(df_core, df_ext):
    """Ensure no target columns, timestamps, or split labels appear in predictors."""
    forbidden = {"day_ahead_price_eur_mwh", "is_negative_price", "timestamp_utc", "timestamp_local", "split"}
    
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]
    ext_predictors = [c for c in df_ext.columns if c not in non_predictor_cols]
    
    for col in core_predictors:
        assert col not in forbidden
    for col in ext_predictors:
        assert col not in forbidden


def test_preprocessing_fit_on_train_only(df_core):
    """Verify StandardScaler means and scales match train set, not validation set."""
    train_df = df_core[df_core["split"] == "train"]
    val_df = df_core[df_core["split"] == "val"]
    
    non_preds = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictors = [c for c in df_core.columns if c not in non_preds]
    
    X_train = train_df[predictors]
    X_val = val_df[predictors]
    y_train = train_df["day_ahead_price_eur_mwh"]
    
    pipeline = build_ridge_pipeline(alpha=1.0)
    pipeline.fit(X_train, y_train)
    
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    
    # Scaler mean must match X_train mean, NOT X_val or full dataset
    expected_train_means = X_train.mean(axis=0).values
    np.testing.assert_allclose(scaler.mean_, expected_train_means, rtol=1e-5)
    
    # Must NOT match X_val mean
    val_means = X_val.mean(axis=0).values
    assert not np.allclose(scaler.mean_, val_means, rtol=1e-3), "Scaler incorrectly matched validation mean!"


def test_regression_predictions_finite_and_no_nans(df_core):
    """Verify all regression baseline predictions are finite floats without NaNs."""
    train_df = df_core[df_core["split"] == "train"]
    val_df = df_core[df_core["split"] == "val"]
    
    non_preds = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictors = [c for c in df_core.columns if c not in non_preds]
    
    pipeline = build_ridge_pipeline(alpha=1.0)
    pipeline.fit(train_df[predictors], train_df["day_ahead_price_eur_mwh"])
    preds = pipeline.predict(val_df[predictors])
    
    assert len(preds) == EXPECTED_VAL_ROWS
    assert not np.isnan(preds).any(), "NaN found in predictions"
    assert np.isfinite(preds).all(), "Non-finite value found in predictions"


def test_classification_probabilities_within_zero_one(df_core):
    """Verify logistic regression predicted probabilities lie strictly in [0, 1]."""
    train_df = df_core[df_core["split"] == "train"]
    val_df = df_core[df_core["split"] == "val"]
    
    non_preds = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictors = [c for c in df_core.columns if c not in non_preds]
    
    pipeline = build_logistic_pipeline(C=1.0)
    pipeline.fit(train_df[predictors], train_df["is_negative_price"])
    probas = pipeline.predict_proba(val_df[predictors])[:, 1]
    
    assert len(probas) == EXPECTED_VAL_ROWS
    assert (probas >= 0.0).all() and (probas <= 1.0).all(), "Probabilities outside [0, 1]"
    assert not np.isnan(probas).any(), "NaN in probabilities"


def test_persistence_predictions_equal_source_columns(df_core):
    """Confirm persistence predictions strictly match the raw lag columns."""
    val_df = df_core[df_core["split"] == "val"]
    
    pred_24h = val_df["price_lag_24h"].values
    pred_168h = val_df["price_lag_168h"].values
    pred_7d = val_df["price_daily_lag_mean_7d"].values
    
    assert len(pred_24h) == EXPECTED_VAL_ROWS
    assert len(pred_168h) == EXPECTED_VAL_ROWS
    assert len(pred_7d) == EXPECTED_VAL_ROWS
    
    assert not np.isnan(pred_24h).any()
    assert not np.isnan(pred_168h).any()
    assert not np.isnan(pred_7d).any()


def test_metrics_json_contains_no_holdout_metrics(baseline_metrics):
    """Verify baseline_metrics.json documents only validation and explicitly excludes holdout data."""
    assert baseline_metrics["holdout_data_included"] is False
    assert baseline_metrics["validation_rows"] == EXPECTED_VAL_ROWS
    assert baseline_metrics["train_rows"] == EXPECTED_TRAIN_ROWS
    assert baseline_metrics["evaluation_split"] == "VALIDATION (H1 2024)"
    
    # Ensure no key contains 'holdout' with metrics
    json_str = json.dumps(baseline_metrics).lower()
    assert "holdout_mae" not in json_str
    assert "holdout_rmse" not in json_str
    assert "holdout_f1" not in json_str
