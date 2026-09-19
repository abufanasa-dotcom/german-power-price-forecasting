"""
Tests for Milestone 6: First Nonlinear ML Models (LightGBM).

Enforces:
- Holdout protection: holdout data is never passed to model fitting or evaluation.
- Holdout metrics are completely absent from reports/ml_validation_metrics.json.
- Train row count is 8,592; Validation row count is 4,367.
- No targets or metadata columns are included in predictors.
- Core Strict and Forecast Extension feature sets remain separate.
- Model predictions contain no NaNs; classifier probabilities lie in [0, 1].
- Fixed random_state=42 is used for reproducibility.
- Class-imbalance weight is derived strictly from TRAIN split.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.ml_models import (
    DEFAULT_LGBM_REGRESSOR_PARAMS,
    DEFAULT_LGBM_CLASSIFIER_PARAMS,
    compute_train_class_weight,
    build_lgbm_regressor,
    build_lgbm_classifier,
    extract_feature_importance,
)

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


def test_train_and_val_row_counts_and_holdout_exclusion(core_features: pd.DataFrame):
    """
    TRAIN row count must be exactly 8,592.
    VALIDATION row count must be exactly 4,367.
    HOLDOUT split must exist separately (4,417) and must never be merged with train or val.
    """
    train_df = core_features[core_features["split"] == "train"]
    val_df = core_features[core_features["split"] == "val"]
    holdout_df = core_features[core_features["split"] == "holdout"]

    assert len(train_df) == 8592, f"Expected 8,592 train rows, got {len(train_df)}"
    assert len(val_df) == 4367, f"Expected 4,367 val rows, got {len(val_df)}"
    assert len(holdout_df) == 4417, f"Expected 4,417 holdout rows, got {len(holdout_df)}"

    # Ensure splits do not overlap
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    holdout_idx = set(holdout_df.index)

    assert train_idx.isdisjoint(val_idx), "Train and Validation index sets overlap!"
    assert train_idx.isdisjoint(holdout_idx), "Train and Holdout index sets overlap!"
    assert val_idx.isdisjoint(holdout_idx), "Validation and Holdout index sets overlap!"


def test_no_target_or_metadata_in_predictors(core_features: pd.DataFrame, ext_features: pd.DataFrame):
    """
    Ensure predictors do not contain target columns or timestamps / metadata.
    """
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


def test_class_imbalance_weight_derived_strictly_from_train(core_features: pd.DataFrame):
    """
    Verify scale_pos_weight is calculated strictly from the TRAIN split.
    """
    train_df = core_features[core_features["split"] == "train"]
    val_df = core_features[core_features["split"] == "val"]

    train_y = train_df["is_negative_price"]
    val_y = val_df["is_negative_price"]

    train_n_pos = (train_y == 1).sum()
    train_n_neg = (train_y == 0).sum()
    assert train_n_pos == 287, f"Expected 287 positive train samples, got {train_n_pos}"
    assert train_n_neg == 8305, f"Expected 8,305 negative train samples, got {train_n_neg}"

    weight_train = compute_train_class_weight(train_y)
    expected_weight = 8305 / 287
    assert np.isclose(weight_train, expected_weight, rtol=1e-6)

    # Verify that validation prevalence would produce a different weight, ensuring no leak
    val_weight = compute_train_class_weight(val_y)
    assert not np.isclose(weight_train, val_weight), "Training weight matches validation weight unexpectedly!"


def test_fixed_random_state_and_reproducibility():
    """
    Verify random_state=42 is fixed in regressor and classifier builders.
    """
    reg = build_lgbm_regressor()
    assert reg.random_state == 42
    assert reg.n_estimators == 300
    assert reg.learning_rate == 0.05
    assert reg.num_leaves == 31
    assert reg.max_depth == 6

    clf = build_lgbm_classifier(scale_pos_weight=28.93728)
    assert clf.random_state == 42
    assert clf.n_estimators == 300
    assert clf.scale_pos_weight == 28.93728
    assert clf.num_leaves == 31
    assert clf.max_depth == 6


def test_model_predictions_integrity_and_bounds(core_features: pd.DataFrame):
    """
    Fit LightGBM models on a small sample of TRAIN data and evaluate on a small sample of VAL data.
    Verify:
    - No NaNs or Infs in regression or classification predictions.
    - Classifier probabilities are within [0.0, 1.0].
    - Classifier predictions are binary {0, 1}.
    """
    train_full = core_features[core_features["split"] == "train"]
    pos_samples = train_full[train_full["is_negative_price"] == 1].head(50)
    neg_samples = train_full[train_full["is_negative_price"] == 0].head(450)
    train_df = pd.concat([pos_samples, neg_samples]).sort_index()
    val_df = core_features[core_features["split"] == "val"].iloc[:200]

    forbidden = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictors = [c for c in core_features.columns if c not in forbidden]

    X_train = train_df[predictors]
    y_train_price = train_df["day_ahead_price_eur_mwh"]
    y_train_neg = train_df["is_negative_price"]

    X_val = val_df[predictors]

    # Regressor
    reg = build_lgbm_regressor()
    reg.fit(X_train, y_train_price)
    pred_price = reg.predict(X_val)

    assert not np.isnan(pred_price).any(), "Regression predictions contain NaNs!"
    assert not np.isinf(pred_price).any(), "Regression predictions contain Infs!"
    assert len(pred_price) == len(val_df)

    # Classifier
    scale_w = compute_train_class_weight(y_train_neg)
    clf = build_lgbm_classifier(scale_pos_weight=scale_w)
    clf.fit(X_train, y_train_neg)
    proba_neg = clf.predict_proba(X_val)[:, 1]
    pred_neg = (proba_neg >= 0.50).astype(int)

    assert not np.isnan(proba_neg).any(), "Classification probabilities contain NaNs!"
    assert (proba_neg >= 0.0).all() and (proba_neg <= 1.0).all(), "Probabilities outside [0, 1]!"
    assert set(np.unique(pred_neg)).issubset({0, 1}), "Predictions are not binary {0, 1}!"


def test_feature_importance_structure(core_features: pd.DataFrame):
    """
    Verify extract_feature_importance returns a properly structured DataFrame.
    """
    train_df = core_features[core_features["split"] == "train"].iloc[:500]
    forbidden = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    predictors = [c for c in core_features.columns if c not in forbidden]

    X_train = train_df[predictors]
    y_train_price = train_df["day_ahead_price_eur_mwh"]

    reg = build_lgbm_regressor()
    reg.fit(X_train, y_train_price)

    df_imp = extract_feature_importance(reg, predictors)
    assert "feature" in df_imp.columns
    assert "importance_gain" in df_imp.columns
    assert "importance_split" in df_imp.columns
    assert "gain_share_pct" in df_imp.columns
    assert len(df_imp) == len(predictors)
    assert np.isclose(df_imp["gain_share_pct"].sum(), 100.0, rtol=1e-4)


def test_holdout_metrics_absent_from_metrics_json():
    """
    Ensure reports/ml_validation_metrics.json has zero holdout metrics.
    """
    metrics_path = REPORTS_DIR / "ml_validation_metrics.json"
    if not metrics_path.exists():
        pytest.skip("reports/ml_validation_metrics.json does not exist yet.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("holdout_data_included") is False, "holdout_data_included must be False!"
    assert data.get("train_rows") == 8592
    assert data.get("validation_rows") == 4367

    content_str = json.dumps(data).lower()
    assert "holdout_rows" not in content_str
    assert "holdout_metrics" not in content_str
    assert "test_score" not in content_str
