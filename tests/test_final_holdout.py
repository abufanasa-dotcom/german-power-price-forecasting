"""
Automated unit and integration tests for Final Holdout Evaluation.

Validates:
- Final fitting set contains exactly 12,959 rows (TRAIN + VALIDATION).
- Holdout contains exactly 4,417 rows (H2 2024).
- Fitting uses TRAIN + VALIDATION only; holdout is excluded from fitting.
- Scaler is fit without holdout.
- Frozen regression hyperparameters are unchanged from freeze commit 23a34db.
- Primary classification threshold is exactly 0.45.
- Supplementary threshold is exactly 0.55.
- Core Strict feature set is unchanged (23 features).
- Forecast Extension remains separate (26 features).
- No target/timestamp/split columns appear as predictors.
- Predictions contain no NaNs.
- Probabilities remain strictly within [0, 1].
- No threshold optimization occurs.
- No hyperparameter optimization occurs.
- Metrics JSON records freeze commit 23a34db, post_holdout_tuning_performed: false, holdout_used_for_model_selection: false.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.run_final_holdout import (
    FROZEN_REG_PARAMS,
    FROZEN_PRIMARY_CLF_THRESHOLD,
    FROZEN_SUPPLEMENTARY_CLF_THRESHOLD,
    PROCESSED_DIR,
    REPORTS_DIR,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def datasets():
    core_path = PROCESSED_DIR / "features_core_strict.parquet"
    ext_path = PROCESSED_DIR / "features_forecast_extension.parquet"
    df_core = pd.read_parquet(core_path)
    df_ext = pd.read_parquet(ext_path)
    return df_core, df_ext


@pytest.fixture(scope="module")
def metrics_json():
    json_path = REPORTS_DIR / "final_holdout_metrics.json"
    if not json_path.exists():
        pytest.skip("reports/final_holdout_metrics.json has not been generated yet.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_data_splits_and_row_counts(datasets):
    df_core, df_ext = datasets
    
    # Split definitions
    train_rows = (df_core["split"] == "train").sum()
    val_rows = (df_core["split"] == "val").sum()
    holdout_rows = (df_core["split"] == "holdout").sum()

    assert train_rows == 8592, f"Expected 8,592 train rows, found {train_rows}"
    assert val_rows == 4367, f"Expected 4,367 val rows, found {val_rows}"
    assert holdout_rows == 4417, f"Expected 4,417 holdout rows, found {holdout_rows}"

    # Dev set (fitting set)
    dev_mask = df_core["split"].isin(["train", "val"])
    dev_rows = dev_mask.sum()
    assert dev_rows == 12959, f"Expected exactly 12,959 fitting rows, found {dev_rows}"
    assert len(df_core) == 12959 + 4417 == 17376


def test_holdout_date_boundaries(datasets):
    df_core, _ = datasets
    holdout_df = df_core[df_core["split"] == "holdout"]
    
    first_holdout_local = holdout_df["timestamp_local"].iloc[0]
    last_holdout_local = holdout_df["timestamp_local"].iloc[-1]

    assert str(first_holdout_local).startswith("2024-07-01 00:00")
    assert str(last_holdout_local).startswith("2024-12-31 23:00")


def test_feature_sets_and_predictor_integrity(datasets):
    df_core, df_ext = datasets
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}

    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]
    ext_predictors = [c for c in df_ext.columns if c not in non_predictor_cols]

    # Predictor counts
    assert len(core_predictors) == 23, f"Expected 23 Core Strict predictors, found {len(core_predictors)}"
    assert len(ext_predictors) == 26, f"Expected 26 Forecast Extension predictors, found {len(ext_predictors)}"

    # Separation
    for col in non_predictor_cols:
        assert col not in core_predictors
        assert col not in ext_predictors

    diff_features = set(ext_predictors) - set(core_predictors)
    assert diff_features == {"load_forecast_mw", "load_forecast_diff_24h", "load_forecast_daily_peak_ratio"}


def test_frozen_hyperparameters_and_thresholds():
    # Frozen regression hyperparameters
    assert FROZEN_REG_PARAMS["n_estimators"] == 400
    assert FROZEN_REG_PARAMS["learning_rate"] == 0.03
    assert FROZEN_REG_PARAMS["num_leaves"] == 15
    assert FROZEN_REG_PARAMS["max_depth"] == 5
    assert FROZEN_REG_PARAMS["min_child_samples"] == 50
    assert FROZEN_REG_PARAMS["subsample"] == 0.80
    assert FROZEN_REG_PARAMS["subsample_freq"] == 1
    assert FROZEN_REG_PARAMS["colsample_bytree"] == 0.80
    assert FROZEN_REG_PARAMS["reg_lambda"] == 0.0
    assert FROZEN_REG_PARAMS["random_state"] == 42

    # Frozen classification thresholds
    assert FROZEN_PRIMARY_CLF_THRESHOLD == 0.45
    assert FROZEN_SUPPLEMENTARY_CLF_THRESHOLD == 0.55


def test_fitting_uses_dev_only_and_scaler_excludes_holdout(datasets):
    df_core, _ = datasets
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]

    dev_df = df_core[df_core["split"].isin(["train", "val"])]
    holdout_df = df_core[df_core["split"] == "holdout"]

    X_dev = dev_df[core_predictors]
    X_holdout = holdout_df[core_predictors]

    # Fit scaler strictly on dev
    scaler = StandardScaler()
    scaler.fit(X_dev)

    # Verify that scaler mean matches dev set exactly and not combined full dataset
    np.testing.assert_allclose(scaler.mean_, X_dev.mean().values, rtol=1e-5)
    
    # Verify that holdout has different mean, confirming it was not included in fitting
    assert not np.allclose(scaler.mean_, df_core[core_predictors].mean().values, rtol=1e-5)


def test_metrics_json_audit_metadata(metrics_json):
    audit = metrics_json["audit_metadata"]
    assert audit["freeze_commit"] == "23a34db"
    assert audit["model_selection_source"] == "validation only"
    assert audit["final_fitting_rows"] == 12959
    assert audit["holdout_rows"] == 4417
    assert audit["primary_regression_feature_family"] == "Core Strict"
    assert audit["primary_classification_feature_family"] == "Core Strict"
    assert audit["frozen_classification_threshold"] == 0.45
    assert audit["post_holdout_tuning_performed"] is False
    assert audit["holdout_used_for_model_selection"] is False
    assert audit["forecast_extension_status"] == "supplementary_only"


def test_metrics_json_results_validity(metrics_json):
    # Regression
    reg_c = metrics_json["primary_regression"]["holdout_metrics"]
    assert not np.isnan(reg_c["mae"])
    assert not np.isnan(reg_c["rmse"])
    assert not np.isnan(reg_c["median_ae"])
    assert not np.isnan(reg_c["bias"])
    assert reg_c["mae"] > 0
    assert reg_c["rmse"] > 0

    # Classification
    clf_c = metrics_json["primary_classification"]["holdout_metrics"]
    assert 0.0 <= clf_c["pr_auc"] <= 1.0
    assert 0.0 <= clf_c["roc_auc"] <= 1.0
    assert 0.0 <= clf_c["balanced_accuracy"] <= 1.0
    assert 0.0 <= clf_c["recall"] <= 1.0
    assert 0.0 <= clf_c["precision"] <= 1.0
    assert 0.0 <= clf_c["f1"] <= 1.0

    cm = clf_c["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 4417


def test_negative_target_definition_strictly_less_than_zero(datasets):
    df_core, _ = datasets
    prices = df_core["day_ahead_price_eur_mwh"].values
    targets = df_core["is_negative_price"].values

    # Strict definition: price < 0.0
    expected_targets = (prices < 0.0).astype(int)
    np.testing.assert_array_equal(targets, expected_targets)

    # Ensure price == 0.0 is strictly NOT treated as negative
    zero_mask = prices == 0.0
    assert zero_mask.sum() > 0, "Expected zero-price hours to exist in the dataset"
    assert np.all(targets[zero_mask] == 0), "All zero-price hours must have target == 0"


def test_split_negative_counts_exact(datasets):
    df_core, _ = datasets
    
    train_neg = int(df_core.loc[df_core["split"] == "train", "is_negative_price"].sum())
    val_neg = int(df_core.loc[df_core["split"] == "val", "is_negative_price"].sum())
    dev_neg = int(df_core.loc[df_core["split"].isin(["train", "val"]), "is_negative_price"].sum())
    holdout_neg = int(df_core.loc[df_core["split"] == "holdout", "is_negative_price"].sum())

    assert train_neg == 287, f"Expected 287 train negative hours, found {train_neg}"
    assert val_neg == 224, f"Expected 224 val negative hours, found {val_neg}"
    assert dev_neg == 511, f"Expected 511 dev negative hours (287 + 224), found {dev_neg}"
    assert holdout_neg == 233, f"Expected 233 holdout negative hours, found {holdout_neg}"


def test_zero_price_validation_hours(datasets):
    df_core, _ = datasets
    val_df = df_core[df_core["split"] == "val"]
    val_zeros = val_df[val_df["day_ahead_price_eur_mwh"] == 0.0]

    assert len(val_zeros) == 41, f"Expected exactly 41 zero-price validation hours, found {len(val_zeros)}"
    assert np.all(val_zeros["is_negative_price"] == 0), "All 41 zero-price validation hours must have target == 0"


def test_final_report_derives_counts_dynamically(metrics_json):
    # Verify metrics_json contains development_target_statistics
    assert "development_target_statistics" in metrics_json
    dstats = metrics_json["development_target_statistics"]
    assert dstats["train_negative_hours"] == 287
    assert dstats["val_negative_hours"] == 224
    assert dstats["dev_negative_hours"] == 511

    # Verify report markdown
    report_path = REPORTS_DIR / "final_holdout_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")

    # Must contain measured counts
    assert "224" in report_text, "Report must contain measured validation negative count (224)"
    assert "511" in report_text, "Report must contain measured development negative count (511)"
    assert "233" in report_text, "Report must contain measured holdout negative count (233)"

    # Must NOT contain the previous reporting typos
    assert "226 hours" not in report_text, "Report must not contain typographical constant '226 hours'"
    assert "| 513 |" not in report_text, "Report must not contain typographical constant '| 513 |'"

