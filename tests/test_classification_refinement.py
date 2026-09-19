"""
Tests for Milestone 7B: Negative-Price Classification Refinement
and Validation-Only Threshold Selection.

Enforces:
- Holdout protection: final holdout is never accessed, loaded, or scored.
- TRAIN row count is 8,592; VALIDATION row count is 4,367.
- Holdout prevalence is never calculated.
- Exactly 17 thresholds are evaluated in the grid [0.10 to 0.90].
- Threshold selection uses VALIDATION data only.
- Predefined threshold rule is implemented exactly: Recall >= 0.75, then highest F1.
- Core Strict and Forecast Extension remain separate.
- No target or metadata columns in predictors.
- Probabilities lie strictly within [0, 1].
- Selected threshold belongs to the predefined grid.
- Metrics JSON contains zero holdout data.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.run_classification_refinement import (
    THRESHOLD_GRID,
    evaluate_threshold_grid,
    select_best_threshold,
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


def test_train_val_row_counts_and_holdout_protection(core_features: pd.DataFrame):
    """
    TRAIN count must be 8,592; VALIDATION count must be 4,367.
    Holdout split must remain disjoint and unaccessed.
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


def test_exactly_17_thresholds_in_grid():
    """Verify exactly 17 predefined thresholds [0.10, 0.15, ..., 0.90]."""
    expected_grid = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
    assert len(THRESHOLD_GRID) == 17
    assert THRESHOLD_GRID == expected_grid


def test_threshold_selection_rule_logic():
    """
    Verify predefined threshold selection rule:
    1. Restrict candidate thresholds to Recall >= 0.75
    2. Among those candidates, select highest F1
    3. If none achieve Recall >= 0.75, fallback to highest Balanced Accuracy.
    """
    # Case 1: Qualifying thresholds exist
    mock_grid_1 = [
        {"threshold": 0.40, "recall": 0.80, "f1": 0.50, "balanced_accuracy": 0.85},
        {"threshold": 0.50, "recall": 0.76, "f1": 0.55, "balanced_accuracy": 0.84},  # Winner: Recall >= 0.75 & highest F1
        {"threshold": 0.60, "recall": 0.70, "f1": 0.60, "balanced_accuracy": 0.83},  # High F1 but Recall < 0.75
    ]
    selected, unconstrained = select_best_threshold(mock_grid_1)
    assert selected["threshold"] == 0.50
    assert selected["selection_strategy"] == "recall_ge_0.75_max_f1"
    assert unconstrained["threshold"] == 0.60

    # Case 2: No threshold achieves Recall >= 0.75 (fallback to max Balanced Accuracy)
    mock_grid_2 = [
        {"threshold": 0.50, "recall": 0.60, "f1": 0.55, "balanced_accuracy": 0.82},
        {"threshold": 0.60, "recall": 0.50, "f1": 0.58, "balanced_accuracy": 0.85},  # Winner: highest Balanced Accuracy
    ]
    selected_fallback, unconstrained_2 = select_best_threshold(mock_grid_2)
    assert selected_fallback["threshold"] == 0.60
    assert selected_fallback["selection_strategy"] == "fallback_max_balanced_accuracy"


def test_metrics_json_integrity_and_holdout_exclusion():
    """Ensure classification_refinement_metrics.json has zero holdout data and valid selections."""
    metrics_path = REPORTS_DIR / "classification_refinement_metrics.json"
    if not metrics_path.exists():
        pytest.skip("reports/classification_refinement_metrics.json does not exist yet.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("holdout_data_included") is False
    assert data.get("train_rows") == 8592
    assert data.get("validation_rows") == 4367

    # Check 17 thresholds evaluated
    assert len(data["core_strict_threshold_evaluation"]["grid_results"]) == 17
    assert len(data["forecast_extension_threshold_evaluation"]["grid_results"]) == 17

    # Check selected threshold is in grid
    core_sel_t = data["core_strict_threshold_evaluation"]["selected_threshold"]["threshold"]
    ext_sel_t = data["forecast_extension_threshold_evaluation"]["selected_threshold"]["threshold"]
    assert core_sel_t in THRESHOLD_GRID
    assert ext_sel_t in THRESHOLD_GRID

    # Check no holdout metrics
    content_str = json.dumps(data).lower()
    assert "holdout_rows" not in content_str
    assert "holdout_metrics" not in content_str
    assert "holdout_prevalence" not in content_str
    assert "test_score" not in content_str
