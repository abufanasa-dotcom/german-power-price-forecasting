"""Unit tests for the Streamlit portfolio dashboard.

Verifies that the dashboard is strictly a presentation layer, loads committed metrics,
preserves frozen decisions and thresholds, and contains no model training or inference logic.
"""

import ast
from pathlib import Path
import pytest

from src.dashboard import (
    get_figure_path,
    get_primary_kpis,
    get_project_root,
    load_final_metrics,
    load_model_freeze,
)


def test_dashboard_module_imports():
    """Verify src.dashboard imports without errors."""
    import src.dashboard as db
    assert hasattr(db, "load_final_metrics")
    assert hasattr(db, "get_primary_kpis")
    assert hasattr(db, "get_figure_path")


def test_final_holdout_metrics_loads():
    """Verify final_holdout_metrics.json loads successfully."""
    metrics = load_final_metrics()
    assert isinstance(metrics, dict)
    assert "primary_regression" in metrics
    assert "primary_classification" in metrics


def test_required_metric_keys_exist():
    """Verify all required top-level and nested keys exist in final_holdout_metrics.json."""
    metrics = load_final_metrics()
    required_keys = [
        "audit_metadata",
        "development_target_statistics",
        "holdout_target_statistics",
        "benchmarks",
        "primary_regression",
        "primary_classification",
        "supplementary_regression",
        "supplementary_classification",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"

    # Verify primary regression metrics
    reg_metrics = metrics["primary_regression"]["holdout_metrics"]
    assert "mae" in reg_metrics
    assert "rmse" in reg_metrics
    assert "median_ae" in reg_metrics
    assert "bias" in reg_metrics
    assert "improvement_vs_24h_pct" in reg_metrics

    # Verify primary classification metrics
    clf_metrics = metrics["primary_classification"]["holdout_metrics"]
    assert "pr_auc" in clf_metrics
    assert "roc_auc" in clf_metrics
    assert "balanced_accuracy" in clf_metrics
    assert "recall" in clf_metrics
    assert "precision" in clf_metrics
    assert "f1" in clf_metrics
    assert "confusion_matrix" in clf_metrics


def test_primary_kpis_extraction():
    """Verify get_primary_kpis extracts and formats the five primary Core Strict KPIs."""
    metrics = load_final_metrics()
    kpis = get_primary_kpis(metrics)

    assert kpis["regression_mae"] == "26.87 EUR/MWh"
    assert kpis["improvement_vs_24h"] == "17.91%"
    assert kpis["negative_recall"] == "84.12%"
    assert kpis["negative_pr_auc"] == "0.4538"
    assert kpis["final_holdout_hours"] == "4,417 hours"


def test_expected_figure_paths_exist():
    """Verify all five expected final holdout figures exist on disk."""
    expected_figures = [
        "final_holdout_regression_benchmark_comparison.png",
        "final_holdout_regression_actual_vs_predicted.png",
        "final_holdout_regression_error_distribution.png",
        "final_holdout_classification_pr_curve.png",
        "final_holdout_classification_confusion_matrix.png",
    ]
    for fig_name in expected_figures:
        path = get_figure_path(fig_name)
        assert path.is_file(), f"Expected figure not found: {path}"
        assert path.stat().st_size > 0, f"Figure file is empty: {path}"


def test_core_strict_is_primary():
    """Verify Core Strict is identified as primary and Forecast Extension is supplementary."""
    metrics = load_final_metrics()

    assert metrics["primary_regression"]["feature_family"] == "Core Strict"
    assert metrics["primary_classification"]["feature_family"] == "Core Strict"

    assert metrics["supplementary_regression"]["status"] == "SUPPLEMENTARY SENSITIVITY ANALYSIS"
    assert "ARCHIVE_VINTAGE_LIMITATION" in metrics["supplementary_regression"]["qualification"]

    assert metrics["supplementary_classification"]["status"] == "SUPPLEMENTARY SENSITIVITY ANALYSIS"


def test_threshold_045_preserved():
    """Verify operating threshold 0.45 is preserved in metrics and freeze files."""
    metrics = load_final_metrics()
    freeze = load_model_freeze()

    assert metrics["audit_metadata"]["frozen_classification_threshold"] == 0.45
    assert metrics["primary_classification"]["operating_threshold"] == 0.45

    primary_clf_freeze = freeze["primary_classification_model"]
    assert primary_clf_freeze["operating_threshold"] == 0.45


def test_dashboard_is_presentation_only():
    """Verify app.py and src/dashboard.py contain no model training, inference, or tuning."""
    root = get_project_root()
    files_to_check = [root / "app.py", root / "src" / "dashboard.py"]

    prohibited_tokens = [
        ".fit(",
        ".fit_transform(",
        "GridSearchCV",
        "RandomizedSearchCV",
        "optuna",
        "hyperopt",
        ".predict(",
        ".predict_proba(",
    ]

    for file_path in files_to_check:
        assert file_path.is_file(), f"File missing: {file_path}"
        code = file_path.read_text(encoding="utf-8")
        for token in prohibited_tokens:
            assert token not in code, (
                f"Prohibited operation '{token}' found in presentation layer {file_path.name}"
            )


def test_app_syntax():
    """Verify app.py parses cleanly into a valid Python AST."""
    root = get_project_root()
    app_file = root / "app.py"
    code = app_file.read_text(encoding="utf-8")
    ast.parse(code, filename="app.py")
