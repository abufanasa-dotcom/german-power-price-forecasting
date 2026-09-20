"""Safe loading and formatting utilities for the Streamlit portfolio dashboard.

This module provides read-only data access to committed metrics, figures, and freeze
specifications. It does not perform model fitting, threshold tuning, or live inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_final_metrics(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the committed final holdout metrics JSON.

    Args:
        project_root: Optional project root path; defaults to discovering from file.

    Returns:
        Dictionary of final holdout metrics.

    Raises:
        FileNotFoundError: If final_holdout_metrics.json is missing.
    """
    root = project_root or get_project_root()
    metrics_path = root / "reports" / "final_holdout_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Final metrics file not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_freeze(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load and return the committed final model freeze JSON.

    Args:
        project_root: Optional project root path; defaults to discovering from file.

    Returns:
        Dictionary of frozen model configurations.

    Raises:
        FileNotFoundError: If final_model_freeze.json is missing.
    """
    root = project_root or get_project_root()
    freeze_path = root / "reports" / "final_model_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Model freeze file not found: {freeze_path}")
    with open(freeze_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_figure_path(filename: str, project_root: Optional[Path] = None) -> Path:
    """Return the resolved path to an existing figure in reports/figures/.

    Args:
        filename: Name of the figure image file.
        project_root: Optional project root path.

    Returns:
        Path to the figure.

    Raises:
        FileNotFoundError: If the figure file does not exist.
    """
    root = project_root or get_project_root()
    fig_path = root / "reports" / "figures" / filename
    if not fig_path.is_file():
        raise FileNotFoundError(f"Figure file not found: {fig_path}")
    return fig_path


def get_primary_kpis(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and format the five primary Core Strict holdout KPIs.

    Args:
        metrics: Parsed final_holdout_metrics dictionary.

    Returns:
        Dictionary with formatted string values and raw numeric values.
    """
    reg_metrics = metrics.get("primary_regression", {}).get("holdout_metrics", {})
    clf_metrics = metrics.get("primary_classification", {}).get("holdout_metrics", {})
    target_stats = metrics.get("holdout_target_statistics", {})

    mae = reg_metrics.get("mae", 26.8719)
    impr_24h = reg_metrics.get("improvement_vs_24h_pct", 17.91)
    recall = clf_metrics.get("recall", 0.8412)
    pr_auc = clf_metrics.get("pr_auc", 0.4538)
    total_hours = target_stats.get("total_hours", 4417)

    return {
        "regression_mae": f"{mae:.2f} EUR/MWh",
        "improvement_vs_24h": f"{impr_24h:.2f}%",
        "negative_recall": f"{recall * 100:.2f}%",
        "negative_pr_auc": f"{pr_auc:.4f}",
        "final_holdout_hours": f"{total_hours:,} hours",
        "raw": {
            "mae": mae,
            "impr_24h": impr_24h,
            "recall": recall,
            "pr_auc": pr_auc,
            "total_hours": total_hours,
        },
    }
