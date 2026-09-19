"""
Execution script for Milestone 6: First Nonlinear ML Models (LightGBM).

Fits and evaluates LightGBM Regressors and Classifiers on TRAIN and VALIDATION splits.
Does NOT access, inspect, or score the final holdout split.
Generates:
- reports/ml_validation_metrics.json
- reports/ml_validation_report.md
- reports/figures/ml_*.png
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import precision_recall_curve

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.baselines import (
    compute_regression_metrics,
    compute_classification_metrics,
    build_ridge_pipeline,
    build_logistic_pipeline,
)
from src.models.ml_models import (
    DEFAULT_LGBM_REGRESSOR_PARAMS,
    DEFAULT_LGBM_CLASSIFIER_PARAMS,
    compute_train_class_weight,
    build_lgbm_regressor,
    build_lgbm_classifier,
    extract_feature_importance,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_ml_figures(
    val_df: pd.DataFrame,
    reg_predictions: Dict[str, np.ndarray],
    clf_probabilities: Dict[str, np.ndarray],
    reg_metrics: Dict[str, Dict[str, Any]],
    clf_metrics: Dict[str, Dict[str, Any]],
    feat_importances: Dict[str, pd.DataFrame],
) -> None:
    """
    Generate the 7 required ML diagnostic charts in reports/figures/.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    })

    y_val_price = val_df["day_ahead_price_eur_mwh"].values
    y_val_neg = val_df["is_negative_price"].values

    # -------------------------------------------------------------
    # Figure 1: MAE comparison across regression models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    model_labels = [
        "24h Persistence\n(Baseline)",
        "Ridge\nCore Strict",
        "LightGBM\nCore Strict",
        "Ridge\nForecast Ext*",
        "LightGBM\nForecast Ext*",
    ]
    maes = [
        reg_metrics["24h_persistence"]["mae"],
        reg_metrics["ridge_core_strict"]["mae"],
        reg_metrics["lgbm_core_strict"]["mae"],
        reg_metrics["ridge_forecast_ext"]["mae"],
        reg_metrics["lgbm_forecast_ext"]["mae"],
    ]
    colors = ["#94a3b8", "#3b82f6", "#1d4ed8", "#a855f7", "#6d28d9"]

    bars = ax.bar(model_labels, maes, color=colors, edgecolor="#0f172a", linewidth=0.7, width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f} EUR/MWh",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold",
        )
    ax.set_ylabel("Validation MAE (EUR/MWh) [Lower is Better]")
    ax.set_title("Figure 1: Validation MAE Comparison: Persistence vs Ridge vs LightGBM (N=4,367)")
    ax.set_ylim(0, max(maes) * 1.18)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "ml_01_regression_mae_comparison.png"
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Actual vs LightGBM prediction for representative week
    # Representative spring week: 2024-04-15 00:00 to 2024-04-21 23:00 CEST
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    start_wk = pd.Timestamp("2024-04-15 00:00:00", tz="Europe/Berlin")
    end_wk = pd.Timestamp("2024-04-21 23:00:00", tz="Europe/Berlin")
    mask_wk = (val_df["timestamp_local"] >= start_wk) & (val_df["timestamp_local"] <= end_wk)
    df_wk = val_df[mask_wk].copy()
    indices_wk = df_wk.index - val_df.index[0]

    ax.plot(df_wk["timestamp_local"], df_wk["day_ahead_price_eur_mwh"], color="#0f172a", linewidth=2.2, label="Actual Price (Settlement)")
    ax.plot(df_wk["timestamp_local"], reg_predictions["24h_persistence"][indices_wk], color="#94a3b8", linestyle=":", linewidth=1.4, label="24h Persistence")
    ax.plot(df_wk["timestamp_local"], reg_predictions["lgbm_core_strict"][indices_wk], color="#2563eb", linestyle="-", linewidth=1.8, label="LightGBM (Core Strict)")
    ax.plot(df_wk["timestamp_local"], reg_predictions["lgbm_forecast_ext"][indices_wk], color="#7c3aed", linestyle="--", linewidth=1.8, label="LightGBM (Forecast Ext*)")

    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1, alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d\n%H:00", tz=start_wk.tz))
    ax.set_ylabel("Day-Ahead Price (EUR/MWh)")
    ax.set_title("Figure 2: Validation Actual vs LightGBM Predicted Day-Ahead Price (Apr 15–21, 2024)")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "ml_02_val_actual_vs_lgbm_week.png"
    plt.savefig(fig2_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: LightGBM validation residual distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    res_persist = reg_predictions["24h_persistence"] - y_val_price
    res_ridge = reg_predictions["ridge_core_strict"] - y_val_price
    res_lgbm_core = reg_predictions["lgbm_core_strict"] - y_val_price
    res_lgbm_ext = reg_predictions["lgbm_forecast_ext"] - y_val_price

    ax.hist(res_persist, bins=80, range=(-80, 80), color="#94a3b8", alpha=0.35, density=True, label=f"24h Persistence (MAE: {reg_metrics['24h_persistence']['mae']:.2f}, Bias: {res_persist.mean():+.2f})")
    ax.hist(res_ridge, bins=80, range=(-80, 80), color="#3b82f6", alpha=0.35, density=True, label=f"Ridge Core Strict (MAE: {reg_metrics['ridge_core_strict']['mae']:.2f}, Bias: {res_ridge.mean():+.2f})")
    ax.hist(res_lgbm_core, bins=80, range=(-80, 80), color="#1d4ed8", alpha=0.5, density=True, label=f"LightGBM Core Strict (MAE: {reg_metrics['lgbm_core_strict']['mae']:.2f}, Bias: {res_lgbm_core.mean():+.2f})")

    ax.axvline(0, color="#0f172a", linestyle="-", linewidth=1.2)
    ax.axvline(res_lgbm_core.mean(), color="#1d4ed8", linestyle="--", linewidth=1.5, label=f"LightGBM Core Bias: {res_lgbm_core.mean():+.2f} EUR/MWh")
    ax.axvline(res_ridge.mean(), color="#dc2626", linestyle=":", linewidth=1.5, label=f"Ridge Core Bias: {res_ridge.mean():+.2f} EUR/MWh")

    ax.set_xlabel("Prediction Error / Residual (Predicted - Actual) [EUR/MWh]")
    ax.set_ylabel("Density")
    ax.set_title("Figure 3: Validation Price Prediction Residual Distribution")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "ml_03_lgbm_residual_distribution.png"
    plt.savefig(fig3_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Regression Feature Importance (Top 10 features)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    models_reg_imp = [
        ("LightGBM Regressor (Core Strict)", feat_importances["reg_core_strict"], "#1d4ed8"),
        ("LightGBM Regressor (Forecast Ext*)", feat_importances["reg_forecast_ext"], "#6d28d9"),
    ]

    for ax_imp, (title, df_imp, color) in zip(axes, models_reg_imp):
        top10 = df_imp.head(10).iloc[::-1]  # ascending for horizontal bar plot
        ax_imp.barh(top10["feature"], top10["importance_gain"], color=color, alpha=0.85, edgecolor="#0f172a", linewidth=0.6)
        ax_imp.set_xlabel("Total Gain Importance")
        ax_imp.set_title(title)
        ax_imp.grid(True, axis="x")
        # Annotate gain share percentage
        for i, (gain, share) in enumerate(zip(top10["importance_gain"], top10["gain_share_pct"])):
            ax_imp.text(gain + (top10["importance_gain"].max() * 0.01), i, f" {share:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Figure 4: Top 10 Feature Importance (Gain) for LightGBM Regression", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "ml_04_regression_feature_importance.png"
    plt.savefig(fig4_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 5: Precision-Recall Curves: Logistic vs LightGBM
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    prevalence = y_val_neg.mean()

    # Logistic Core Strict
    p_log_core, r_log_core, _ = precision_recall_curve(y_val_neg, clf_probabilities["logistic_core_strict"])
    pr_auc_log_core = clf_metrics["logistic_core_strict"]["pr_auc"]
    ax.plot(r_log_core, p_log_core, color="#94a3b8", linewidth=1.6, linestyle=":", label=f"Logistic Core Strict (PR-AUC: {pr_auc_log_core:.3f})")

    # LightGBM Core Strict
    p_lgb_core, r_lgb_core, _ = precision_recall_curve(y_val_neg, clf_probabilities["lgbm_core_strict"])
    pr_auc_lgb_core = clf_metrics["lgbm_core_strict"]["pr_auc"]
    ax.plot(r_lgb_core, p_lgb_core, color="#1d4ed8", linewidth=2.2, label=f"LightGBM Core Strict (PR-AUC: {pr_auc_lgb_core:.3f})")

    # LightGBM Forecast Extension
    p_lgb_ext, r_lgb_ext, _ = precision_recall_curve(y_val_neg, clf_probabilities["lgbm_forecast_ext"])
    pr_auc_lgb_ext = clf_metrics["lgbm_forecast_ext"]["pr_auc"]
    ax.plot(r_lgb_ext, p_lgb_ext, color="#6d28d9", linewidth=2.2, linestyle="--", label=f"LightGBM Forecast Ext* (PR-AUC: {pr_auc_lgb_ext:.3f})")

    # Baseline random/prior line (no skill line)
    ax.axhline(prevalence, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Prior Prevalence Baseline ({prevalence*100:.2f}%)")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Figure 5: Negative-Price Classification Precision-Recall Curves (Validation)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "ml_05_classification_pr_curves.png"
    plt.savefig(fig5_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 6: LightGBM confusion matrices at threshold 0.50
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    models_cm = [
        ("LightGBM Classifier (Core Strict)\nThreshold = 0.50", clf_metrics["lgbm_core_strict"]["confusion_matrix"]),
        ("LightGBM Classifier (Forecast Ext*)\nThreshold = 0.50", clf_metrics["lgbm_forecast_ext"]["confusion_matrix"]),
    ]

    for ax_cm, (title, cm_dict) in zip(axes, models_cm):
        cm_arr = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ])
        cax = ax_cm.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.7)
        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j, i, f"{cm_arr[i, j]:,}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="#0f172a",
                )
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred Positive (P>=0)", "Pred Negative (P<0)"])
        ax_cm.set_yticklabels(["True Positive (P>=0)", "True Negative (P<0)"])
        ax_cm.set_title(title, pad=15)
        ax_cm.grid(False)

    plt.tight_layout()
    fig6_path = FIGURES_DIR / "ml_06_classification_confusion_matrices.png"
    plt.savefig(fig6_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 7: Classification Feature Importance (Top 10 features)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    models_clf_imp = [
        ("LightGBM Classifier (Core Strict)", feat_importances["clf_core_strict"], "#1d4ed8"),
        ("LightGBM Classifier (Forecast Ext*)", feat_importances["clf_forecast_ext"], "#6d28d9"),
    ]

    for ax_imp, (title, df_imp, color) in zip(axes, models_clf_imp):
        top10 = df_imp.head(10).iloc[::-1]  # ascending for horizontal bar plot
        ax_imp.barh(top10["feature"], top10["importance_gain"], color=color, alpha=0.85, edgecolor="#0f172a", linewidth=0.6)
        ax_imp.set_xlabel("Total Gain Importance")
        ax_imp.set_title(title)
        ax_imp.grid(True, axis="x")
        # Annotate gain share percentage
        for i, (gain, share) in enumerate(zip(top10["importance_gain"], top10["gain_share_pct"])):
            ax_imp.text(gain + (top10["importance_gain"].max() * 0.01), i, f" {share:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Figure 7: Top 10 Feature Importance (Gain) for LightGBM Classification", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig7_path = FIGURES_DIR / "ml_07_classification_feature_importance.png"
    plt.savefig(fig7_path)
    plt.close()


def generate_ml_report(
    lgb_version: str,
    reg_params: Dict[str, Any],
    clf_params: Dict[str, Any],
    scale_pos_weight: float,
    train_count: int,
    val_count: int,
    train_neg_count: int,
    val_neg_count: int,
    reg_metrics: Dict[str, Dict[str, Any]],
    clf_metrics: Dict[str, Dict[str, Any]],
    feat_importances: Dict[str, pd.DataFrame],
) -> str:
    """
    Format complete ML evaluation results into reports/ml_validation_report.md.
    """
    train_neg_prev = (train_neg_count / train_count) * 100.0
    val_neg_prev = (val_neg_count / val_count) * 100.0

    # Format top 10 feature tables
    def format_top10_table(df_imp: pd.DataFrame) -> str:
        rows = []
        for rank, row in enumerate(df_imp.head(10).itertuples(), start=1):
            rows.append(f"| {rank} | `{row.feature}` | {row.importance_gain:,.1f} | {row.gain_share_pct:.2f}% | {row.importance_split:,} |")
        return "\n".join(rows)

    top10_reg_core = format_top10_table(feat_importances["reg_core_strict"])
    top10_reg_ext = format_top10_table(feat_importances["reg_forecast_ext"])
    top10_clf_core = format_top10_table(feat_importances["clf_core_strict"])
    top10_clf_ext = format_top10_table(feat_importances["clf_forecast_ext"])

    report = f"""# Milestone 6: First Nonlinear ML Models Validation Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**{train_count:,} hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**{val_count:,} hours**)  
**LightGBM Version**: `{lgb_version}`  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Executive Summary & Model Configurations

In Milestone 6, we introduce nonlinear tree-based gradient boosting models using LightGBM (`lightgbm {lgb_version}`) to capture complex, non-linear merit-order dynamics, solar midday depressions, and load-price elasticities.

### Exact Fixed Model Configurations (No Hyperparameter Search)

To prevent subtle test leakage and overfitting, fixed conservative hyperparameters were specified a priori:

```python
# LightGBM Regressor Configuration
DEFAULT_LGBM_REGRESSOR_PARAMS = {{
    "objective": "regression",
    "n_estimators": {reg_params['n_estimators']},
    "learning_rate": {reg_params['learning_rate']},
    "num_leaves": {reg_params['num_leaves']},
    "max_depth": {reg_params['max_depth']},
    "min_child_samples": {reg_params['min_child_samples']},
    "subsample": {reg_params['subsample']},
    "subsample_freq": {reg_params['subsample_freq']},
    "colsample_bytree": {reg_params['colsample_bytree']},
    "random_state": {reg_params['random_state']},
    "n_jobs": -1,
    "verbose": -1,
}}

# LightGBM Classifier Configuration
DEFAULT_LGBM_CLASSIFIER_PARAMS = {{
    "objective": "binary",
    "scale_pos_weight": {scale_pos_weight:.5f},  # (8,592 - 287) / 287 = 8,305 / 287
    "n_estimators": {clf_params['n_estimators']},
    "learning_rate": {clf_params['learning_rate']},
    "num_leaves": {clf_params['num_leaves']},
    "max_depth": {clf_params['max_depth']},
    "min_child_samples": {clf_params['min_child_samples']},
    "subsample": {clf_params['subsample']},
    "subsample_freq": {clf_params['subsample_freq']},
    "colsample_bytree": {clf_params['colsample_bytree']},
    "random_state": {clf_params['random_state']},
    "n_jobs": -1,
    "verbose": -1,
}}
```

### Training Class-Imbalance Weight Calculation:
The classifier `scale_pos_weight` is derived **strictly from the TRAIN split**:
$$\\text{{scale\\_pos\\_weight}} = \\frac{{N_{{\\text{{class\\_0}}}}}}{{N_{{\\text{{class\\_1}}}}}} = \\frac{{8,592 - 287}}{{287}} = \\frac{{8,305}}{{287}} \\approx {scale_pos_weight:.5f}$$
where:
- class 0 = non-negative price hour ($P \\ge 0.00$ EUR/MWh)
- class 1 = negative-price hour ($P < 0.00$ EUR/MWh)

*Validation split prevalence was strictly NOT used to determine training weights.*

---

## 2. Regression Validation Performance

Primary metric: **Mean Absolute Error (MAE)** in EUR/MWh.  
Secondary metrics: **RMSE**, **Median Absolute Error**, **Mean Error / Bias**.  
*Bias definition*: Formally defined as `mean(prediction - actual)`. A positive bias indicates over-predicting relative to settled spot prices.

| Regression Model | Feature Group | Train Rows | Val Rows | MAE (EUR/MWh) | RMSE (EUR/MWh) | Median AE (EUR/MWh) | Bias (EUR/MWh) | Improvement vs 24h (%) | Improvement vs Ridge (%) |
|---|---|---|---|---|---|---|---|---|---|
| **24h Persistence** | Zero-parameter | — | {val_count:,} | **{reg_metrics['24h_persistence']['mae']:.2f}** | {reg_metrics['24h_persistence']['rmse']:.2f} | {reg_metrics['24h_persistence']['median_ae']:.2f} | {reg_metrics['24h_persistence']['bias']:+.2f} | 0.00% | — |
| **Ridge (Core Strict)** | 23 Core Predictors | {train_count:,} | {val_count:,} | **{reg_metrics['ridge_core_strict']['mae']:.2f}** | {reg_metrics['ridge_core_strict']['rmse']:.2f} | {reg_metrics['ridge_core_strict']['median_ae']:.2f} | {reg_metrics['ridge_core_strict']['bias']:+.2f} | +{reg_metrics['ridge_core_strict']['improvement_vs_24h_pct']:.2f}% | 0.00% |
| **LightGBM (Core Strict)** | 23 Core Predictors | {train_count:,} | {val_count:,} | **{reg_metrics['lgbm_core_strict']['mae']:.2f}** | {reg_metrics['lgbm_core_strict']['rmse']:.2f} | {reg_metrics['lgbm_core_strict']['median_ae']:.2f} | {reg_metrics['lgbm_core_strict']['bias']:+.2f} | **+{reg_metrics['lgbm_core_strict']['improvement_vs_24h_pct']:.2f}%** | **+{reg_metrics['lgbm_core_strict']['improvement_vs_ridge_pct']:.2f}%** |
| **Ridge (Forecast Ext\*)** | 26 Predictors | {train_count:,} | {val_count:,} | **{reg_metrics['ridge_forecast_ext']['mae']:.2f}** | {reg_metrics['ridge_forecast_ext']['rmse']:.2f} | {reg_metrics['ridge_forecast_ext']['median_ae']:.2f} | {reg_metrics['ridge_forecast_ext']['bias']:+.2f} | +{reg_metrics['ridge_forecast_ext']['improvement_vs_24h_pct']:.2f}% | 0.00% |
| **LightGBM (Forecast Ext\*)** | 26 Predictors | {train_count:,} | {val_count:,} | **{reg_metrics['lgbm_forecast_ext']['mae']:.2f}** | {reg_metrics['lgbm_forecast_ext']['rmse']:.2f} | {reg_metrics['lgbm_forecast_ext']['median_ae']:.2f} | {reg_metrics['lgbm_forecast_ext']['bias']:+.2f} | **+{reg_metrics['lgbm_forecast_ext']['improvement_vs_24h_pct']:.2f}%** | **+{reg_metrics['lgbm_forecast_ext']['improvement_vs_ridge_pct']:.2f}%** |

> *\*PROVENANCE CAVEAT ON FORECAST EXTENSION*:  
> Contains features qualified under `"ARCHIVE_VINTAGE_LIMITATION"` (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`). Because SMARD retrospective archives may reflect updated historical vintages rather than strictly preserved pre-auction point-in-time bids, this result is reported separately and is not equivalent in leakage certainty to the Core Strict pipeline.

---

## 3. Negative-Price Classification Performance

Evaluated at the standard fixed default threshold of **0.50** (threshold optimization is intentionally deferred to Milestone 7):

| Classification Model | Feature Group | Balanced Accuracy | Recall | Precision | F1 Score | PR-AUC | ROC-AUC | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|---|---|
| **Dummy Prior** | `prior` | {clf_metrics['dummy_prior']['balanced_accuracy']:.4f} | {clf_metrics['dummy_prior']['recall']:.4f} | {clf_metrics['dummy_prior']['precision']:.4f} | {clf_metrics['dummy_prior']['f1']:.4f} | {clf_metrics['dummy_prior']['pr_auc']:.4f} | {clf_metrics['dummy_prior']['roc_auc']:.4f} | [{clf_metrics['dummy_prior']['confusion_matrix']['tn']}, {clf_metrics['dummy_prior']['confusion_matrix']['fp']}, {clf_metrics['dummy_prior']['confusion_matrix']['fn']}, {clf_metrics['dummy_prior']['confusion_matrix']['tp']}] |
| **Logistic Regression (Core Strict)** | 23 Core Predictors | {clf_metrics['logistic_core_strict']['balanced_accuracy']:.4f} | {clf_metrics['logistic_core_strict']['recall']:.4f} | {clf_metrics['logistic_core_strict']['precision']:.4f} | {clf_metrics['logistic_core_strict']['f1']:.4f} | {clf_metrics['logistic_core_strict']['pr_auc']:.4f} | {clf_metrics['logistic_core_strict']['roc_auc']:.4f} | [{clf_metrics['logistic_core_strict']['confusion_matrix']['tn']}, {clf_metrics['logistic_core_strict']['confusion_matrix']['fp']}, {clf_metrics['logistic_core_strict']['confusion_matrix']['fn']}, {clf_metrics['logistic_core_strict']['confusion_matrix']['tp']}] |
| **LightGBM (Core Strict)** | 23 Core Predictors | **{clf_metrics['lgbm_core_strict']['balanced_accuracy']:.4f}** | **{clf_metrics['lgbm_core_strict']['recall']:.4f}** | **{clf_metrics['lgbm_core_strict']['precision']:.4f}** | **{clf_metrics['lgbm_core_strict']['f1']:.4f}** | **{clf_metrics['lgbm_core_strict']['pr_auc']:.4f}** | **{clf_metrics['lgbm_core_strict']['roc_auc']:.4f}** | [{clf_metrics['lgbm_core_strict']['confusion_matrix']['tn']}, {clf_metrics['lgbm_core_strict']['confusion_matrix']['fp']}, {clf_metrics['lgbm_core_strict']['confusion_matrix']['fn']}, {clf_metrics['lgbm_core_strict']['confusion_matrix']['tp']}] |
| **Logistic Regression (Forecast Ext\*)** | 26 Predictors | {clf_metrics['logistic_forecast_ext']['balanced_accuracy']:.4f} | {clf_metrics['logistic_forecast_ext']['recall']:.4f} | {clf_metrics['logistic_forecast_ext']['precision']:.4f} | {clf_metrics['logistic_forecast_ext']['f1']:.4f} | {clf_metrics['logistic_forecast_ext']['pr_auc']:.4f} | {clf_metrics['logistic_forecast_ext']['roc_auc']:.4f} | [{clf_metrics['logistic_forecast_ext']['confusion_matrix']['tn']}, {clf_metrics['logistic_forecast_ext']['confusion_matrix']['fp']}, {clf_metrics['logistic_forecast_ext']['confusion_matrix']['fn']}, {clf_metrics['logistic_forecast_ext']['confusion_matrix']['tp']}] |
| **LightGBM (Forecast Ext\*)** | 26 Predictors | **{clf_metrics['lgbm_forecast_ext']['balanced_accuracy']:.4f}** | **{clf_metrics['lgbm_forecast_ext']['recall']:.4f}** | **{clf_metrics['lgbm_forecast_ext']['precision']:.4f}** | **{clf_metrics['lgbm_forecast_ext']['f1']:.4f}** | **{clf_metrics['lgbm_forecast_ext']['pr_auc']:.4f}** | **{clf_metrics['lgbm_forecast_ext']['roc_auc']:.4f}** | [{clf_metrics['lgbm_forecast_ext']['confusion_matrix']['tn']}, {clf_metrics['lgbm_forecast_ext']['confusion_matrix']['fp']}, {clf_metrics['lgbm_forecast_ext']['confusion_matrix']['fn']}, {clf_metrics['lgbm_forecast_ext']['confusion_matrix']['tp']}] |

---

## 4. Top 10 Feature Importances

Feature importances represent model-internal split and gain attribution in the gradient-boosted decision trees.  
**Note**: Feature importance measures empirical predictive utility within the fitted tree ensemble; it **does not imply physical causality**.

### A. LightGBM Regressor — Core Strict (23 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
{top10_reg_core}

### B. LightGBM Regressor — Forecast Extension (26 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
{top10_reg_ext}

### C. LightGBM Classifier — Core Strict (23 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
{top10_clf_core}

### D. LightGBM Classifier — Forecast Extension (26 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
{top10_clf_ext}

---

## 5. Diagnostic Figures

Generated validation diagnostic figures in `reports/figures/`:
1. [Figure 1: Regression MAE Comparison Across Models](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_01_regression_mae_comparison.png)
2. [Figure 2: Validation Actual vs LightGBM Predicted Day-Ahead Price](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_02_val_actual_vs_lgbm_week.png)
3. [Figure 3: LightGBM Validation Price Prediction Residual Distribution](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_03_lgbm_residual_distribution.png)
4. [Figure 4: Top 10 Feature Importance (Gain) for LightGBM Regression](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_04_regression_feature_importance.png)
5. [Figure 5: Negative-Price Classification Precision-Recall Curves](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_05_classification_pr_curves.png)
6. [Figure 6: LightGBM Classification Confusion Matrices at Threshold 0.50](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_06_classification_confusion_matrices.png)
7. [Figure 7: Top 10 Feature Importance (Gain) for LightGBM Classification](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/ml_07_classification_feature_importance.png)
"""
    return report


def run_ml_models() -> None:
    """
    Main orchestration routine for Milestone 6 ML models.
    """
    print("=" * 70)
    print("MILESTONE 6: FIRST NONLINEAR ML MODELS (LIGHTGBM)")
    print("=" * 70)

    lgb_version = lgb.__version__
    print(f"LightGBM version: {lgb_version}")

    # 1. Load feature files
    core_path = PROCESSED_DIR / "features_core_strict.parquet"
    ext_path = PROCESSED_DIR / "features_forecast_extension.parquet"

    print(f"Loading {core_path.name}...")
    df_core = pd.read_parquet(core_path)
    print(f"Loading {ext_path.name}...")
    df_ext = pd.read_parquet(ext_path)

    # 2. Strict split filtering (TRAIN and VALIDATION only; NEVER touch HOLDOUT)
    train_core = df_core[df_core["split"] == "train"].reset_index(drop=True)
    val_core = df_core[df_core["split"] == "val"].reset_index(drop=True)

    train_ext = df_ext[df_ext["split"] == "train"].reset_index(drop=True)
    val_ext = df_ext[df_ext["split"] == "val"].reset_index(drop=True)

    train_count = len(train_core)
    val_count = len(val_core)

    print(f"\nTrain Rows: {train_count:,} | Validation Rows: {val_count:,}")
    print("Confirming Holdout is NOT accessed in ML runner: Excluded completely.")

    # 3. Targets and Prevalences
    y_train_price = train_core["day_ahead_price_eur_mwh"]
    y_val_price = val_core["day_ahead_price_eur_mwh"]

    y_train_neg = train_core["is_negative_price"]
    y_val_neg = val_core["is_negative_price"]

    train_neg_count = int(y_train_neg.sum())
    val_neg_count = int(y_val_neg.sum())

    train_neg_prev = (train_neg_count / train_count) * 100.0
    val_neg_prev = (val_neg_count / val_count) * 100.0

    print(f"\nNegative Price Prevalence:")
    print(f"  TRAIN:      {train_neg_count:,} / {train_count:,} ({train_neg_prev:.2f}%)")
    print(f"  VALIDATION: {val_neg_count:,} / {val_count:,} ({val_neg_prev:.2f}%)")

    # 4. Class-Imbalance Weight derived strictly from TRAIN
    scale_pos_weight = compute_train_class_weight(y_train_neg)
    print(f"\nTraining Class-Imbalance Weight (scale_pos_weight): {scale_pos_weight:.5f}")
    print(f"  Formula: N_class_0 / N_class_1 (class 0: non-negative, class 1: negative-price)")
    print(f"  Calculation: ({train_count} - {train_neg_count}) / {train_neg_count} = {train_count - train_neg_count} / {train_neg_count}")

    # 5. Define Predictor Columns (Exclude metadata and targets)
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]
    ext_predictors = [c for c in df_ext.columns if c not in non_predictor_cols]

    print(f"\nPredictor Columns:")
    print(f"  Core Strict:        {len(core_predictors)} predictors")
    print(f"  Forecast Extension: {len(ext_predictors)} predictors")

    X_train_core = train_core[core_predictors]
    X_val_core = val_core[core_predictors]

    X_train_ext = train_ext[ext_predictors]
    X_val_ext = val_ext[ext_predictors]

    # 6. Baseline Reference Metrics (Persistence and Ridge)
    print("\nComputing Reference Baselines on Validation...")
    reg_predictions = {}
    reg_metrics = {}

    # 24h Persistence
    pred_24h = val_core["price_lag_24h"].values
    reg_predictions["24h_persistence"] = pred_24h
    reg_metrics["24h_persistence"] = compute_regression_metrics(y_val_price, pred_24h)
    mae_24h = reg_metrics["24h_persistence"]["mae"]

    # Ridge Core Strict
    pipe_ridge_core = build_ridge_pipeline(alpha=1.0)
    pipe_ridge_core.fit(X_train_core, y_train_price)
    pred_ridge_core = pipe_ridge_core.predict(X_val_core)
    reg_predictions["ridge_core_strict"] = pred_ridge_core
    reg_metrics["ridge_core_strict"] = compute_regression_metrics(
        y_val_price, pred_ridge_core, baseline_mae_24h=mae_24h
    )
    mae_ridge_core = reg_metrics["ridge_core_strict"]["mae"]

    # Ridge Forecast Extension
    pipe_ridge_ext = build_ridge_pipeline(alpha=1.0)
    pipe_ridge_ext.fit(X_train_ext, y_train_price)
    pred_ridge_ext = pipe_ridge_ext.predict(X_val_ext)
    reg_predictions["ridge_forecast_ext"] = pred_ridge_ext
    reg_metrics["ridge_forecast_ext"] = compute_regression_metrics(
        y_val_price, pred_ridge_ext, baseline_mae_24h=mae_24h
    )
    mae_ridge_ext = reg_metrics["ridge_forecast_ext"]["mae"]

    # 7. LightGBM Regressors
    print("\nFitting and Evaluating LightGBM Regressors...")
    # Model 1: LightGBM Regressor — Core Strict
    lgb_reg_core = build_lgbm_regressor()
    lgb_reg_core.fit(X_train_core, y_train_price)
    pred_lgb_core = lgb_reg_core.predict(X_val_core)
    reg_predictions["lgbm_core_strict"] = pred_lgb_core
    reg_metrics["lgbm_core_strict"] = compute_regression_metrics(
        y_val_price, pred_lgb_core, baseline_mae_24h=mae_24h
    )
    # Add improvement vs corresponding Ridge model
    reg_metrics["lgbm_core_strict"]["improvement_vs_ridge_pct"] = float(
        ((mae_ridge_core - reg_metrics["lgbm_core_strict"]["mae"]) / mae_ridge_core) * 100.0
    )

    # Model 2: LightGBM Regressor — Forecast Extension
    lgb_reg_ext = build_lgbm_regressor()
    lgb_reg_ext.fit(X_train_ext, y_train_price)
    pred_lgb_ext = lgb_reg_ext.predict(X_val_ext)
    reg_predictions["lgbm_forecast_ext"] = pred_lgb_ext
    reg_metrics["lgbm_forecast_ext"] = compute_regression_metrics(
        y_val_price, pred_lgb_ext, baseline_mae_24h=mae_24h
    )
    reg_metrics["lgbm_forecast_ext"]["improvement_vs_ridge_pct"] = float(
        ((mae_ridge_ext - reg_metrics["lgbm_forecast_ext"]["mae"]) / mae_ridge_ext) * 100.0
    )

    for m_name, m_dict in reg_metrics.items():
        imp_24 = m_dict.get("improvement_vs_24h_pct", 0.0)
        imp_ridge = m_dict.get("improvement_vs_ridge_pct", 0.0)
        print(f"  {m_name:22s} -> MAE: {m_dict['mae']:6.2f} | RMSE: {m_dict['rmse']:6.2f} | Bias: {m_dict['bias']:+6.2f} | vs 24h: {imp_24:+5.2f}% | vs Ridge: {imp_ridge:+5.2f}%")

    # 8. Classification Models
    print("\nEvaluating Negative-Price Classifiers on Validation (Threshold 0.50)...")
    clf_predictions = {}
    clf_probabilities = {}
    clf_metrics = {}

    # Baseline: Dummy Prior
    from sklearn.dummy import DummyClassifier
    dummy_prior = DummyClassifier(strategy="prior")
    dummy_prior.fit(X_train_core, y_train_neg)
    pred_dummy_prior = dummy_prior.predict(X_val_core)
    proba_dummy_prior = dummy_prior.predict_proba(X_val_core)[:, 1]
    clf_predictions["dummy_prior"] = pred_dummy_prior
    clf_probabilities["dummy_prior"] = proba_dummy_prior
    clf_metrics["dummy_prior"] = compute_classification_metrics(y_val_neg, pred_dummy_prior, y_proba=proba_dummy_prior)

    # Baseline: Logistic Regression Core Strict
    pipe_logit_core = build_logistic_pipeline(C=1.0)
    pipe_logit_core.fit(X_train_core, y_train_neg)
    pred_logit_core = pipe_logit_core.predict(X_val_core)
    proba_logit_core = pipe_logit_core.predict_proba(X_val_core)[:, 1]
    clf_predictions["logistic_core_strict"] = pred_logit_core
    clf_probabilities["logistic_core_strict"] = proba_logit_core
    clf_metrics["logistic_core_strict"] = compute_classification_metrics(y_val_neg, pred_logit_core, y_proba=proba_logit_core)

    # Baseline: Logistic Regression Forecast Extension
    pipe_logit_ext = build_logistic_pipeline(C=1.0)
    pipe_logit_ext.fit(X_train_ext, y_train_neg)
    pred_logit_ext = pipe_logit_ext.predict(X_val_ext)
    proba_logit_ext = pipe_logit_ext.predict_proba(X_val_ext)[:, 1]
    clf_predictions["logistic_forecast_ext"] = pred_logit_ext
    clf_probabilities["logistic_forecast_ext"] = proba_logit_ext
    clf_metrics["logistic_forecast_ext"] = compute_classification_metrics(y_val_neg, pred_logit_ext, y_proba=proba_logit_ext)

    # Model 1: LightGBM Classifier — Core Strict
    lgb_clf_core = build_lgbm_classifier(scale_pos_weight=scale_pos_weight)
    lgb_clf_core.fit(X_train_core, y_train_neg)
    proba_lgb_core = lgb_clf_core.predict_proba(X_val_core)[:, 1]
    pred_lgb_core = (proba_lgb_core >= 0.50).astype(int)
    clf_predictions["lgbm_core_strict"] = pred_lgb_core
    clf_probabilities["lgbm_core_strict"] = proba_lgb_core
    clf_metrics["lgbm_core_strict"] = compute_classification_metrics(y_val_neg, pred_lgb_core, y_proba=proba_lgb_core)

    # Model 2: LightGBM Classifier — Forecast Extension
    lgb_clf_ext = build_lgbm_classifier(scale_pos_weight=scale_pos_weight)
    lgb_clf_ext.fit(X_train_ext, y_train_neg)
    proba_lgb_ext = lgb_clf_ext.predict_proba(X_val_ext)[:, 1]
    pred_lgb_ext = (proba_lgb_ext >= 0.50).astype(int)
    clf_predictions["lgbm_forecast_ext"] = pred_lgb_ext
    clf_probabilities["lgbm_forecast_ext"] = proba_lgb_ext
    clf_metrics["lgbm_forecast_ext"] = compute_classification_metrics(y_val_neg, pred_lgb_ext, y_proba=proba_lgb_ext)

    for m_name, m_dict in clf_metrics.items():
        pr_str = f"PR-AUC: {m_dict.get('pr_auc', float('nan')):5.3f}" if "pr_auc" in m_dict else "PR-AUC:   N/A"
        roc_str = f"ROC-AUC: {m_dict.get('roc_auc', float('nan')):5.3f}" if "roc_auc" in m_dict else "ROC-AUC:   N/A"
        print(f"  {m_name:24s} -> BalAcc: {m_dict['balanced_accuracy']:5.3f} | Recall: {m_dict['recall']:5.3f} | Prec: {m_dict['precision']:5.3f} | F1: {m_dict['f1']:5.3f} | {pr_str} | {roc_str}")

    # 9. Extract Feature Importances
    print("\nExtracting Feature Importances for LightGBM Models...")
    feat_importances = {
        "reg_core_strict": extract_feature_importance(lgb_reg_core, core_predictors),
        "reg_forecast_ext": extract_feature_importance(lgb_reg_ext, ext_predictors),
        "clf_core_strict": extract_feature_importance(lgb_clf_core, core_predictors),
        "clf_forecast_ext": extract_feature_importance(lgb_clf_ext, ext_predictors),
    }

    # 10. Generate Figures
    print("\nGenerating ML diagnostic figures in reports/figures/...")
    generate_ml_figures(
        val_core, reg_predictions, clf_probabilities,
        reg_metrics, clf_metrics, feat_importances
    )
    print("  [OK] ml_01_regression_mae_comparison.png")
    print("  [OK] ml_02_val_actual_vs_lgbm_week.png")
    print("  [OK] ml_03_lgbm_residual_distribution.png")
    print("  [OK] ml_04_regression_feature_importance.png")
    print("  [OK] ml_05_classification_pr_curves.png")
    print("  [OK] ml_06_classification_confusion_matrices.png")
    print("  [OK] ml_07_classification_feature_importance.png")

    # 11. Save Machine-Readable Metrics JSON (ZERO holdout metrics)
    metrics_json_path = REPORTS_DIR / "ml_validation_metrics.json"
    metrics_payload = {
        "project": "german-power-price-forecasting",
        "milestone": "Milestone 6 - First Nonlinear ML Models",
        "lightgbm_version": lgb_version,
        "evaluation_split": "VALIDATION (H1 2024)",
        "train_rows": train_count,
        "validation_rows": val_count,
        "prevalence": {
            "train_negative_count": train_neg_count,
            "train_negative_pct": train_neg_prev,
            "val_negative_count": val_neg_count,
            "val_negative_pct": val_neg_prev,
        },
        "class_imbalance_weight": {
            "strategy": "scale_pos_weight",
            "formula": "N_class_0 / N_class_1",
            "definition": {
                "class_0": "non-negative price hour (P >= 0)",
                "class_1": "negative-price hour (P < 0)",
            },
            "value": scale_pos_weight,
            "derived_split": "train_only",
        },
        "hyperparameters": {
            "regression": DEFAULT_LGBM_REGRESSOR_PARAMS,
            "classification": {
                **DEFAULT_LGBM_CLASSIFIER_PARAMS,
                "scale_pos_weight": scale_pos_weight,
            },
        },
        "regression_models": reg_metrics,
        "classification_models": clf_metrics,
        "feature_importance_top10": {
            k: v.head(10).to_dict(orient="records") for k, v in feat_importances.items()
        },
        "holdout_data_included": False,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n  [OK] Saved reports/ml_validation_metrics.json (Zero holdout data)")

    # 12. Generate and Save Report
    print("Writing reports/ml_validation_report.md...")
    report_md = generate_ml_report(
        lgb_version, DEFAULT_LGBM_REGRESSOR_PARAMS, DEFAULT_LGBM_CLASSIFIER_PARAMS,
        scale_pos_weight, train_count, val_count, train_neg_count, val_neg_count,
        reg_metrics, clf_metrics, feat_importances
    )
    report_path = REPORTS_DIR / "ml_validation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  [OK] Saved reports/ml_validation_report.md ({len(report_md):,} characters)")
    print("=" * 70)


if __name__ == "__main__":
    run_ml_models()
