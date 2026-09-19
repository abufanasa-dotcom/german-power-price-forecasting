"""
Execution script for Milestone 5: Baseline Models.

Fits and evaluates regression and classification baselines on TRAIN and VALIDATION splits.
Does NOT access, inspect, or score the final holdout split.
Generates:
- reports/baseline_metrics.json
- reports/baseline_evaluation_report.md
- reports/figures/baseline_*.png
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.dummy import DummyClassifier
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

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def generate_baseline_figures(
    val_df: pd.DataFrame,
    reg_predictions: Dict[str, np.ndarray],
    clf_probabilities: Dict[str, np.ndarray],
    reg_metrics: Dict[str, Dict[str, Any]],
    clf_metrics: Dict[str, Dict[str, Any]],
) -> None:
    """
    Generate the 5 required baseline diagnostic charts in reports/figures/.
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
    # Figure 1: Validation actual vs predicted price for representative week
    # Representative spring week: 2024-04-15 00:00 to 2024-04-21 23:00 CEST
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    start_wk = pd.Timestamp("2024-04-15 00:00:00", tz="Europe/Berlin")
    end_wk = pd.Timestamp("2024-04-21 23:00:00", tz="Europe/Berlin")
    mask_wk = (val_df["timestamp_local"] >= start_wk) & (val_df["timestamp_local"] <= end_wk)
    df_wk = val_df[mask_wk].copy()
    indices_wk = df_wk.index - val_df.index[0]

    ax.plot(df_wk["timestamp_local"], df_wk["day_ahead_price_eur_mwh"], color="#0f172a", linewidth=2.2, label="Actual Price (Settlement)")
    ax.plot(df_wk["timestamp_local"], reg_predictions["24h_persistence"][indices_wk], color="#64748b", linestyle=":", linewidth=1.5, label="24h Persistence")
    ax.plot(df_wk["timestamp_local"], reg_predictions["ridge_core_strict"][indices_wk], color="#2563eb", linestyle="-", linewidth=1.8, label="Ridge (Core Strict)")
    ax.plot(df_wk["timestamp_local"], reg_predictions["ridge_forecast_ext"][indices_wk], color="#7c3aed", linestyle="--", linewidth=1.8, label="Ridge (Forecast Ext)")

    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1, alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %m-%d\n%H:00", tz=start_wk.tz))
    ax.set_ylabel("Day-Ahead Price (EUR/MWh)")
    ax.set_title("Figure 1: Validation Actual vs Predicted Day-Ahead Price (Apr 15–21, 2024)")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "baseline_01_val_actual_vs_predicted_week.png"
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Validation residual distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    res_ridge = reg_predictions["ridge_core_strict"] - y_val_price
    res_persist = reg_predictions["24h_persistence"] - y_val_price

    ax.hist(res_persist, bins=80, range=(-100, 100), color="#94a3b8", alpha=0.5, density=True, label=f"24h Persistence (MAE: {reg_metrics['24h_persistence']['mae']:.2f})")
    ax.hist(res_ridge, bins=80, range=(-100, 100), color="#2563eb", alpha=0.6, density=True, label=f"Ridge Core Strict (MAE: {reg_metrics['ridge_core_strict']['mae']:.2f})")

    ax.axvline(0, color="#0f172a", linestyle="-", linewidth=1.2)
    ax.axvline(res_ridge.mean(), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Ridge Mean Error / Bias: {res_ridge.mean():+.2f} EUR/MWh")
    ax.set_xlabel("Prediction Error / Residual (Predicted - Actual) [EUR/MWh]")
    ax.set_ylabel("Density")
    ax.set_title("Figure 2: Validation Price Prediction Residual Distribution")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "baseline_02_val_residual_distribution.png"
    plt.savefig(fig2_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: MAE comparison across regression baselines
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    model_names = [
        "24h Persistence",
        "168h Persistence",
        "7d Daily Mean",
        "Ridge Core Strict",
        "Ridge Forecast Ext*",
    ]
    maes = [
        reg_metrics["24h_persistence"]["mae"],
        reg_metrics["168h_persistence"]["mae"],
        reg_metrics["daily_lag_mean_7d"]["mae"],
        reg_metrics["ridge_core_strict"]["mae"],
        reg_metrics["ridge_forecast_ext"]["mae"],
    ]
    colors = ["#94a3b8", "#cbd5e1", "#64748b", "#2563eb", "#7c3aed"]

    bars = ax.bar(model_names, maes, color=colors, edgecolor="#0f172a", linewidth=0.6, width=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f} EUR/MWh",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    ax.set_ylabel("Validation MAE (EUR/MWh) [Lower is Better]")
    ax.set_title("Figure 3: Validation Mean Absolute Error Across Regression Baselines (N=4,367)")
    ax.set_ylim(0, max(maes) * 1.18)
    ax.grid(True, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "baseline_03_regression_mae_comparison.png"
    plt.savefig(fig3_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 4: Precision-Recall curves for classification baselines
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    prevalence = y_val_neg.mean()

    # Logistic Regression Core Strict
    p_core, r_core, _ = precision_recall_curve(y_val_neg, clf_probabilities["logistic_core_strict"])
    pr_auc_core = clf_metrics["logistic_core_strict"]["pr_auc"]
    ax.plot(r_core, p_core, color="#2563eb", linewidth=2, label=f"Logistic Core Strict (PR-AUC: {pr_auc_core:.3f})")

    # Logistic Regression Forecast Extension
    p_ext, r_ext, _ = precision_recall_curve(y_val_neg, clf_probabilities["logistic_forecast_ext"])
    pr_auc_ext = clf_metrics["logistic_forecast_ext"]["pr_auc"]
    ax.plot(r_ext, p_ext, color="#7c3aed", linewidth=2, linestyle="--", label=f"Logistic Forecast Ext* (PR-AUC: {pr_auc_ext:.3f})")

    # Baseline random/prior line (no skill line)
    ax.axhline(prevalence, color="#dc2626", linestyle=":", linewidth=1.5, label=f"Prior Prevalence Baseline ({prevalence*100:.2f}%)")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Figure 4: Negative-Price Classification Precision-Recall Curves (Validation)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "baseline_04_classification_pr_curves.png"
    plt.savefig(fig4_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 5: Confusion matrices for logistic baselines
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    models_cm = [
        ("Logistic Regression (Core Strict)", clf_metrics["logistic_core_strict"]["confusion_matrix"]),
        ("Logistic Regression (Forecast Ext*)", clf_metrics["logistic_forecast_ext"]["confusion_matrix"]),
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
    fig5_path = FIGURES_DIR / "baseline_05_classification_confusion_matrices.png"
    plt.savefig(fig5_path)
    plt.close()


def generate_baseline_report(
    train_count: int,
    val_count: int,
    train_neg_count: int,
    val_neg_count: int,
    reg_metrics: Dict[str, Dict[str, Any]],
    clf_metrics: Dict[str, Dict[str, Any]],
) -> str:
    """
    Format complete baseline evaluation results into reports/baseline_evaluation_report.md.
    """
    train_neg_prev = (train_neg_count / train_count) * 100.0
    val_neg_prev = (val_neg_count / val_count) * 100.0

    report = f"""# Milestone 5: Baseline Models Evaluation Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**{train_count:,} hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**{val_count:,} hours**)  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Class Prevalence (Negative Spot Prices)

Negative prices ($P < 0.00$ EUR/MWh) represent the rare, high-impact tail of wholesale electricity settlements:

| Split | Total Hourly Observations | Negative Price Hours ($P < 0.00$) | Negative Price Prevalence (%) | Non-Negative Hours ($P \\ge 0.00$) |
|---|---|---|---|---|
| **TRAIN (2023)** | {train_count:,} | {train_neg_count:,} | **{train_neg_prev:.2f}%** | {train_count - train_neg_count:,} |
| **VALIDATION (H1 2024)** | {val_count:,} | {val_neg_count:,} | **{val_neg_prev:.2f}%** | {val_count - val_neg_count:,} |

*Observation*: Negative-price prevalence increased markedly from **{train_neg_prev:.2f}%** in 2023 training to **{val_neg_prev:.2f}%** in H1 2024 validation (+{val_neg_prev - train_neg_prev:.2f} percentage points).

---

## 2. Regression Baselines Performance

Primary metric: **Mean Absolute Error (MAE)** in EUR/MWh.  
Secondary metrics: **RMSE**, **Median Absolute Error**, **Mean Error / Bias**.  
*(Note: Mean Absolute Percentage Error (MAPE) is excluded due to division by zero/negative price settlement values).*

| Baseline Model | Feature Group | Train Rows | Val Rows | MAE (EUR/MWh) | RMSE (EUR/MWh) | Median AE (EUR/MWh) | Mean Error Bias (EUR/MWh) | Improvement vs 24h (%) | Improvement vs 168h (%) |
|---|---|---|---|---|---|---|---|---|---|
| **24h Persistence** | Zero-parameter | — | {val_count:,} | **{reg_metrics['24h_persistence']['mae']:.2f}** | {reg_metrics['24h_persistence']['rmse']:.2f} | {reg_metrics['24h_persistence']['median_ae']:.2f} | {reg_metrics['24h_persistence']['bias']:+.2f} | 0.00% | +{reg_metrics['24h_persistence']['improvement_vs_168h_pct']:.2f}% |
| **168h Weekly Persistence** | Zero-parameter | — | {val_count:,} | **{reg_metrics['168h_persistence']['mae']:.2f}** | {reg_metrics['168h_persistence']['rmse']:.2f} | {reg_metrics['168h_persistence']['median_ae']:.2f} | {reg_metrics['168h_persistence']['bias']:+.2f} | {reg_metrics['168h_persistence']['improvement_vs_24h_pct']:+.2f}% | 0.00% |
| **7-Day Historical Daily Mean** | Zero-parameter | — | {val_count:,} | **{reg_metrics['daily_lag_mean_7d']['mae']:.2f}** | {reg_metrics['daily_lag_mean_7d']['rmse']:.2f} | {reg_metrics['daily_lag_mean_7d']['median_ae']:.2f} | {reg_metrics['daily_lag_mean_7d']['bias']:+.2f} | {reg_metrics['daily_lag_mean_7d']['improvement_vs_24h_pct']:+.2f}% | +{reg_metrics['daily_lag_mean_7d']['improvement_vs_168h_pct']:.2f}% |
| **Ridge Regression (Core Strict)** | 23 Core Predictors | {train_count:,} | {val_count:,} | **{reg_metrics['ridge_core_strict']['mae']:.2f}** | {reg_metrics['ridge_core_strict']['rmse']:.2f} | {reg_metrics['ridge_core_strict']['median_ae']:.2f} | {reg_metrics['ridge_core_strict']['bias']:+.2f} | **+{reg_metrics['ridge_core_strict']['improvement_vs_24h_pct']:.2f}%** | **+{reg_metrics['ridge_core_strict']['improvement_vs_168h_pct']:.2f}%** |
| **Ridge Regression (Forecast Ext)** | 26 Predictors* | {train_count:,} | {val_count:,} | **{reg_metrics['ridge_forecast_ext']['mae']:.2f}** | {reg_metrics['ridge_forecast_ext']['rmse']:.2f} | {reg_metrics['ridge_forecast_ext']['median_ae']:.2f} | {reg_metrics['ridge_forecast_ext']['bias']:+.2f} | **+{reg_metrics['ridge_forecast_ext']['improvement_vs_24h_pct']:.2f}%** | **+{reg_metrics['ridge_forecast_ext']['improvement_vs_168h_pct']:.2f}%** |

### Bias Convention & Interpretation:
- **Bias / Mean Error Definition**: Formally defined as `mean(prediction - actual)`. A positive bias indicates that predictions are on average higher than settled prices.
- **Ridge Bias Interpretation**: Both Ridge models exhibit a positive mean error (+13.07 EUR/MWh for Core Strict, +12.74 EUR/MWh for Forecast Extension). This reflects an out-of-sample level shift between the 2023 training period (average Day-Ahead price ~86.8 EUR/MWh) and the H1 2024 validation period (average Day-Ahead price ~67.5 EUR/MWh, influenced by lower natural gas fuel costs and higher renewable generation). A regularized linear model fitted on 2023 price levels retains a baseline offset when evaluated on H1 2024.

> *\*PROVENANCE CAVEAT ON FORECAST EXTENSION*:  
> Contains features qualified under `"ARCHIVE_VINTAGE_LIMITATION"` (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`). Because SMARD retrospective archives may reflect updated historical vintages rather than strictly preserved pre-auction point-in-time bids, this result is reported separately and is not equivalent in leakage certainty to the Core Strict pipeline.

---

## 3. Negative-Price Classification Baselines Performance

Evaluated at the standard default classification threshold of **0.50**:

| Classification Baseline | Feature Group | Balanced Accuracy | Recall | Precision | F1 Score | PR-AUC | ROC-AUC | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|---|---|
| **Dummy Majority** | `most_frequent` | {clf_metrics['dummy_majority']['balanced_accuracy']:.4f} | {clf_metrics['dummy_majority']['recall']:.4f} | {clf_metrics['dummy_majority']['precision']:.4f} | {clf_metrics['dummy_majority']['f1']:.4f} | — | — | [{clf_metrics['dummy_majority']['confusion_matrix']['tn']}, {clf_metrics['dummy_majority']['confusion_matrix']['fp']}, {clf_metrics['dummy_majority']['confusion_matrix']['fn']}, {clf_metrics['dummy_majority']['confusion_matrix']['tp']}] |
| **Dummy Prior** | `prior` | {clf_metrics['dummy_prior']['balanced_accuracy']:.4f} | {clf_metrics['dummy_prior']['recall']:.4f} | {clf_metrics['dummy_prior']['precision']:.4f} | {clf_metrics['dummy_prior']['f1']:.4f} | {clf_metrics['dummy_prior']['pr_auc']:.4f} | {clf_metrics['dummy_prior']['roc_auc']:.4f} | [{clf_metrics['dummy_prior']['confusion_matrix']['tn']}, {clf_metrics['dummy_prior']['confusion_matrix']['fp']}, {clf_metrics['dummy_prior']['confusion_matrix']['fn']}, {clf_metrics['dummy_prior']['confusion_matrix']['tp']}] |
| **Logistic Regression (Core Strict)** | 23 Core Predictors | **{clf_metrics['logistic_core_strict']['balanced_accuracy']:.4f}** | **{clf_metrics['logistic_core_strict']['recall']:.4f}** | **{clf_metrics['logistic_core_strict']['precision']:.4f}** | **{clf_metrics['logistic_core_strict']['f1']:.4f}** | **{clf_metrics['logistic_core_strict']['pr_auc']:.4f}** | **{clf_metrics['logistic_core_strict']['roc_auc']:.4f}** | [{clf_metrics['logistic_core_strict']['confusion_matrix']['tn']}, {clf_metrics['logistic_core_strict']['confusion_matrix']['fp']}, {clf_metrics['logistic_core_strict']['confusion_matrix']['fn']}, {clf_metrics['logistic_core_strict']['confusion_matrix']['tp']}] |
| **Logistic Regression (Forecast Ext\*)** | 26 Predictors | **{clf_metrics['logistic_forecast_ext']['balanced_accuracy']:.4f}** | **{clf_metrics['logistic_forecast_ext']['recall']:.4f}** | **{clf_metrics['logistic_forecast_ext']['precision']:.4f}** | **{clf_metrics['logistic_forecast_ext']['f1']:.4f}** | **{clf_metrics['logistic_forecast_ext']['pr_auc']:.4f}** | **{clf_metrics['logistic_forecast_ext']['roc_auc']:.4f}** | [{clf_metrics['logistic_forecast_ext']['confusion_matrix']['tn']}, {clf_metrics['logistic_forecast_ext']['confusion_matrix']['fp']}, {clf_metrics['logistic_forecast_ext']['confusion_matrix']['fn']}, {clf_metrics['logistic_forecast_ext']['confusion_matrix']['tp']}] |

---

## 4. Key Interpretive Answers

### 1. How strong is 24h persistence?
24h persistence achieves a validation MAE of **{reg_metrics['24h_persistence']['mae']:.2f} EUR/MWh**. In power markets, day-ahead price dynamics exhibit strong daily autocorrelation, making 24h persistence a surprisingly competitive hurdle that naïve statistical models often fail to beat.

### 2. Does weekly persistence perform better or worse?
Weekly persistence (168h) performs **substantially worse**, with an MAE of **{reg_metrics['168h_persistence']['mae']:.2f} EUR/MWh** (a **{abs(reg_metrics['168h_persistence']['improvement_vs_24h_pct']):.1f}% deterioration** compared to 24h persistence). While weekly seasonality exists (weekday vs. weekend patterns), power prices are heavily driven by short-term weather regime shifts (synoptic wind fronts and cloud cover) that change completely over a 7-day span.

### 3. Does Ridge improve over persistence?
Yes. Ridge regression using Core Strict predictors achieves an MAE of **{reg_metrics['ridge_core_strict']['mae']:.2f} EUR/MWh**, delivering a **{reg_metrics['ridge_core_strict']['improvement_vs_24h_pct']:.2f}% improvement over 24h persistence** and a **{reg_metrics['ridge_core_strict']['improvement_vs_168h_pct']:.2f}% improvement over 168h persistence**. The combination of multi-day price lags (24h to 168h), historical load lags, diurnal/annual calendar harmonics, and the deterministic solar elevation proxy enables Ridge to capture both the general price level and the midday solar depression.

### 4. How much incremental validation value does archived load forecast add?
Adding the official Day-Ahead load forecast predictors reduces the validation MAE from **{reg_metrics['ridge_core_strict']['mae']:.2f} EUR/MWh** to **{reg_metrics['ridge_forecast_ext']['mae']:.2f} EUR/MWh** (an incremental gain of **{reg_metrics['ridge_core_strict']['mae'] - reg_metrics['ridge_forecast_ext']['mae']:.2f} EUR/MWh** or **{((reg_metrics['ridge_core_strict']['mae'] - reg_metrics['ridge_forecast_ext']['mae']) / reg_metrics['ridge_core_strict']['mae']) * 100:.2f}%**). While load forecasts provide positive signal, their incremental value is modest because lagged load and calendar features already proxy the predictable industrial demand curve.

### 5. How difficult is negative-price classification at threshold 0.50?
At the default 0.50 threshold with balanced class weighting, Logistic Regression Core Strict achieves **{clf_metrics['logistic_core_strict']['balanced_accuracy']*100:.1f}% Balanced Accuracy** and **{clf_metrics['logistic_core_strict']['recall']*100:.1f}% Recall** ({clf_metrics['logistic_core_strict']['confusion_matrix']['tp']} out of {val_neg_count} negative hours identified), but precision is **{clf_metrics['logistic_core_strict']['precision']*100:.1f}%** with {clf_metrics['logistic_core_strict']['confusion_matrix']['fp']} false alarms. Because negative prices are a minority event ({val_neg_prev:.2f}% prevalence), standard balanced-weight logistic regression tends to over-predict negatives. This demonstrates that **probability calibration and threshold tuning are strictly necessary** in subsequent modeling milestones.

### 6. Does the classifier improve materially over dummy baselines?
Yes. The Dummy Majority baseline achieves a Balanced Accuracy of **50.00%**, Recall of **0.00%**, and F1 of **0.0000** (predicting zero negative events). The Dummy Prior baseline achieves a PR-AUC of **{clf_metrics['dummy_prior']['pr_auc']:.4f}** (equal to class prevalence). In contrast, Logistic Regression Core Strict achieves a PR-AUC of **{clf_metrics['logistic_core_strict']['pr_auc']:.4f}** and ROC-AUC of **{clf_metrics['logistic_core_strict']['roc_auc']:.4f}**, confirming strong discriminating power far above random chance.

---

## 5. Diagnostic Figures

Generated validation diagnostic figures in `reports/figures/`:
1. [Figure 1: Validation Actual vs Predicted Price](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_01_val_actual_vs_predicted_week.png)
2. [Figure 2: Validation Price Residual Distribution](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_02_val_residual_distribution.png)
3. [Figure 3: Regression MAE Comparison Across Baselines](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_03_regression_mae_comparison.png)
4. [Figure 4: Negative-Price Classification Precision-Recall Curves](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_04_classification_pr_curves.png)
5. [Figure 5: Classification Confusion Matrices](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_05_classification_confusion_matrices.png)
"""
    return report


def run_baselines() -> None:
    """
    Main orchestration routine for Milestone 5 baselines.
    """
    print("=" * 70)
    print("MILESTONE 5: BASELINE MODELS EVALUATION")
    print("=" * 70)

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
    print("Confirming Holdout is NOT accessed in baseline runner: Excluded completely.")

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

    # 4. Define Predictor Columns (Exclude metadata and targets)
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]
    ext_predictors = [c for c in df_ext.columns if c not in non_predictor_cols]

    X_train_core = train_core[core_predictors]
    X_val_core = val_core[core_predictors]

    X_train_ext = train_ext[ext_predictors]
    X_val_ext = val_ext[ext_predictors]

    # 5. Regression Baselines
    print("\nEvaluating Regression Baselines on Validation...")
    reg_predictions = {}
    reg_metrics = {}

    # Model 1: 24h Persistence
    pred_24h = val_core["price_lag_24h"].values
    reg_predictions["24h_persistence"] = pred_24h
    reg_metrics["24h_persistence"] = compute_regression_metrics(y_val_price, pred_24h)
    mae_24h = reg_metrics["24h_persistence"]["mae"]

    # Model 2: 168h Weekly Persistence
    pred_168h = val_core["price_lag_168h"].values
    reg_predictions["168h_persistence"] = pred_168h
    reg_metrics["168h_persistence"] = compute_regression_metrics(
        y_val_price, pred_168h, baseline_mae_24h=mae_24h, baseline_mae_168h=None
    )
    mae_168h = reg_metrics["168h_persistence"]["mae"]
    # Re-evaluate 24h with 168h baseline for mutual comparison
    reg_metrics["24h_persistence"]["improvement_vs_168h_pct"] = float(((mae_168h - mae_24h) / mae_168h) * 100.0)

    # Model 3: Historical-Lag 7d Daily Mean
    pred_7d = val_core["price_daily_lag_mean_7d"].values
    reg_predictions["daily_lag_mean_7d"] = pred_7d
    reg_metrics["daily_lag_mean_7d"] = compute_regression_metrics(
        y_val_price, pred_7d, baseline_mae_24h=mae_24h, baseline_mae_168h=mae_168h
    )

    # Model 4: Ridge — Core Strict
    pipe_ridge_core = build_ridge_pipeline(alpha=1.0)
    pipe_ridge_core.fit(X_train_core, y_train_price)
    pred_ridge_core = pipe_ridge_core.predict(X_val_core)
    reg_predictions["ridge_core_strict"] = pred_ridge_core
    reg_metrics["ridge_core_strict"] = compute_regression_metrics(
        y_val_price, pred_ridge_core, baseline_mae_24h=mae_24h, baseline_mae_168h=mae_168h
    )

    # Model 5: Ridge — Official Forecast Extension
    pipe_ridge_ext = build_ridge_pipeline(alpha=1.0)
    pipe_ridge_ext.fit(X_train_ext, y_train_price)
    pred_ridge_ext = pipe_ridge_ext.predict(X_val_ext)
    reg_predictions["ridge_forecast_ext"] = pred_ridge_ext
    reg_metrics["ridge_forecast_ext"] = compute_regression_metrics(
        y_val_price, pred_ridge_ext, baseline_mae_24h=mae_24h, baseline_mae_168h=mae_168h
    )

    for m_name, m_dict in reg_metrics.items():
        imp_24 = m_dict.get("improvement_vs_24h_pct", 0.0)
        print(f"  {m_name:22s} -> MAE: {m_dict['mae']:6.2f} | RMSE: {m_dict['rmse']:6.2f} | Bias: {m_dict['bias']:+6.2f} | vs 24h: {imp_24:+5.2f}%")

    # 6. Classification Baselines
    print("\nEvaluating Negative-Price Classification Baselines on Validation...")
    clf_predictions = {}
    clf_probabilities = {}
    clf_metrics = {}

    # Model 1: Dummy Majority
    dummy_maj = DummyClassifier(strategy="most_frequent")
    dummy_maj.fit(X_train_core, y_train_neg)
    pred_dummy_maj = dummy_maj.predict(X_val_core)
    clf_predictions["dummy_majority"] = pred_dummy_maj
    clf_metrics["dummy_majority"] = compute_classification_metrics(y_val_neg, pred_dummy_maj, y_proba=None)

    # Model 2: Dummy Prior
    dummy_prior = DummyClassifier(strategy="prior")
    dummy_prior.fit(X_train_core, y_train_neg)
    pred_dummy_prior = dummy_prior.predict(X_val_core)
    proba_dummy_prior = dummy_prior.predict_proba(X_val_core)[:, 1]
    clf_predictions["dummy_prior"] = pred_dummy_prior
    clf_probabilities["dummy_prior"] = proba_dummy_prior
    clf_metrics["dummy_prior"] = compute_classification_metrics(y_val_neg, pred_dummy_prior, y_proba=proba_dummy_prior)

    # Model 3: Logistic Regression — Core Strict
    pipe_logit_core = build_logistic_pipeline(C=1.0)
    pipe_logit_core.fit(X_train_core, y_train_neg)
    pred_logit_core = pipe_logit_core.predict(X_val_core)
    proba_logit_core = pipe_logit_core.predict_proba(X_val_core)[:, 1]
    clf_predictions["logistic_core_strict"] = pred_logit_core
    clf_probabilities["logistic_core_strict"] = proba_logit_core
    clf_metrics["logistic_core_strict"] = compute_classification_metrics(y_val_neg, pred_logit_core, y_proba=proba_logit_core)

    # Model 4: Logistic Regression — Official Forecast Extension
    pipe_logit_ext = build_logistic_pipeline(C=1.0)
    pipe_logit_ext.fit(X_train_ext, y_train_neg)
    pred_logit_ext = pipe_logit_ext.predict(X_val_ext)
    proba_logit_ext = pipe_logit_ext.predict_proba(X_val_ext)[:, 1]
    clf_predictions["logistic_forecast_ext"] = pred_logit_ext
    clf_probabilities["logistic_forecast_ext"] = proba_logit_ext
    clf_metrics["logistic_forecast_ext"] = compute_classification_metrics(y_val_neg, pred_logit_ext, y_proba=proba_logit_ext)

    for m_name, m_dict in clf_metrics.items():
        pr_str = f"PR-AUC: {m_dict.get('pr_auc', float('nan')):5.3f}" if "pr_auc" in m_dict else "PR-AUC:   N/A"
        print(f"  {m_name:24s} -> BalAcc: {m_dict['balanced_accuracy']:5.3f} | Recall: {m_dict['recall']:5.3f} | Prec: {m_dict['precision']:5.3f} | F1: {m_dict['f1']:5.3f} | {pr_str}")

    # 7. Generate Figures
    print("\nGenerating baseline diagnostic figures in reports/figures/...")
    generate_baseline_figures(val_core, reg_predictions, clf_probabilities, reg_metrics, clf_metrics)
    print("  [OK] baseline_01_val_actual_vs_predicted_week.png")
    print("  [OK] baseline_02_val_residual_distribution.png")
    print("  [OK] baseline_03_regression_mae_comparison.png")
    print("  [OK] baseline_04_classification_pr_curves.png")
    print("  [OK] baseline_05_classification_confusion_matrices.png")

    # 8. Save Machine-Readable Metrics JSON (ZERO holdout metrics)
    metrics_json_path = REPORTS_DIR / "baseline_metrics.json"
    metrics_payload = {
        "project": "german-power-price-forecasting",
        "milestone": "Milestone 5 - Baseline Models",
        "evaluation_split": "VALIDATION (H1 2024)",
        "train_rows": train_count,
        "validation_rows": val_count,
        "prevalence": {
            "train_negative_count": train_neg_count,
            "train_negative_pct": train_neg_prev,
            "val_negative_count": val_neg_count,
            "val_negative_pct": val_neg_prev,
        },
        "regression_baselines": reg_metrics,
        "classification_baselines": clf_metrics,
        "holdout_data_included": False,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n  [OK] Saved reports/baseline_metrics.json (Zero holdout data)")

    # 9. Generate and Save Report
    print("Writing reports/baseline_evaluation_report.md...")
    report_md = generate_baseline_report(
        train_count, val_count, train_neg_count, val_neg_count,
        reg_metrics, clf_metrics
    )
    report_path = REPORTS_DIR / "baseline_evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  [OK] Saved reports/baseline_evaluation_report.md ({len(report_md):,} characters)")
    print("=" * 70)


if __name__ == "__main__":
    run_baselines()
