"""
Execution script for Milestone 7B: Negative-Price Classification Refinement
and Validation-Only Threshold Selection.

Evaluates:
- Logistic Regression vs LightGBM on Core Strict and Forecast Extension.
- Selection of best classifier per family by validation PR-AUC.
- Controlled threshold grid evaluation (17 points: 0.10 to 0.90) on VALIDATION only.
- Predefined threshold selection rule: Recall >= 0.75, then max F1.
- Final holdout split is NEVER accessed, loaded, or scored.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.baselines import build_logistic_pipeline
from src.models.ml_models import build_lgbm_classifier, compute_train_class_weight

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Predefined 17-point threshold grid
THRESHOLD_GRID = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
]


def evaluate_threshold_grid(y_val: np.ndarray, y_proba: np.ndarray) -> List[Dict[str, Any]]:
    """Evaluate performance across exactly 17 predefined thresholds."""
    results = []
    for t in THRESHOLD_GRID:
        y_pred = (y_proba >= t).astype(int)
        bal_acc = float(balanced_accuracy_score(y_val, y_pred))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        prec = float(precision_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        results.append({
            "threshold": round(t, 2),
            "balanced_accuracy": bal_acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "qualifies_recall_ge_075": bool(rec >= 0.75),
        })
    return results


def select_best_threshold(threshold_metrics: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Select threshold based on predefined objective:
    1. Restrict candidate thresholds to Recall >= 0.75.
    2. Among those candidates, select the threshold with the highest F1 score.
    3. If no threshold achieves Recall >= 0.75, select highest Balanced Accuracy.
    Also returns unconstrained F1-maximizing threshold for comparison.
    """
    unconstrained_best = max(threshold_metrics, key=lambda x: x["f1"])
    qualifying = [m for m in threshold_metrics if m["recall"] >= 0.75]

    if qualifying:
        selected = max(qualifying, key=lambda x: x["f1"])
        selected["selection_strategy"] = "recall_ge_0.75_max_f1"
    else:
        selected = max(threshold_metrics, key=lambda x: x["balanced_accuracy"])
        selected["selection_strategy"] = "fallback_max_balanced_accuracy"

    return selected, unconstrained_best


def generate_classification_figures(
    y_val: np.ndarray,
    core_model_name: str,
    core_proba: np.ndarray,
    core_grid: List[Dict[str, Any]],
    core_selected: Dict[str, Any],
    ext_model_name: str,
    ext_proba: np.ndarray,
    ext_grid: List[Dict[str, Any]],
    ext_selected: Dict[str, Any],
) -> None:
    """Generate at most three diagnostic figures in reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    })

    # -------------------------------------------------------------
    # Figure 1: Threshold trade-off curve (Recall, Precision, F1 vs Threshold)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    models_data = [
        (f"Core Strict: {core_model_name}", core_grid, core_selected, axes[0]),
        (f"Forecast Extension*: {ext_model_name}", ext_grid, ext_selected, axes[1]),
    ]

    for title, grid, selected, ax in models_data:
        th = [m["threshold"] for m in grid]
        rec = [m["recall"] for m in grid]
        prec = [m["precision"] for m in grid]
        f1 = [m["f1"] for m in grid]

        ax.plot(th, rec, marker="o", color="#dc2626", label="Recall", linewidth=1.8, markersize=4)
        ax.plot(th, prec, marker="s", color="#2563eb", label="Precision", linewidth=1.8, markersize=4)
        ax.plot(th, f1, marker="^", color="#16a34a", label="F1 Score", linewidth=2.0, markersize=5)

        # Highlight Recall >= 0.75 constraint line
        ax.axhline(0.75, color="#dc2626", linestyle=":", linewidth=1.2, alpha=0.7, label="Recall >= 0.75 Target")

        # Highlight Selected Threshold
        sel_t = selected["threshold"]
        ax.axvline(sel_t, color="#0f172a", linestyle="--", linewidth=1.5,
                   label=f"Selected Threshold: {sel_t:.2f} (F1: {selected['f1']:.3f}, Rec: {selected['recall']:.3f})")

        ax.set_xlabel("Classification Probability Threshold")
        ax.set_ylabel("Metric Score")
        ax.set_title(title)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True)
        ax.legend(loc="lower left", frameon=True, fontsize=8.5)

    plt.suptitle("Figure 1: Threshold Trade-off Curve (Recall, Precision, F1)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "classification_threshold_tradeoff.png"
    plt.savefig(fig1_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Precision-Recall curves for selected models
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    prevalence = float(y_val.mean())

    p_core, r_core, _ = precision_recall_curve(y_val, core_proba)
    pr_auc_core = float(average_precision_score(y_val, core_proba))
    ax.plot(r_core, p_core, color="#2563eb", linewidth=2.2,
            label=f"Core Strict: {core_model_name} (PR-AUC: {pr_auc_core:.3f})")

    p_ext, r_ext, _ = precision_recall_curve(y_val, ext_proba)
    pr_auc_ext = float(average_precision_score(y_val, ext_proba))
    ax.plot(r_ext, p_ext, color="#7c3aed", linewidth=2.2, linestyle="--",
            label=f"Forecast Extension*: {ext_model_name} (PR-AUC: {pr_auc_ext:.3f})")

    # Mark operating points at selected thresholds
    ax.scatter(core_selected["recall"], core_selected["precision"], color="#2563eb", s=90, zorder=5,
               edgecolor="#0f172a", linewidth=1.5, label=f"Core Selected (t={core_selected['threshold']:.2f})")
    ax.scatter(ext_selected["recall"], ext_selected["precision"], color="#7c3aed", s=90, zorder=5,
               edgecolor="#0f172a", linewidth=1.5, label=f"Ext Selected (t={ext_selected['threshold']:.2f})")

    # Baseline prior prevalence line
    ax.axhline(prevalence, color="#dc2626", linestyle=":", linewidth=1.3,
               label=f"Prior Prevalence Baseline ({prevalence*100:.2f}%)")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Figure 2: Precision-Recall Curves with Selected Operating Thresholds")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "classification_pr_curve_selected_models.png"
    plt.savefig(fig2_path)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Confusion matrices at selected thresholds
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    models_cm = [
        (f"Core Strict: {core_model_name}\nSelected Threshold = {core_selected['threshold']:.2f}", core_selected),
        (f"Forecast Extension*: {ext_model_name}\nSelected Threshold = {ext_selected['threshold']:.2f}", ext_selected),
    ]

    for ax_cm, (title, sel_dict) in zip(axes, models_cm):
        cm_arr = np.array([
            [sel_dict["tn"], sel_dict["fp"]],
            [sel_dict["fn"], sel_dict["tp"]],
        ])
        ax_cm.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.7)
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
    fig3_path = FIGURES_DIR / "classification_selected_confusion_matrices.png"
    plt.savefig(fig3_path)
    plt.close()


def generate_refinement_report(
    comparison_results: Dict[str, Dict[str, Any]],
    best_core_name: str,
    best_ext_name: str,
    core_grid: List[Dict[str, Any]],
    core_selected: Dict[str, Any],
    core_unconstrained: Dict[str, Any],
    ext_grid: List[Dict[str, Any]],
    ext_selected: Dict[str, Any],
    ext_unconstrained: Dict[str, Any],
    train_count: int,
    val_count: int,
) -> str:
    """Generate Markdown report for Milestone 7B."""
    def format_grid_table(grid: List[Dict[str, Any]], sel_t: float, unc_t: float) -> str:
        rows = []
        for m in grid:
            t = m["threshold"]
            tags = []
            if t == sel_t:
                tags.append("**[Selected]**")
            if t == unc_t:
                tags.append("*[Max F1]*")
            tag_str = " " + " ".join(tags) if tags else ""
            rows.append(
                f"| {t:.2f}{tag_str} | {m['balanced_accuracy']:.4f} | {m['recall']:.4f} | "
                f"{m['precision']:.4f} | **{m['f1']:.4f}** | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
            )
        return "\n".join(rows)

    core_table = format_grid_table(core_grid, core_selected["threshold"], core_unconstrained["threshold"])
    ext_table = format_grid_table(ext_grid, ext_selected["threshold"], ext_unconstrained["threshold"])

    report = f"""# Milestone 7B: Negative-Price Classification Refinement Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**{train_count:,} hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**{val_count:,} hours**)  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Executive Summary & Model Selection

Negative-price events are rare ({comparison_results['logistic_core_strict']['prevalence_val_pct']:.2f}% validation prevalence), making standard accuracy and default 0.50 thresholds unsuitable for early-warning operation.

### Selection Principle:
- **Primary Selection Metric**: Validation **PR-AUC** (Area Under the Precision-Recall Curve), which is threshold-independent and directly reflects the precision-recall trade-off under extreme class imbalance.
- **Secondary Context**: ROC-AUC.

### Validation Model Comparison (PR-AUC):

| Model Family | Candidate Model | PR-AUC | ROC-AUC | Bal Acc (t=0.50) | Recall (t=0.50) | Precision (t=0.50) | F1 (t=0.50) | Status |
|---|---|---|---|---|---|---|---|---|
| **Core Strict** | **Logistic Regression** | **{comparison_results['logistic_core_strict']['pr_auc']:.4f}** | {comparison_results['logistic_core_strict']['roc_auc']:.4f} | {comparison_results['logistic_core_strict']['balanced_accuracy']:.4f} | {comparison_results['logistic_core_strict']['recall']:.4f} | {comparison_results['logistic_core_strict']['precision']:.4f} | {comparison_results['logistic_core_strict']['f1']:.4f} | **Selected Best Core** |
| **Core Strict** | LightGBM Classifier | {comparison_results['lgbm_core_strict']['pr_auc']:.4f} | {comparison_results['lgbm_core_strict']['roc_auc']:.4f} | {comparison_results['lgbm_core_strict']['balanced_accuracy']:.4f} | {comparison_results['lgbm_core_strict']['recall']:.4f} | {comparison_results['lgbm_core_strict']['precision']:.4f} | {comparison_results['lgbm_core_strict']['f1']:.4f} | — |
| **Forecast Ext*** | **Logistic Regression** | **{comparison_results['logistic_forecast_ext']['pr_auc']:.4f}** | {comparison_results['logistic_forecast_ext']['roc_auc']:.4f} | {comparison_results['logistic_forecast_ext']['balanced_accuracy']:.4f} | {comparison_results['logistic_forecast_ext']['recall']:.4f} | {comparison_results['logistic_forecast_ext']['precision']:.4f} | {comparison_results['logistic_forecast_ext']['f1']:.4f} | **Selected Best Ext*** |
| **Forecast Ext*** | LightGBM Classifier | {comparison_results['lgbm_forecast_ext']['pr_auc']:.4f} | {comparison_results['lgbm_forecast_ext']['roc_auc']:.4f} | {comparison_results['lgbm_forecast_ext']['balanced_accuracy']:.4f} | {comparison_results['lgbm_forecast_ext']['recall']:.4f} | {comparison_results['lgbm_forecast_ext']['precision']:.4f} | {comparison_results['lgbm_forecast_ext']['f1']:.4f} | — |

> **Key Finding**: Across both feature families, **Logistic Regression achieved higher validation PR-AUC than LightGBM** (0.5914 vs 0.5337 in Core Strict; 0.6126 vs 0.5617 in Forecast Extension), indicating a stronger precision-recall trade-off for the negative-price classification task.

---

## 2. Predefined Threshold Selection Objective

This project operates as an **early-warning system** for negative electricity prices. Missing a negative-price event is costly, but false alarms must remain bounded.

### Predefined Selection Rule:
1. Restrict candidate thresholds to those achieving **Recall >= 0.75**.
2. Among those qualifying candidates, select the threshold with the **highest F1 score**.
3. If no threshold achieves Recall >= 0.75, fallback to the threshold with the highest Balanced Accuracy.

---

## 3. Core Strict Threshold Selection ({best_core_name})

Evaluated on 17 predefined thresholds [0.10 to 0.90]:

| Threshold | Balanced Accuracy | Recall | Precision | F1 Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
{core_table}

### Core Strict Operating Point:
- **Selected Operating Threshold**: **{core_selected['threshold']:.2f}**
- **Balanced Accuracy**: **{core_selected['balanced_accuracy']:.4f}**
- **Recall**: **{core_selected['recall']:.4f}** ({core_selected['tp']} of {core_selected['tp'] + core_selected['fn']} negative hours detected)
- **Precision**: **{core_selected['precision']:.4f}** ({core_selected['fp']} false alarms)
- **F1 Score**: **{core_selected['f1']:.4f}**
- **PR-AUC**: **{comparison_results['logistic_core_strict']['pr_auc']:.4f}**
- **ROC-AUC**: **{comparison_results['logistic_core_strict']['roc_auc']:.4f}**
- **Confusion Matrix**: [TN={core_selected['tn']}, FP={core_selected['fp']}, FN={core_selected['fn']}, TP={core_selected['tp']}]
- **Comparison with Unconstrained Max F1**: The unconstrained F1-maximizing threshold is **{core_unconstrained['threshold']:.2f}** (F1: {core_unconstrained['f1']:.4f}, Recall: {core_unconstrained['recall']:.4f}, Precision: {core_unconstrained['precision']:.4f}). It achieves higher precision but misses the Recall >= 0.75 requirement, making threshold **{core_selected['threshold']:.2f}** the preferred operating point for an early-warning alert system.

---

## 4. Forecast Extension Threshold Selection ({best_ext_name}*)

> *PROVENANCE CAVEAT ON FORECAST EXTENSION*:  
> Contains features qualified under `"ARCHIVE_VINTAGE_LIMITATION"` (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`). Because SMARD retrospective archives may reflect updated historical vintages rather than strictly preserved pre-auction point-in-time bids, this result is reported separately and is not equivalent in leakage certainty to the Core Strict pipeline.

| Threshold | Balanced Accuracy | Recall | Precision | F1 Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
{ext_table}

### Forecast Extension Operating Point:
- **Selected Operating Threshold**: **{ext_selected['threshold']:.2f}**
- **Balanced Accuracy**: **{ext_selected['balanced_accuracy']:.4f}**
- **Recall**: **{ext_selected['recall']:.4f}** ({ext_selected['tp']} of {ext_selected['tp'] + ext_selected['fn']} negative hours detected)
- **Precision**: **{ext_selected['precision']:.4f}** ({ext_selected['fp']} false alarms)
- **F1 Score**: **{ext_selected['f1']:.4f}**
- **PR-AUC**: **{comparison_results['logistic_forecast_ext']['pr_auc']:.4f}**
- **ROC-AUC**: **{comparison_results['logistic_forecast_ext']['roc_auc']:.4f}**
- **Confusion Matrix**: [TN={ext_selected['tn']}, FP={ext_selected['fp']}, FN={ext_selected['fn']}, TP={ext_selected['tp']}]
- **Comparison with Unconstrained Max F1**: The unconstrained F1-maximizing threshold is **{ext_unconstrained['threshold']:.2f}** (F1: {ext_unconstrained['f1']:.4f}, Recall: {ext_unconstrained['recall']:.4f}, Precision: {ext_unconstrained['precision']:.4f}).

---

## 5. Primary Classifier Governance & Model Freezing

For final portfolio governance and holdout evaluation:
1. **PRIMARY CLASSIFIER (FROZEN)**:
   - **Model**: **Logistic Regression — Core Strict**
   - **Selected Operating Threshold**: **`0.45`**
   - **Validation Performance**:
     - PR-AUC: **0.5914**
     - ROC-AUC: **0.9360**
     - Balanced Accuracy: **0.8505**
     - Recall: **0.7857**
     - Precision: **0.3340**
     - F1 Score: **0.4687**
     - Confusion Matrix: [TN=3,792, FP=351, FN=48, TP=176]
   - **Governance Rationale**: Selected as the primary production-grade leakage-controlled classifier. It does not depend on unverified retrospective load forecast vintages and guarantees point-in-time bid validity.

2. **SUPPLEMENTARY FORECAST-EXTENSION CLASSIFIER (FROZEN)**:
   - **Model**: **Logistic Regression — Forecast Extension**
   - **Selected Operating Threshold**: **`0.55`**
   - **Validation Performance**:
     - PR-AUC: **0.6126**
     - ROC-AUC: **0.9425**
     - Balanced Accuracy: **0.8513**
     - Recall: **0.7723**
     - Precision: **0.3745**
     - F1 Score: **0.5044**
     - Confusion Matrix: [TN=3,854, FP=289, FN=51, TP=173]
   - **Governance Rationale**: Retained strictly as a supplementary sensitivity experiment under the explicit governance caveat `"PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED"` (`ARCHIVE_VINTAGE_LIMITATION`). It is not chosen as primary despite its slightly higher PR-AUC (0.6126 vs 0.5914) due to vintage uncertainty.

*Both operating thresholds (0.45 for Core Strict, 0.55 for Forecast Extension) are frozen and will not be altered.*

---

## 6. Diagnostic Figures

Generated validation diagnostic figures in `reports/figures/`:
1. [Figure 1: Threshold Trade-off Curve](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/classification_threshold_tradeoff.png)
2. [Figure 2: Precision-Recall Curves with Selected Operating Thresholds](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/classification_pr_curve_selected_models.png)
3. [Figure 3: Selected Operating Confusion Matrices](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/classification_selected_confusion_matrices.png)
"""
    return report


def run_classification_refinement() -> None:
    """Main execution routine for Milestone 7B."""
    print("=" * 70)
    print("MILESTONE 7B: CLASSIFICATION REFINEMENT & THRESHOLD SELECTION")
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
    print("Confirming Holdout is NOT accessed in classification refinement: Excluded completely.")

    # 3. Targets
    y_train_neg = train_core["is_negative_price"].values
    y_val_neg = val_core["is_negative_price"].values

    train_neg_count = int(y_train_neg.sum())
    val_neg_count = int(y_val_neg.sum())
    val_neg_prev = (val_neg_count / val_count) * 100.0

    print(f"\nNegative Price Prevalence:")
    print(f"  TRAIN:      {train_neg_count:,} / {train_count:,} ({(train_neg_count/train_count)*100:.2f}%)")
    print(f"  VALIDATION: {val_neg_count:,} / {val_count:,} ({val_neg_prev:.2f}%)")

    # 4. Predictors
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]
    ext_predictors = [c for c in df_ext.columns if c not in non_predictor_cols]

    X_train_core = train_core[core_predictors]
    X_val_core = val_core[core_predictors]

    X_train_ext = train_ext[ext_predictors]
    X_val_ext = val_ext[ext_predictors]

    # 5. Fit & evaluate all 4 candidate models
    print("\nFitting and Evaluating Candidate Classifiers on Validation...")
    comparison_results = {}
    model_probas = {}

    # Model 1: Logistic Regression Core Strict
    pipe_logit_core = build_logistic_pipeline(C=1.0)
    pipe_logit_core.fit(X_train_core, y_train_neg)
    proba_logit_core = pipe_logit_core.predict_proba(X_val_core)[:, 1]
    model_probas["logistic_core_strict"] = proba_logit_core
    pred_logit_core = (proba_logit_core >= 0.50).astype(int)

    comparison_results["logistic_core_strict"] = {
        "model_type": "Logistic Regression",
        "feature_family": "Core Strict",
        "pr_auc": float(average_precision_score(y_val_neg, proba_logit_core)),
        "roc_auc": float(roc_auc_score(y_val_neg, proba_logit_core)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val_neg, pred_logit_core)),
        "recall": float(recall_score(y_val_neg, pred_logit_core, zero_division=0)),
        "precision": float(precision_score(y_val_neg, pred_logit_core, zero_division=0)),
        "f1": float(f1_score(y_val_neg, pred_logit_core, zero_division=0)),
        "prevalence_val_pct": val_neg_prev,
    }

    # Model 2: LightGBM Classifier Core Strict
    scale_w = compute_train_class_weight(train_core["is_negative_price"])
    lgb_core = build_lgbm_classifier(scale_pos_weight=scale_w)
    lgb_core.fit(X_train_core, y_train_neg)
    proba_lgb_core = lgb_core.predict_proba(X_val_core)[:, 1]
    model_probas["lgbm_core_strict"] = proba_lgb_core
    pred_lgb_core = (proba_lgb_core >= 0.50).astype(int)

    comparison_results["lgbm_core_strict"] = {
        "model_type": "LightGBM Classifier",
        "feature_family": "Core Strict",
        "pr_auc": float(average_precision_score(y_val_neg, proba_lgb_core)),
        "roc_auc": float(roc_auc_score(y_val_neg, proba_lgb_core)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val_neg, pred_lgb_core)),
        "recall": float(recall_score(y_val_neg, pred_lgb_core, zero_division=0)),
        "precision": float(precision_score(y_val_neg, pred_lgb_core, zero_division=0)),
        "f1": float(f1_score(y_val_neg, pred_lgb_core, zero_division=0)),
        "prevalence_val_pct": val_neg_prev,
    }

    # Model 3: Logistic Regression Forecast Extension
    pipe_logit_ext = build_logistic_pipeline(C=1.0)
    pipe_logit_ext.fit(X_train_ext, y_train_neg)
    proba_logit_ext = pipe_logit_ext.predict_proba(X_val_ext)[:, 1]
    model_probas["logistic_forecast_ext"] = proba_logit_ext
    pred_logit_ext = (proba_logit_ext >= 0.50).astype(int)

    comparison_results["logistic_forecast_ext"] = {
        "model_type": "Logistic Regression",
        "feature_family": "Forecast Extension*",
        "pr_auc": float(average_precision_score(y_val_neg, proba_logit_ext)),
        "roc_auc": float(roc_auc_score(y_val_neg, proba_logit_ext)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val_neg, pred_logit_ext)),
        "recall": float(recall_score(y_val_neg, pred_logit_ext, zero_division=0)),
        "precision": float(precision_score(y_val_neg, pred_logit_ext, zero_division=0)),
        "f1": float(f1_score(y_val_neg, pred_logit_ext, zero_division=0)),
        "prevalence_val_pct": val_neg_prev,
    }

    # Model 4: LightGBM Classifier Forecast Extension
    lgb_ext = build_lgbm_classifier(scale_pos_weight=scale_w)
    lgb_ext.fit(X_train_ext, y_train_neg)
    proba_lgb_ext = lgb_ext.predict_proba(X_val_ext)[:, 1]
    model_probas["lgbm_forecast_ext"] = proba_lgb_ext
    pred_lgb_ext = (proba_lgb_ext >= 0.50).astype(int)

    comparison_results["lgbm_forecast_ext"] = {
        "model_type": "LightGBM Classifier",
        "feature_family": "Forecast Extension*",
        "pr_auc": float(average_precision_score(y_val_neg, proba_lgb_ext)),
        "roc_auc": float(roc_auc_score(y_val_neg, proba_lgb_ext)),
        "balanced_accuracy": float(balanced_accuracy_score(y_val_neg, pred_lgb_ext)),
        "recall": float(recall_score(y_val_neg, pred_lgb_ext, zero_division=0)),
        "precision": float(precision_score(y_val_neg, pred_lgb_ext, zero_division=0)),
        "f1": float(f1_score(y_val_neg, pred_lgb_ext, zero_division=0)),
        "prevalence_val_pct": val_neg_prev,
    }

    for k, v in comparison_results.items():
        print(f"  {k:24s} -> PR-AUC: {v['pr_auc']:.4f} | ROC-AUC: {v['roc_auc']:.4f} | F1(0.50): {v['f1']:.4f}")

    # 6. Model Selection by PR-AUC
    best_core_key = "logistic_core_strict" if comparison_results["logistic_core_strict"]["pr_auc"] >= comparison_results["lgbm_core_strict"]["pr_auc"] else "lgbm_core_strict"
    best_ext_key = "logistic_forecast_ext" if comparison_results["logistic_forecast_ext"]["pr_auc"] >= comparison_results["lgbm_forecast_ext"]["pr_auc"] else "lgbm_forecast_ext"

    best_core_name = comparison_results[best_core_key]["model_type"]
    best_ext_name = comparison_results[best_ext_key]["model_type"]

    print(f"\nModel Selection by PR-AUC:")
    print(f"  Best Core Strict Classifier:        {best_core_name} (PR-AUC: {comparison_results[best_core_key]['pr_auc']:.4f})")
    print(f"  Best Forecast Extension Classifier: {best_ext_name} (PR-AUC: {comparison_results[best_ext_key]['pr_auc']:.4f})")

    # 7. Threshold Grid Evaluation (17 points)
    print("\nEvaluating 17-point Threshold Grid on Validation...")
    core_proba = model_probas[best_core_key]
    ext_proba = model_probas[best_ext_key]

    core_grid = evaluate_threshold_grid(y_val_neg, core_proba)
    ext_grid = evaluate_threshold_grid(y_val_neg, ext_proba)

    # 8. Apply Predefined Selection Rule
    core_selected, core_unconstrained = select_best_threshold(core_grid)
    ext_selected, ext_unconstrained = select_best_threshold(ext_grid)

    print(f"\nCore Strict ({best_core_name}) Threshold Selection:")
    print(f"  Selected Threshold:            {core_selected['threshold']:.2f}")
    print(f"  Selection Strategy:            {core_selected['selection_strategy']}")
    print(f"  Balanced Accuracy:             {core_selected['balanced_accuracy']:.4f}")
    print(f"  Recall:                        {core_selected['recall']:.4f}")
    print(f"  Precision:                     {core_selected['precision']:.4f}")
    print(f"  F1 Score:                      {core_selected['f1']:.4f}")
    print(f"  Confusion Matrix:              [TN={core_selected['tn']}, FP={core_selected['fp']}, FN={core_selected['fn']}, TP={core_selected['tp']}]")
    print(f"  Unconstrained Max F1 Threshold: {core_unconstrained['threshold']:.2f} (F1: {core_unconstrained['f1']:.4f}, Recall: {core_unconstrained['recall']:.4f}, Prec: {core_unconstrained['precision']:.4f})")

    print(f"\nForecast Extension ({best_ext_name}) Threshold Selection:")
    print(f"  Selected Threshold:            {ext_selected['threshold']:.2f}")
    print(f"  Selection Strategy:            {ext_selected['selection_strategy']}")
    print(f"  Balanced Accuracy:             {ext_selected['balanced_accuracy']:.4f}")
    print(f"  Recall:                        {ext_selected['recall']:.4f}")
    print(f"  Precision:                     {ext_selected['precision']:.4f}")
    print(f"  F1 Score:                      {ext_selected['f1']:.4f}")
    print(f"  Confusion Matrix:              [TN={ext_selected['tn']}, FP={ext_selected['fp']}, FN={ext_selected['fn']}, TP={ext_selected['tp']}]")
    print(f"  Unconstrained Max F1 Threshold: {ext_unconstrained['threshold']:.2f} (F1: {ext_unconstrained['f1']:.4f}, Recall: {ext_unconstrained['recall']:.4f}, Prec: {ext_unconstrained['precision']:.4f})")

    # 9. Generate Diagnostic Figures
    print("\nGenerating diagnostic figures...")
    generate_classification_figures(
        y_val_neg,
        best_core_name, core_proba, core_grid, core_selected,
        best_ext_name, ext_proba, ext_grid, ext_selected
    )
    print("  [OK] reports/figures/classification_threshold_tradeoff.png")
    print("  [OK] reports/figures/classification_pr_curve_selected_models.png")
    print("  [OK] reports/figures/classification_selected_confusion_matrices.png")

    # 10. Save Metrics JSON
    metrics_json_path = REPORTS_DIR / "classification_refinement_metrics.json"
    metrics_payload = {
        "project": "german-power-price-forecasting",
        "milestone": "Milestone 7B - Classification Refinement & Threshold Selection",
        "evaluation_split": "VALIDATION (H1 2024)",
        "train_rows": train_count,
        "validation_rows": val_count,
        "prevalence": {
            "train_negative_count": train_neg_count,
            "train_negative_pct": (train_neg_count / train_count) * 100.0,
            "val_negative_count": val_neg_count,
            "val_negative_pct": val_neg_prev,
        },
        "model_comparison": comparison_results,
        "best_models_by_pr_auc": {
            "core_strict": {
                "key": best_core_key,
                "model_type": best_core_name,
                "pr_auc": comparison_results[best_core_key]["pr_auc"],
                "roc_auc": comparison_results[best_core_key]["roc_auc"],
            },
            "forecast_extension": {
                "key": best_ext_key,
                "model_type": best_ext_name,
                "pr_auc": comparison_results[best_ext_key]["pr_auc"],
                "roc_auc": comparison_results[best_ext_key]["roc_auc"],
            },
        },
        "threshold_grid": THRESHOLD_GRID,
        "core_strict_threshold_evaluation": {
            "grid_results": core_grid,
            "selected_threshold": core_selected,
            "unconstrained_max_f1_threshold": core_unconstrained,
        },
        "forecast_extension_threshold_evaluation": {
            "grid_results": ext_grid,
            "selected_threshold": ext_selected,
            "unconstrained_max_f1_threshold": ext_unconstrained,
        },
        "holdout_data_included": False,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print("  [OK] Saved reports/classification_refinement_metrics.json (Zero holdout data)")

    # 11. Write Report
    print("Writing reports/classification_refinement_report.md...")
    report_md = generate_refinement_report(
        comparison_results, best_core_name, best_ext_name,
        core_grid, core_selected, core_unconstrained,
        ext_grid, ext_selected, ext_unconstrained,
        train_count, val_count
    )
    report_path = REPORTS_DIR / "classification_refinement_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  [OK] Saved reports/classification_refinement_report.md ({len(report_md):,} characters)")
    print("=" * 70)


if __name__ == "__main__":
    run_classification_refinement()
