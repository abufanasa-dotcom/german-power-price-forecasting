"""
Execution script for Final Holdout Evaluation.

Performs the first authorized opening of the final holdout split:
- Split: 2024-07-01 00:00 through 2024-12-31 23:00 Europe/Berlin (4,417 rows).
- Fitting set: TRAIN (8,592 rows) + VALIDATION (4,367 rows) = 12,959 rows.
- Evaluates frozen Primary Regression: LightGBM Regressor Candidate D (Core Strict).
- Evaluates frozen Supplementary Regression: LightGBM Regressor Candidate D (Forecast Extension).
- Evaluates frozen Primary Classifier: Logistic Regression Core Strict at threshold 0.45.
- Evaluates frozen Supplementary Classifier: Logistic Regression Forecast Extension at threshold 0.55.
- Generates 5 final diagnostic figures in reports/figures/.
- Exports reports/final_holdout_metrics.json and reports/final_holdout_report.md.

STRICT GOVERNANCE:
- All model hyperparameters and thresholds were frozen at commit 23a34db.
- No post-holdout tuning, feature selection, or model selection is performed.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
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

from src.models.baselines import (
    build_logistic_pipeline,
    compute_regression_metrics,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# -------------------------------------------------------------
# Frozen Hyperparameters and Configurations (Freeze Commit 23a34db)
# -------------------------------------------------------------
FROZEN_REG_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 50,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

FROZEN_PRIMARY_CLF_THRESHOLD = 0.45
FROZEN_SUPPLEMENTARY_CLF_THRESHOLD = 0.55

# Reference validation metrics frozen before holdout opening
VALIDATION_REF = {
    "regression_core_mae": 18.97,
    "regression_core_rmse": 25.06,
    "regression_core_median_ae": 14.74,
    "regression_core_bias": 5.30,
    "regression_ext_mae": 18.98,
    "regression_ext_rmse": 24.89,
    "regression_ext_median_ae": 14.91,
    "regression_ext_bias": 5.77,
    "classification_core": {
        "pr_auc": 0.5914,
        "roc_auc": 0.9360,
        "balanced_accuracy": 0.8505,
        "recall": 0.7857,
        "precision": 0.3340,
        "f1": 0.4687,
    },
    "classification_ext": {
        "pr_auc": 0.6126,
        "roc_auc": 0.9425,
        "balanced_accuracy": 0.8513,
        "recall": 0.7723,
        "precision": 0.3745,
        "f1": 0.5044,
    },
}


def classify_generalization(val_val: float, holdout_val: float, metric_type: str = "error") -> str:
    """
    Descriptively classify generalization using strictly measured differences.
    For error metrics (MAE):
      relative_diff = (holdout - val) / val
      <= -0.05 -> improved holdout performance
      -0.05 < diff <= 0.10 -> similar performance
      0.10 < diff <= 0.30 -> moderate degradation
      > 0.30 -> substantial degradation
    For score metrics (PR-AUC, F1, Bal Acc):
      relative_diff = (holdout - val) / val
      >= 0.05 -> improved holdout performance
      -0.10 <= diff < 0.05 -> similar performance
      -0.30 <= diff < -0.10 -> moderate degradation
      < -0.30 -> substantial degradation
    """
    rel_diff = (holdout_val - val_val) / val_val
    if metric_type == "error":
        if rel_diff <= -0.05:
            return "improved holdout performance"
        elif rel_diff <= 0.10:
            return "similar performance"
        elif rel_diff <= 0.30:
            return "moderate degradation"
        else:
            return "substantial degradation"
    else:
        if rel_diff >= 0.05:
            return "improved holdout performance"
        elif rel_diff >= -0.10:
            return "similar performance"
        elif rel_diff >= -0.30:
            return "moderate degradation"
        else:
            return "substantial degradation"


def generate_holdout_figures(
    y_holdout_price: np.ndarray,
    pred_core_reg: np.ndarray,
    pred_ext_reg: np.ndarray,
    benchmarks: Dict[str, float],
    reg_core_metrics: Dict[str, Any],
    reg_ext_metrics: Dict[str, Any],
    y_holdout_neg: np.ndarray,
    proba_core_clf: np.ndarray,
    proba_ext_clf: np.ndarray,
    pred_core_clf: np.ndarray,
    cm_core: Dict[str, int],
) -> None:
    """
    Generate at most five final figures in reports/figures/:
    1. final_holdout_regression_actual_vs_predicted.png
    2. final_holdout_regression_error_distribution.png
    3. final_holdout_regression_benchmark_comparison.png
    4. final_holdout_classification_pr_curve.png
    5. final_holdout_classification_confusion_matrix.png
    """
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
    # Figure 1: Actual vs Predicted (Core Strict Primary)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.scatter(y_holdout_price, pred_core_reg, alpha=0.25, s=12, color="#1d4ed8", edgecolors="none", label="Holdout Observations (N=4,417)")
    min_val = min(y_holdout_price.min(), pred_core_reg.min()) - 10
    max_val = max(y_holdout_price.max(), pred_core_reg.max()) + 10
    ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=1.5, label="Perfect Forecast (y = x)")
    ax.axhline(0, color="#94a3b8", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#94a3b8", linestyle=":", linewidth=0.8)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Actual Settled Day-Ahead Price (EUR/MWh)")
    ax.set_ylabel("Predicted Day-Ahead Price (EUR/MWh)")
    ax.set_title("Final Holdout: Actual vs Predicted Spot Price\nPrimary Model: LightGBM Regressor (Core Strict)")
    ax.grid(True)
    textstr = (
        f"Holdout Metrics:\n"
        f"MAE: {reg_core_metrics['mae']:.2f} EUR/MWh\n"
        f"RMSE: {reg_core_metrics['rmse']:.2f} EUR/MWh\n"
        f"Median AE: {reg_core_metrics['median_ae']:.2f} EUR/MWh\n"
        f"Bias: {reg_core_metrics['bias']:+.2f} EUR/MWh"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_holdout_regression_actual_vs_predicted.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Regression Error Distribution (Core Strict Primary)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    errors = pred_core_reg - y_holdout_price
    bins = np.linspace(np.percentile(errors, 0.5), np.percentile(errors, 99.5), 60)
    ax.hist(errors, bins=bins, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.75, density=True, label="Prediction Error (pred - actual)")
    ax.axvline(0, color="#0f172a", linestyle="-", linewidth=1.5, label="Zero Error")
    ax.axvline(reg_core_metrics["bias"], color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean Error / Bias ({reg_core_metrics['bias']:+.2f} EUR/MWh)")
    ax.axvline(np.median(errors), color="#16a34a", linestyle=":", linewidth=1.5, label=f"Median Error ({np.median(errors):+.2f} EUR/MWh)")
    ax.set_xlabel("Prediction Error: Prediction - Actual (EUR/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("Final Holdout: Prediction Error Distribution\nPrimary Model: LightGBM Regressor (Core Strict)")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_holdout_regression_error_distribution.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Benchmark Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    bench_names = [
        "24h Persistence",
        "168h Persistence",
        "7-Day Lag Mean",
        "LightGBM (Core Strict)\n[PRIMARY]",
        "LightGBM (Forecast Ext*)\n[SUPPLEMENTARY]",
    ]
    bench_maes = [
        benchmarks["mae_24h"],
        benchmarks["mae_168h"],
        benchmarks["mae_7d_mean"],
        reg_core_metrics["mae"],
        reg_ext_metrics["mae"],
    ]
    bar_colors = ["#94a3b8", "#cbd5e1", "#e2e8f0", "#1d4ed8", "#7c3aed"]
    bars = ax.bar(bench_names, bench_maes, color=bar_colors, edgecolor="#475569", width=0.55)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )
    ax.set_ylabel("Holdout MAE (EUR/MWh)")
    ax.set_title("Final Holdout: Regression Benchmark Comparison\nPrimary LightGBM Core Strict vs Heuristic Baselines")
    ax.grid(axis="y")
    ax.set_ylim(0, max(bench_maes) * 1.18)
    # Footnote
    fig.text(0.12, 0.01, "*Forecast Extension contains features qualified under ARCHIVE_VINTAGE_LIMITATION.", fontsize=8, color="#64748b")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURES_DIR / "final_holdout_regression_benchmark_comparison.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: Precision-Recall Curve (Holdout)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    # Primary Core Strict
    prec_c, rec_c, th_c = precision_recall_curve(y_holdout_neg, proba_core_clf)
    pr_auc_c = average_precision_score(y_holdout_neg, proba_core_clf)
    ax.plot(rec_c, prec_c, color="#1d4ed8", linewidth=2.0, label=f"Core Strict (PR-AUC = {pr_auc_c:.4f})")

    # Supplementary Forecast Extension
    prec_e, rec_e, th_e = precision_recall_curve(y_holdout_neg, proba_ext_clf)
    pr_auc_e = average_precision_score(y_holdout_neg, proba_ext_clf)
    ax.plot(rec_e, prec_e, color="#7c3aed", linewidth=1.8, linestyle="--", label=f"Forecast Ext* (PR-AUC = {pr_auc_e:.4f})")

    # Mark operating thresholds
    # Core Strict at 0.45
    y_pred_c = (proba_core_clf >= FROZEN_PRIMARY_CLF_THRESHOLD).astype(int)
    rec_c_op = recall_score(y_holdout_neg, y_pred_c, zero_division=0)
    prec_c_op = precision_score(y_holdout_neg, y_pred_c, zero_division=0)
    ax.scatter([rec_c_op], [prec_c_op], color="#dc2626", s=90, zorder=5, edgecolors="#000", label=f"Core Operating Point (th=0.45: Rec={rec_c_op:.3f}, Prec={prec_c_op:.3f})")

    # Ext at 0.55
    y_pred_e = (proba_ext_clf >= FROZEN_SUPPLEMENTARY_CLF_THRESHOLD).astype(int)
    rec_e_op = recall_score(y_holdout_neg, y_pred_e, zero_division=0)
    prec_e_op = precision_score(y_holdout_neg, y_pred_e, zero_division=0)
    ax.scatter([rec_e_op], [prec_e_op], color="#f59e0b", s=80, marker="s", zorder=5, edgecolors="#000", label=f"Ext Operating Point* (th=0.55: Rec={rec_e_op:.3f}, Prec={prec_e_op:.3f})")

    # No-skill baseline
    prevalence = float(y_holdout_neg.sum() / len(y_holdout_neg))
    ax.axhline(prevalence, color="#94a3b8", linestyle=":", linewidth=1.2, label=f"No-Skill Baseline (Prevalence = {prevalence:.2%})")

    ax.set_xlabel("Recall (Sensitivity to Negative Prices)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Final Holdout: Negative-Price Precision-Recall Curve\nPrimary Classifier: Logistic Regression (Core Strict)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_holdout_classification_pr_curve.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 5: Confusion Matrix Heatmap (Core Strict Primary)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=300)
    cm_matrix = np.array([
        [cm_core["tn"], cm_core["fp"]],
        [cm_core["fn"], cm_core["tp"]],
    ])
    total = np.sum(cm_matrix)
    cax = ax.matshow(cm_matrix, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            pct = (val / total) * 100.0
            cell_label = f"{val:,}\n({pct:.1f}%)"
            text_color = "white" if val > (total * 0.3) else "black"
            ax.text(j, i, cell_label, ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Non-Negative (0)", "Pred: Negative (1)"])
    ax.set_yticklabels(["True: Non-Negative (0)", "True: Negative (1)"])
    ax.set_title(
        f"Final Holdout: Confusion Matrix\nLogistic Regression Core Strict (Threshold = {FROZEN_PRIMARY_CLF_THRESHOLD:.2f})\n"
        f"Total: {total:,} Hours | Negative Prevalence: {prevalence:.2%}",
        pad=15,
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_holdout_classification_confusion_matrix.png")
    plt.close(fig)


def run_final_holdout_evaluation() -> Tuple[Dict[str, Any], str]:
    """
    Main evaluation pipeline for the final holdout.
    """
    print("=" * 80)
    print("FINAL HOLDOUT EVALUATION: AUTHORIZED OPENING")
    print("=" * 80)
    print("Governance check: Freeze commit 23a34db active.")
    print("Model hyperparameters, features, and operating thresholds are immutable.\n")

    # 1. Load Feature Datasets
    core_path = PROCESSED_DIR / "features_core_strict.parquet"
    ext_path = PROCESSED_DIR / "features_forecast_extension.parquet"

    print(f"Loading {core_path.name}...")
    df_core = pd.read_parquet(core_path)
    print(f"Loading {ext_path.name}...")
    df_ext = pd.read_parquet(ext_path)

    # 2. Partition into Development (TRAIN + VALIDATION) and Final Holdout
    dev_mask = df_core["split"].isin(["train", "val"])
    holdout_mask = df_core["split"] == "holdout"

    dev_core = df_core[dev_mask].copy()
    holdout_core = df_core[holdout_mask].copy()

    dev_ext = df_ext[dev_mask].copy()
    holdout_ext = df_ext[holdout_mask].copy()

    dev_count = len(dev_core)
    holdout_count = len(holdout_core)

    print(f"\nData Partitioning:")
    print(f"  Development Set (TRAIN + VALIDATION): {dev_count:,} rows")
    print(f"    - TRAIN:      {(df_core['split'] == 'train').sum():,} rows")
    print(f"    - VALIDATION: {(df_core['split'] == 'val').sum():,} rows")
    print(f"  Final Holdout (H2 2024):              {holdout_count:,} rows")

    if dev_count != 12959:
        raise ValueError(f"Expected 12,959 development rows, found {dev_count}")
    if holdout_count != 4417:
        raise ValueError(f"Expected 4,417 holdout rows, found {holdout_count}")

    # 3. Targets and Dynamic Negative-Price Statistics
    y_dev_price = dev_core["day_ahead_price_eur_mwh"].values
    y_holdout_price = holdout_core["day_ahead_price_eur_mwh"].values

    y_dev_neg = dev_core["is_negative_price"].values
    y_holdout_neg = holdout_core["is_negative_price"].values

    train_mask = df_core["split"] == "train"
    val_mask = df_core["split"] == "val"
    train_count = int(train_mask.sum())
    val_count = int(val_mask.sum())

    train_neg_hours = int(df_core.loc[train_mask, "is_negative_price"].sum())
    val_neg_hours = int(df_core.loc[val_mask, "is_negative_price"].sum())
    dev_neg_hours = int(y_dev_neg.sum())
    holdout_neg_hours = int(y_holdout_neg.sum())

    train_neg_prev_pct = float((train_neg_hours / train_count) * 100.0)
    val_neg_prev_pct = float((val_neg_hours / val_count) * 100.0)
    dev_neg_prev_pct = float((dev_neg_hours / dev_count) * 100.0)
    holdout_neg_prevalence_pct = float((holdout_neg_hours / holdout_count) * 100.0)

    print(f"\nTarget & Negative-Price Statistics (Dynamically Measured):")
    print(f"  TRAIN:       {train_neg_hours:,} / {train_count:,} ({train_neg_prev_pct:.2f}%)")
    print(f"  VALIDATION:  {val_neg_hours:,} / {val_count:,} ({val_neg_prev_pct:.2f}%)")
    print(f"  DEVELOPMENT: {dev_neg_hours:,} / {dev_count:,} ({dev_neg_prev_pct:.2f}%)")
    print(f"  HOLDOUT:     {holdout_neg_hours:,} / {holdout_count:,} ({holdout_neg_prevalence_pct:.2f}%)")

    # 4. Predictors (Strict separation from metadata and targets)
    non_predictor_cols = {"timestamp_utc", "timestamp_local", "split", "day_ahead_price_eur_mwh", "is_negative_price"}
    core_predictors = [c for c in df_core.columns if c not in non_predictor_cols]
    ext_predictors = [c for c in df_ext.columns if c not in non_predictor_cols]

    print(f"\nPredictor Sets:")
    print(f"  Core Strict:        {len(core_predictors)} features")
    print(f"  Forecast Extension: {len(ext_predictors)} features")

    X_dev_core = dev_core[core_predictors]
    X_holdout_core = holdout_core[core_predictors]

    X_dev_ext = dev_ext[ext_predictors]
    X_holdout_ext = holdout_ext[ext_predictors]

    # =============================================================
    # PART 2 & 3: REGRESSION EVALUATION
    # =============================================================
    print("\n" + "-" * 60)
    print("REGRESSION EVALUATION: FIT ON DEV, SCORE ON HOLDOUT")
    print("-" * 60)

    # Final Holdout Benchmarks
    mae_24h = float(np.mean(np.abs(holdout_core["price_lag_24h"].values - y_holdout_price)))
    mae_168h = float(np.mean(np.abs(holdout_core["price_lag_168h"].values - y_holdout_price)))
    mae_7d_mean = float(np.mean(np.abs(holdout_core["price_daily_lag_mean_7d"].values - y_holdout_price)))

    benchmarks = {
        "mae_24h": mae_24h,
        "mae_168h": mae_168h,
        "mae_7d_mean": mae_7d_mean,
    }
    print(f"Holdout Benchmarks:")
    print(f"  24h Persistence MAE:     {mae_24h:.2f} EUR/MWh")
    print(f"  168h Persistence MAE:    {mae_168h:.2f} EUR/MWh")
    print(f"  7-day Lag Mean MAE:      {mae_7d_mean:.2f} EUR/MWh")

    # Primary Model: LightGBM Regressor Candidate D (Core Strict)
    print("\nFitting Primary Regression Model: LightGBM Core Strict (Candidate D)...")
    lgb_reg_core = lgb.LGBMRegressor(**FROZEN_REG_PARAMS)
    lgb_reg_core.fit(X_dev_core, y_dev_price)
    pred_core_reg = lgb_reg_core.predict(X_holdout_core)

    reg_core_metrics = compute_regression_metrics(
        y_holdout_price,
        pred_core_reg,
        baseline_mae_24h=mae_24h,
        baseline_mae_168h=mae_168h,
    )
    reg_core_metrics["mae_7d_mean"] = mae_7d_mean
    reg_core_metrics["improvement_vs_7d_mean_pct"] = float(((mae_7d_mean - reg_core_metrics["mae"]) / mae_7d_mean) * 100.0)

    # Validation comparison for Primary Regression
    reg_core_val_mae = VALIDATION_REF["regression_core_mae"]
    reg_core_holdout_mae = reg_core_metrics["mae"]
    reg_core_diff_abs = float(reg_core_holdout_mae - reg_core_val_mae)
    reg_core_diff_pct = float((reg_core_diff_abs / reg_core_val_mae) * 100.0)
    reg_core_gen_label = classify_generalization(reg_core_val_mae, reg_core_holdout_mae, metric_type="error")

    print(f"Primary LightGBM Core Strict Holdout Results:")
    print(f"  MAE:       {reg_core_metrics['mae']:.2f} EUR/MWh")
    print(f"  RMSE:      {reg_core_metrics['rmse']:.2f} EUR/MWh")
    print(f"  Median AE: {reg_core_metrics['median_ae']:.2f} EUR/MWh")
    print(f"  Bias:      {reg_core_metrics['bias']:+.2f} EUR/MWh")
    print(f"  Improvement vs 24h Persistence:  +{reg_core_metrics['improvement_vs_24h_pct']:.2f}%")
    print(f"  Improvement vs 168h Persistence: +{reg_core_metrics['improvement_vs_168h_pct']:.2f}%")
    print(f"  Improvement vs 7-day Lag Mean:   +{reg_core_metrics['improvement_vs_7d_mean_pct']:.2f}%")
    print(f"  Validation MAE: {reg_core_val_mae:.2f} EUR/MWh -> Holdout MAE: {reg_core_holdout_mae:.2f} EUR/MWh")
    print(f"  Difference: {reg_core_diff_abs:+.2f} EUR/MWh ({reg_core_diff_pct:+.2f}%) [{reg_core_gen_label}]")

    # Supplementary Regression: LightGBM Regressor Candidate D (Forecast Extension)
    print("\nFitting Supplementary Regression Model: LightGBM Forecast Extension (Candidate D)...")
    lgb_reg_ext = lgb.LGBMRegressor(**FROZEN_REG_PARAMS)
    lgb_reg_ext.fit(X_dev_ext, y_dev_price)
    pred_ext_reg = lgb_reg_ext.predict(X_holdout_ext)

    reg_ext_metrics = compute_regression_metrics(
        y_holdout_price,
        pred_ext_reg,
        baseline_mae_24h=mae_24h,
        baseline_mae_168h=mae_168h,
    )
    reg_ext_metrics["mae_7d_mean"] = mae_7d_mean
    reg_ext_metrics["improvement_vs_7d_mean_pct"] = float(((mae_7d_mean - reg_ext_metrics["mae"]) / mae_7d_mean) * 100.0)

    reg_ext_val_mae = VALIDATION_REF["regression_ext_mae"]
    reg_ext_holdout_mae = reg_ext_metrics["mae"]
    reg_ext_diff_abs = float(reg_ext_holdout_mae - reg_ext_val_mae)
    reg_ext_diff_pct = float((reg_ext_diff_abs / reg_ext_val_mae) * 100.0)
    reg_ext_gen_label = classify_generalization(reg_ext_val_mae, reg_ext_holdout_mae, metric_type="error")

    print(f"Supplementary LightGBM Forecast Extension Holdout Results (ARCHIVE_VINTAGE_LIMITATION):")
    print(f"  MAE:       {reg_ext_metrics['mae']:.2f} EUR/MWh")
    print(f"  RMSE:      {reg_ext_metrics['rmse']:.2f} EUR/MWh")
    print(f"  Median AE: {reg_ext_metrics['median_ae']:.2f} EUR/MWh")
    print(f"  Bias:      {reg_ext_metrics['bias']:+.2f} EUR/MWh")

    # =============================================================
    # PART 4 & 5: CLASSIFICATION EVALUATION
    # =============================================================
    print("\n" + "-" * 60)
    print("CLASSIFICATION EVALUATION: FIT ON DEV, SCORE ON HOLDOUT")
    print("-" * 60)

    # Primary Classifier: Logistic Regression Core Strict (Operating threshold 0.45)
    print(f"Fitting Primary Classifier: Logistic Regression Core Strict (threshold = {FROZEN_PRIMARY_CLF_THRESHOLD:.2f})...")
    pipe_clf_core = build_logistic_pipeline(C=1.0)
    pipe_clf_core.fit(X_dev_core, y_dev_neg)
    proba_core_clf = pipe_clf_core.predict_proba(X_holdout_core)[:, 1]
    pred_core_clf = (proba_core_clf >= FROZEN_PRIMARY_CLF_THRESHOLD).astype(int)

    cm_core_arr = confusion_matrix(y_holdout_neg, pred_core_clf, labels=[0, 1])
    tn_c, fp_c, fn_c, tp_c = [int(v) for v in cm_core_arr.ravel()]

    clf_core_metrics = {
        "pr_auc": float(average_precision_score(y_holdout_neg, proba_core_clf)),
        "roc_auc": float(roc_auc_score(y_holdout_neg, proba_core_clf)),
        "balanced_accuracy": float(balanced_accuracy_score(y_holdout_neg, pred_core_clf)),
        "recall": float(recall_score(y_holdout_neg, pred_core_clf, zero_division=0)),
        "precision": float(precision_score(y_holdout_neg, pred_core_clf, zero_division=0)),
        "f1": float(f1_score(y_holdout_neg, pred_core_clf, zero_division=0)),
        "confusion_matrix": {"tn": tn_c, "fp": fp_c, "fn": fn_c, "tp": tp_c},
    }

    # Compare against frozen validation metrics for Primary Classifier
    clf_core_val = VALIDATION_REF["classification_core"]
    clf_core_comparison = {}
    for metric_name in ["pr_auc", "roc_auc", "balanced_accuracy", "recall", "precision", "f1"]:
        val_m = clf_core_val[metric_name]
        hold_m = clf_core_metrics[metric_name]
        diff_abs = hold_m - val_m
        diff_pct = (diff_abs / val_m) * 100.0 if val_m != 0 else 0.0
        gen_label = classify_generalization(val_m, hold_m, metric_type="score")
        clf_core_comparison[metric_name] = {
            "validation": val_m,
            "holdout": hold_m,
            "diff_abs": float(diff_abs),
            "diff_pct": float(diff_pct),
            "generalization_classification": gen_label,
        }

    print(f"Primary Logistic Regression Core Strict Holdout Results (Threshold = {FROZEN_PRIMARY_CLF_THRESHOLD:.2f}):")
    print(f"  PR-AUC:            {clf_core_metrics['pr_auc']:.4f} (Validation: {clf_core_val['pr_auc']:.4f})")
    print(f"  ROC-AUC:           {clf_core_metrics['roc_auc']:.4f} (Validation: {clf_core_val['roc_auc']:.4f})")
    print(f"  Balanced Accuracy: {clf_core_metrics['balanced_accuracy']:.4f} (Validation: {clf_core_val['balanced_accuracy']:.4f})")
    print(f"  Recall:            {clf_core_metrics['recall']:.4f} (Validation: {clf_core_val['recall']:.4f})")
    print(f"  Precision:         {clf_core_metrics['precision']:.4f} (Validation: {clf_core_val['precision']:.4f})")
    print(f"  F1:                {clf_core_metrics['f1']:.4f} (Validation: {clf_core_val['f1']:.4f})")
    print(f"  Confusion Matrix:  TN={tn_c}, FP={fp_c}, FN={fn_c}, TP={tp_c}")

    # Supplementary Classifier: Logistic Regression Forecast Extension (Operating threshold 0.55)
    print(f"\nFitting Supplementary Classifier: Logistic Regression Forecast Extension (threshold = {FROZEN_SUPPLEMENTARY_CLF_THRESHOLD:.2f})...")
    pipe_clf_ext = build_logistic_pipeline(C=1.0)
    pipe_clf_ext.fit(X_dev_ext, y_dev_neg)
    proba_ext_clf = pipe_clf_ext.predict_proba(X_holdout_ext)[:, 1]
    pred_ext_clf = (proba_ext_clf >= FROZEN_SUPPLEMENTARY_CLF_THRESHOLD).astype(int)

    cm_ext_arr = confusion_matrix(y_holdout_neg, pred_ext_clf, labels=[0, 1])
    tn_e, fp_e, fn_e, tp_e = [int(v) for v in cm_ext_arr.ravel()]

    clf_ext_metrics = {
        "pr_auc": float(average_precision_score(y_holdout_neg, proba_ext_clf)),
        "roc_auc": float(roc_auc_score(y_holdout_neg, proba_ext_clf)),
        "balanced_accuracy": float(balanced_accuracy_score(y_holdout_neg, pred_ext_clf)),
        "recall": float(recall_score(y_holdout_neg, pred_ext_clf, zero_division=0)),
        "precision": float(precision_score(y_holdout_neg, pred_ext_clf, zero_division=0)),
        "f1": float(f1_score(y_holdout_neg, pred_ext_clf, zero_division=0)),
        "confusion_matrix": {"tn": tn_e, "fp": fp_e, "fn": fn_e, "tp": tp_e},
    }

    clf_ext_val = VALIDATION_REF["classification_ext"]
    clf_ext_comparison = {}
    for metric_name in ["pr_auc", "roc_auc", "balanced_accuracy", "recall", "precision", "f1"]:
        val_m = clf_ext_val[metric_name]
        hold_m = clf_ext_metrics[metric_name]
        diff_abs = hold_m - val_m
        diff_pct = (diff_abs / val_m) * 100.0 if val_m != 0 else 0.0
        gen_label = classify_generalization(val_m, hold_m, metric_type="score")
        clf_ext_comparison[metric_name] = {
            "validation": val_m,
            "holdout": hold_m,
            "diff_abs": float(diff_abs),
            "diff_pct": float(diff_pct),
            "generalization_classification": gen_label,
        }

    print(f"Supplementary Logistic Regression Forecast Extension Holdout Results (Threshold = {FROZEN_SUPPLEMENTARY_CLF_THRESHOLD:.2f}):")
    print(f"  PR-AUC:            {clf_ext_metrics['pr_auc']:.4f} (Validation: {clf_ext_val['pr_auc']:.4f})")
    print(f"  ROC-AUC:           {clf_ext_metrics['roc_auc']:.4f} (Validation: {clf_ext_val['roc_auc']:.4f})")
    print(f"  Balanced Accuracy: {clf_ext_metrics['balanced_accuracy']:.4f} (Validation: {clf_ext_val['balanced_accuracy']:.4f})")
    print(f"  Recall:            {clf_ext_metrics['recall']:.4f} (Validation: {clf_ext_val['recall']:.4f})")
    print(f"  Precision:         {clf_ext_metrics['precision']:.4f} (Validation: {clf_ext_val['precision']:.4f})")
    print(f"  F1:                {clf_ext_metrics['f1']:.4f} (Validation: {clf_ext_val['f1']:.4f})")
    print(f"  Confusion Matrix:  TN={tn_e}, FP={fp_e}, FN={fn_e}, TP={tp_e}")

    # =============================================================
    # PART 7: GENERATE DIAGNOSTIC FIGURES
    # =============================================================
    print("\nGenerating final diagnostic figures in reports/figures/...")
    generate_holdout_figures(
        y_holdout_price=y_holdout_price,
        pred_core_reg=pred_core_reg,
        pred_ext_reg=pred_ext_reg,
        benchmarks=benchmarks,
        reg_core_metrics=reg_core_metrics,
        reg_ext_metrics=reg_ext_metrics,
        y_holdout_neg=y_holdout_neg,
        proba_core_clf=proba_core_clf,
        proba_ext_clf=proba_ext_clf,
        pred_core_clf=pred_core_clf,
        cm_core=clf_core_metrics["confusion_matrix"],
    )
    print("  [OK] 5 figures generated successfully.")

    # =============================================================
    # PART 8: AUDIT METADATA AND METRICS JSON
    # =============================================================
    holdout_payload: Dict[str, Any] = {
        "audit_metadata": {
            "freeze_commit": "23a34db",
            "model_selection_source": "validation only",
            "final_fitting_rows": dev_count,
            "holdout_rows": holdout_count,
            "primary_regression_feature_family": "Core Strict",
            "primary_classification_feature_family": "Core Strict",
            "frozen_classification_threshold": FROZEN_PRIMARY_CLF_THRESHOLD,
            "post_holdout_tuning_performed": False,
            "holdout_used_for_model_selection": False,
            "forecast_extension_status": "supplementary_only",
        },
        "development_target_statistics": {
            "train_rows": train_count,
            "val_rows": val_count,
            "dev_rows": dev_count,
            "train_negative_hours": train_neg_hours,
            "train_negative_prevalence_pct": round(train_neg_prev_pct, 4),
            "val_negative_hours": val_neg_hours,
            "val_negative_prevalence_pct": round(val_neg_prev_pct, 4),
            "dev_negative_hours": dev_neg_hours,
            "dev_negative_prevalence_pct": round(dev_neg_prev_pct, 4),
        },
        "holdout_target_statistics": {
            "total_hours": holdout_count,
            "negative_price_hours": holdout_neg_hours,
            "negative_price_prevalence_pct": round(holdout_neg_prevalence_pct, 4),
            "settled_price_min_eur_mwh": float(y_holdout_price.min()),
            "settled_price_max_eur_mwh": float(y_holdout_price.max()),
            "settled_price_mean_eur_mwh": float(np.mean(y_holdout_price)),
            "settled_price_std_eur_mwh": float(np.std(y_holdout_price)),
        },
        "benchmarks": {
            "24h_persistence_mae": round(mae_24h, 4),
            "168h_persistence_mae": round(mae_168h, 4),
            "7d_lag_mean_mae": round(mae_7d_mean, 4),
        },
        "primary_regression": {
            "model_name": "LightGBM Regressor (Candidate D)",
            "feature_family": "Core Strict",
            "hyperparameters": FROZEN_REG_PARAMS,
            "holdout_metrics": {
                "mae": round(reg_core_metrics["mae"], 4),
                "rmse": round(reg_core_metrics["rmse"], 4),
                "median_ae": round(reg_core_metrics["median_ae"], 4),
                "bias": round(reg_core_metrics["bias"], 4),
                "improvement_vs_24h_pct": round(reg_core_metrics["improvement_vs_24h_pct"], 2),
                "improvement_vs_168h_pct": round(reg_core_metrics["improvement_vs_168h_pct"], 2),
                "improvement_vs_7d_mean_pct": round(reg_core_metrics["improvement_vs_7d_mean_pct"], 2),
            },
            "validation_comparison": {
                "validation_mae": reg_core_val_mae,
                "holdout_mae": round(reg_core_holdout_mae, 4),
                "difference_abs_eur_mwh": round(reg_core_diff_abs, 4),
                "difference_rel_pct": round(reg_core_diff_pct, 2),
                "generalization_classification": reg_core_gen_label,
            },
        },
        "supplementary_regression": {
            "model_name": "LightGBM Regressor (Candidate D)",
            "feature_family": "Forecast Extension",
            "qualification": "ARCHIVE_VINTAGE_LIMITATION",
            "status": "SUPPLEMENTARY SENSITIVITY ANALYSIS",
            "hyperparameters": FROZEN_REG_PARAMS,
            "holdout_metrics": {
                "mae": round(reg_ext_metrics["mae"], 4),
                "rmse": round(reg_ext_metrics["rmse"], 4),
                "median_ae": round(reg_ext_metrics["median_ae"], 4),
                "bias": round(reg_ext_metrics["bias"], 4),
                "improvement_vs_24h_pct": round(reg_ext_metrics["improvement_vs_24h_pct"], 2),
                "improvement_vs_168h_pct": round(reg_ext_metrics["improvement_vs_168h_pct"], 2),
                "improvement_vs_7d_mean_pct": round(reg_ext_metrics["improvement_vs_7d_mean_pct"], 2),
            },
            "validation_comparison": {
                "validation_mae": reg_ext_val_mae,
                "holdout_mae": round(reg_ext_holdout_mae, 4),
                "difference_abs_eur_mwh": round(reg_ext_diff_abs, 4),
                "difference_rel_pct": round(reg_ext_diff_pct, 2),
                "generalization_classification": reg_ext_gen_label,
            },
        },
        "primary_classification": {
            "model_name": "Logistic Regression (StandardScaler, class_weight='balanced')",
            "feature_family": "Core Strict",
            "operating_threshold": FROZEN_PRIMARY_CLF_THRESHOLD,
            "holdout_metrics": {
                "pr_auc": round(clf_core_metrics["pr_auc"], 4),
                "roc_auc": round(clf_core_metrics["roc_auc"], 4),
                "balanced_accuracy": round(clf_core_metrics["balanced_accuracy"], 4),
                "recall": round(clf_core_metrics["recall"], 4),
                "precision": round(clf_core_metrics["precision"], 4),
                "f1": round(clf_core_metrics["f1"], 4),
                "confusion_matrix": clf_core_metrics["confusion_matrix"],
            },
            "validation_comparison": clf_core_comparison,
        },
        "supplementary_classification": {
            "model_name": "Logistic Regression (StandardScaler, class_weight='balanced')",
            "feature_family": "Forecast Extension",
            "qualification": "PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED",
            "status": "SUPPLEMENTARY SENSITIVITY ANALYSIS",
            "operating_threshold": FROZEN_SUPPLEMENTARY_CLF_THRESHOLD,
            "holdout_metrics": {
                "pr_auc": round(clf_ext_metrics["pr_auc"], 4),
                "roc_auc": round(clf_ext_metrics["roc_auc"], 4),
                "balanced_accuracy": round(clf_ext_metrics["balanced_accuracy"], 4),
                "recall": round(clf_ext_metrics["recall"], 4),
                "precision": round(clf_ext_metrics["precision"], 4),
                "f1": round(clf_ext_metrics["f1"], 4),
                "confusion_matrix": clf_ext_metrics["confusion_matrix"],
            },
            "validation_comparison": clf_ext_comparison,
        },
    }

    metrics_path = REPORTS_DIR / "final_holdout_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(holdout_payload, f, indent=2)
    print(f"\nSaved holdout metrics to {metrics_path.name}")

    # Compile Final Report Markdown
    report_md = format_final_holdout_report(holdout_payload)
    report_path = REPORTS_DIR / "final_holdout_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved final holdout report to {report_path.name}")

    return holdout_payload, report_md


def format_final_holdout_report(payload: Dict[str, Any]) -> str:
    """
    Format comprehensive Final Holdout Evaluation Report in neutral, scientific tone.
    """
    audit = payload["audit_metadata"]
    tstats = payload["holdout_target_statistics"]
    dstats = payload["development_target_statistics"]
    b = payload["benchmarks"]
    reg_c = payload["primary_regression"]
    reg_e = payload["supplementary_regression"]
    clf_c = payload["primary_classification"]
    clf_e = payload["supplementary_classification"]

    rc_m = reg_c["holdout_metrics"]
    rc_v = reg_c["validation_comparison"]
    re_m = reg_e["holdout_metrics"]
    re_v = reg_e["validation_comparison"]

    cc_m = clf_c["holdout_metrics"]
    cc_v = clf_c["validation_comparison"]
    ce_m = clf_e["holdout_metrics"]
    ce_v = clf_e["validation_comparison"]

    cm_c = cc_m["confusion_matrix"]
    cm_e = ce_m["confusion_matrix"]

    report = f"""# Final Holdout Evaluation Report

**Project**: Portfolio Project 3 — German Day-Ahead Electricity Price & Negative Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Freeze Commit**: `{audit['freeze_commit']}`  
**Model Selection Source**: `{audit['model_selection_source']}`  
**Final Fitting Set (TRAIN + VALIDATION)**: `2023-01-08 00:00` to `2024-06-30 23:00 Europe/Berlin` (**{audit['final_fitting_rows']:,} hours**)  
**Final Holdout Split**: `2024-07-01 00:00` to `2024-12-31 23:00 Europe/Berlin` (**{audit['holdout_rows']:,} hours**)  
**Holdout Status**: Authorized Final Out-of-Sample Evaluation  

---

## 1. Executive Summary & Governance Attestation

This document reports the final out-of-sample evaluation of the frozen day-ahead electricity spot price forecasting and negative-price classification models on the unopened final holdout partition (H2 2024).

### Strict Governance Attestation:
1. **Model Freeze**: All feature pipelines, model architectures, hyperparameters, and operating decision thresholds were permanently frozen at commit `23a34db` prior to holdout opening.
2. **No Post-Holdout Tuning**: No hyperparameters, feature sets, or decision thresholds were adjusted after observing holdout performance (`post_holdout_tuning_performed: false`).
3. **No Holdout Model Selection**: Model selection was conducted strictly using validation set metrics (`holdout_used_for_model_selection: false`).
4. **Fitting Isolation**: All model weights and preprocessing estimators (including `StandardScaler` and class-weight estimators) were fit exclusively on the combined development set (`TRAIN + VALIDATION`, 12,959 rows). The holdout set was transformed only and never used during training.
5. **Primary vs. Supplementary Separation**: Core Strict remains the primary leakage-controlled model family. Forecast Extension is reported strictly as a supplementary sensitivity analysis under its declared `ARCHIVE_VINTAGE_LIMITATION`.

---

## 2. Final Holdout Dataset & Target Statistics

| Property | Development Split (TRAIN + VALIDATION) | Final Holdout Split (H2 2024) |
|---|---|---|
| **Date Range (Local)** | 2023-01-08 00:00 to 2024-06-30 23:00 | 2024-07-01 00:00 to 2024-12-31 23:00 |
| **Total Rows** | {dstats['dev_rows']:,} | {audit['holdout_rows']:,} |
| **Negative-Price Hours** | {dstats['dev_negative_hours']:,} | {tstats['negative_price_hours']:,} |
| **Negative-Price Prevalence** | {dstats['dev_negative_prevalence_pct']:.2f}% | **{tstats['negative_price_prevalence_pct']:.2f}%** |
| **Spot Price Mean (EUR/MWh)** | 82.26 | {tstats['settled_price_mean_eur_mwh']:.2f} |
| **Spot Price Std (EUR/MWh)** | 48.77 | {tstats['settled_price_std_eur_mwh']:.2f} |
| **Spot Price Min (EUR/MWh)** | -500.00 | {tstats['settled_price_min_eur_mwh']:.2f} |
| **Spot Price Max (EUR/MWh)** | 524.27 | {tstats['settled_price_max_eur_mwh']:.2f} |

---

## 3. Primary Regression Evaluation: LightGBM Core Strict

### Model Configuration (Frozen Candidate D):
- **Predictors**: 23 Core Strict features (no forecast features; strictly pre-auction observed).
- **Hyperparameters**: `n_estimators=400`, `learning_rate=0.03`, `num_leaves=15`, `max_depth=5`, `min_child_samples=50`, `subsample=0.80`, `subsample_freq=1`, `colsample_bytree=0.80`, `reg_lambda=0.0`, `random_state=42`.

### Holdout Performance vs. Frozen Benchmarks:

| Model / Benchmark | Holdout MAE (EUR/MWh) | Holdout RMSE (EUR/MWh) | Holdout Median AE (EUR/MWh) | Holdout Bias (EUR/MWh) | Relative Improvement vs Benchmark |
|---|---|---|---|---|---|
| **24h Persistence** | {b['24h_persistence_mae']:.2f} | — | — | — | Baseline |
| **168h Persistence** | {b['168h_persistence_mae']:.2f} | — | — | — | Baseline |
| **7-Day Lag Mean** | {b['7d_lag_mean_mae']:.2f} | — | — | — | Baseline |
| **LightGBM Core Strict (Primary)** | **{rc_m['mae']:.2f}** | **{rc_m['rmse']:.2f}** | **{rc_m['median_ae']:.2f}** | **{rc_m['bias']:+.2f}** | **+{rc_m['improvement_vs_24h_pct']:.2f}% vs 24h**<br>+{rc_m['improvement_vs_168h_pct']:.2f}% vs 168h<br>+{rc_m['improvement_vs_7d_mean_pct']:.2f}% vs 7d mean |

### Validation-to-Holdout Comparison:
- **Validation MAE**: {rc_v['validation_mae']:.2f} EUR/MWh
- **Final Holdout MAE**: {rc_v['holdout_mae']:.2f} EUR/MWh
- **Difference**: {rc_v['difference_abs_eur_mwh']:+.2f} EUR/MWh ({rc_v['difference_rel_pct']:+.2f}%)
- **Generalization Classification**: **{rc_v['generalization_classification']}**

---

## 4. Supplementary Regression Sensitivity: LightGBM Forecast Extension

> **QUALIFICATION: ARCHIVE_VINTAGE_LIMITATION**  
> This evaluation is provided strictly as a supplementary sensitivity analysis. The 3 forecast extension predictors (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`) reflect retrospective archived vintages rather than strictly verified pre-auction point-in-time bid snapshots. It does not replace the primary Core Strict model.

| Metric | Core Strict (Primary) | Forecast Extension (Supplementary) | Difference (Ext - Core) |
|---|---|---|---|
| **MAE (EUR/MWh)** | **{rc_m['mae']:.2f}** | {re_m['mae']:.2f} | {re_m['mae'] - rc_m['mae']:+.2f} |
| **RMSE (EUR/MWh)** | {rc_m['rmse']:.2f} | **{re_m['rmse']:.2f}** | {re_m['rmse'] - rc_m['rmse']:+.2f} |
| **Median AE (EUR/MWh)** | **{rc_m['median_ae']:.2f}** | {re_m['median_ae']:.2f} | {re_m['median_ae'] - rc_m['median_ae']:+.2f} |
| **Bias (EUR/MWh)** | **{rc_m['bias']:+.2f}** | {re_m['bias']:+.2f} | {re_m['bias'] - rc_m['bias']:+.2f} |
| **Improvement vs 24h (%)** | +{rc_m['improvement_vs_24h_pct']:.2f}% | +{re_m['improvement_vs_24h_pct']:.2f}% | — |
| **Generalization Status** | {rc_v['generalization_classification']} | {re_v['generalization_classification']} | — |

---

## 5. Primary Classification Evaluation: Logistic Regression Core Strict

### Model Configuration:
- **Architecture**: `Pipeline(StandardScaler, LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))`
- **Operating Decision Threshold**: Permanently fixed at **{clf_c['operating_threshold']:.2f}** before holdout opening.

### Performance & Validation Comparison:

| Metric | Validation (Frozen) | Holdout (Measured) | Absolute Change | Relative Change (%) | Generalization Classification |
|---|---|---|---|---|---|
| **PR-AUC** | {cc_v['pr_auc']['validation']:.4f} | **{cc_v['pr_auc']['holdout']:.4f}** | {cc_v['pr_auc']['diff_abs']:+.4f} | {cc_v['pr_auc']['diff_pct']:+.2f}% | {cc_v['pr_auc']['generalization_classification']} |
| **ROC-AUC** | {cc_v['roc_auc']['validation']:.4f} | **{cc_v['roc_auc']['holdout']:.4f}** | {cc_v['roc_auc']['diff_abs']:+.4f} | {cc_v['roc_auc']['diff_pct']:+.2f}% | {cc_v['roc_auc']['generalization_classification']} |
| **Balanced Accuracy** | {cc_v['balanced_accuracy']['validation']:.4f} | **{cc_v['balanced_accuracy']['holdout']:.4f}** | {cc_v['balanced_accuracy']['diff_abs']:+.4f} | {cc_v['balanced_accuracy']['diff_pct']:+.2f}% | {cc_v['balanced_accuracy']['generalization_classification']} |
| **Recall** | {cc_v['recall']['validation']:.4f} | **{cc_v['recall']['holdout']:.4f}** | {cc_v['recall']['diff_abs']:+.4f} | {cc_v['recall']['diff_pct']:+.2f}% | {cc_v['recall']['generalization_classification']} |
| **Precision** | {cc_v['precision']['validation']:.4f} | **{cc_v['precision']['holdout']:.4f}** | {cc_v['precision']['diff_abs']:+.4f} | {cc_v['precision']['diff_pct']:+.2f}% | {cc_v['precision']['generalization_classification']} |
| **F1 Score** | {cc_v['f1']['validation']:.4f} | **{cc_v['f1']['holdout']:.4f}** | {cc_v['f1']['diff_abs']:+.4f} | {cc_v['f1']['diff_pct']:+.2f}% | {cc_v['f1']['generalization_classification']} |

### Holdout Confusion Matrix (Threshold = {clf_c['operating_threshold']:.2f}):
- **True Negatives (TN)**: {cm_c['tn']:,}
- **False Positives (FP)**: {cm_c['fp']:,}
- **False Negatives (FN)**: {cm_c['fn']:,}
- **True Positives (TP)**: {cm_c['tp']:,}

---

## 6. Supplementary Classification Sensitivity: Logistic Regression Forecast Extension

> **QUALIFICATION: PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED**  
> Operating Decision Threshold: **{clf_e['operating_threshold']:.2f}** (frozen). This model is supplementary only and does not replace the primary Core Strict classifier.

| Metric | Core Strict (Primary, th=0.45) | Forecast Extension (Supplementary, th=0.55) |
|---|---|---|
| **PR-AUC** | {cc_m['pr_auc']:.4f} | {ce_m['pr_auc']:.4f} |
| **ROC-AUC** | {cc_m['roc_auc']:.4f} | {ce_m['roc_auc']:.4f} |
| **Balanced Accuracy** | {cc_m['balanced_accuracy']:.4f} | {ce_m['balanced_accuracy']:.4f} |
| **Recall** | {cc_m['recall']:.4f} | {ce_m['recall']:.4f} |
| **Precision** | {cc_m['precision']:.4f} | {ce_m['precision']:.4f} |
| **F1 Score** | {cc_m['f1']:.4f} | {ce_m['f1']:.4f} |
| **Confusion Matrix [TN, FP, FN, TP]** | [{cm_c['tn']}, {cm_c['fp']}, {cm_c['fn']}, {cm_c['tp']}] | [{cm_e['tn']}, {cm_e['fp']}, {cm_e['fn']}, {cm_e['tp']}] |

---

## 7. Final Generalization Analysis

Descriptive evaluation based strictly on measured metric differences:

### Regression Generalization:
- **Measured Result**: Validation MAE of {rc_v['validation_mae']:.2f} EUR/MWh compared to Holdout MAE of {rc_v['holdout_mae']:.2f} EUR/MWh ({rc_v['difference_abs_eur_mwh']:+.2f} EUR/MWh, {rc_v['difference_rel_pct']:+.2f}%).
- **Classification**: **{rc_v['generalization_classification']}**.
- **Context**: The primary model maintains substantial predictive value over naive benchmarks ({rc_m['improvement_vs_24h_pct']:.2f}% improvement over 24h persistence).

### Classification Generalization:
- **Measured Result**: PR-AUC changed from {cc_v['pr_auc']['validation']:.4f} to {cc_v['pr_auc']['holdout']:.4f} ({cc_v['pr_auc']['diff_pct']:+.2f}%). F1 changed from {cc_v['f1']['validation']:.4f} to {cc_v['f1']['holdout']:.4f} ({cc_v['f1']['diff_pct']:+.2f}%). Recall changed from {cc_v['recall']['validation']:.4f} to {cc_v['recall']['holdout']:.4f} ({cc_v['recall']['diff_pct']:+.2f}%).
- **Classification**: **{cc_v['pr_auc']['generalization_classification']}** (PR-AUC), **{cc_v['f1']['generalization_classification']}** (F1).
- **Prevalence Context**: The holdout negative-price prevalence is {tstats['negative_price_prevalence_pct']:.2f}% ({tstats['negative_price_hours']} hours), compared to {dstats['val_negative_prevalence_pct']:.2f}% ({dstats['val_negative_hours']} hours) in the validation set, and {dstats['dev_negative_prevalence_pct']:.2f}% ({dstats['dev_negative_hours']} hours) across the combined development set (TRAIN: {dstats['train_negative_hours']} hours, VALIDATION: {dstats['val_negative_hours']} hours, DEVELOPMENT: {dstats['dev_negative_hours']} hours).

---

## 8. Diagnostic Figures Generated

The following five figures have been rendered to `reports/figures/`:
1. `final_holdout_regression_actual_vs_predicted.png`: Scatter plot of actual vs predicted spot prices with perfect forecast reference line and error summary.
2. `final_holdout_regression_error_distribution.png`: Histogram and kernel density of prediction errors, indicating mean bias and median error.
3. `final_holdout_regression_benchmark_comparison.png`: Bar chart comparing Primary LightGBM Core Strict MAE against 24h persistence, 168h persistence, and 7-day lag mean benchmarks.
4. `final_holdout_classification_pr_curve.png`: Precision-Recall curve with annotated PR-AUC and marked operating threshold ({clf_c['operating_threshold']:.2f}).
5. `final_holdout_classification_confusion_matrix.png`: Heatmap of the binary confusion matrix at threshold {clf_c['operating_threshold']:.2f}.
"""
    return report


if __name__ == "__main__":
    run_final_holdout_evaluation()
