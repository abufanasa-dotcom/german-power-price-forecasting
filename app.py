"""Streamlit portfolio dashboard for German Day-Ahead Electricity Price Forecasting.

This application is a read-only presentation layer displaying frozen, committed
scientific results, benchmark comparisons, and holdout evaluations.
It contains no model training, parameter tuning, or live inference logic.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.dashboard import (
    get_figure_path,
    get_primary_kpis,
    get_project_root,
    load_final_metrics,
    load_model_freeze,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German Day-Ahead Electricity Price Forecasting",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. Data Loading with Safety Handling
# -----------------------------------------------------------------------------
project_root = get_project_root()

try:
    metrics = load_final_metrics(project_root)
    freeze_info = load_model_freeze(project_root)
    kpis = get_primary_kpis(metrics)
except Exception as exc:
    st.error(f"Failed to load project metrics or freeze artifacts: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Hero / Header
# -----------------------------------------------------------------------------
st.title("⚡ German Day-Ahead Electricity Price Forecasting")
st.subheader(
    "Leakage-aware hourly price forecasting and negative-price early warning "
    "for the German/Luxembourg day-ahead electricity market."
)

st.markdown(
    """
    **Project Scope & Context**:
    - **Data Source**: Official SMARD platform of the German Federal Network Agency (*Bundesnetzagentur*).
    - **Timeline**: 2023–2024 at hourly resolution (17,376 usable observations).
    - **Evaluation**: Out-of-sample evaluation on an untouched six-month holdout (H2 2024: July 1 – December 31, 2024).
    - **Governance**: All models, hyperparameters, and decision thresholds were frozen in Git (`23a34db`) prior to opening the holdout.
    - *Note: This project is an analytical research pipeline; no claim of production readiness or live trading profitability is made.*
    """
)

st.divider()

# -----------------------------------------------------------------------------
# 4. Top KPI Cards (Primary Core Strict Results)
# -----------------------------------------------------------------------------
st.markdown("### Primary Holdout Performance (Core Strict)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Regression MAE",
        value=kpis["regression_mae"],
        help="Primary LightGBM Candidate D on unseen H2 2024 holdout",
    )

with col2:
    st.metric(
        label="Improvement vs 24h",
        value=kpis["improvement_vs_24h"],
        help="MAE reduction relative to 24h persistence baseline (32.73 EUR/MWh)",
    )

with col3:
    st.metric(
        label="Negative-Price Recall",
        value=kpis["negative_recall"],
        help="Captured 196 of 233 negative-price hours (frozen threshold = 0.45)",
    )

with col4:
    st.metric(
        label="Negative-Price PR-AUC",
        value=kpis["negative_pr_auc"],
        help="Precision-Recall AUC of primary Logistic Regression classifier",
    )

with col5:
    st.metric(
        label="Final Holdout",
        value=kpis["final_holdout_hours"],
        help="Unseen chronological test period: 2024-07-01 to 2024-12-31",
    )

st.divider()

# -----------------------------------------------------------------------------
# 5. Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_price, tab_negative, tab_governance, tab_details = st.tabs(
    [
        "📋 Overview",
        "📈 Price Forecasting",
        "⚠️ Negative-Price Detection",
        "🛡️ Methodology & Governance",
        "⚙️ Project Details",
    ]
)

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    st.markdown("### Executive Summary & Objectives")
    st.markdown(
        """
        In the German/Luxembourg day-ahead power market (EPEX SPOT), contracts clear daily at
        **12:00 Europe/Berlin local market time on D-1 (CET/CEST as applicable)** for delivery across the 24 hours of day $D$.
        This project establishes two distinct modeling objectives under strict pre-auction operational constraints:
        1. **Continuous Price Forecasting**: Accurately predict hourly day-ahead settlement prices (EUR/MWh) using gradient boosting.
        2. **Negative-Price Risk Warning**: Flag extreme negative-price hours (EUR/MWh $< 0.00$) to provide operational early warning.
        """
    )

    st.markdown("#### Chronological Dataset Partitioning")
    st.markdown(
        "Data is partitioned chronologically to avoid lookahead bias inherent in randomized cross-validation:"
    )

    dev_stats = metrics.get("development_target_statistics", {})
    holdout_stats = metrics.get("holdout_target_statistics", {})

    split_data = [
        {
            "Split": "TRAIN",
            "Period (Local Time Europe/Berlin)": "2023-01-08 00:00 to 2023-12-31 23:00",
            "Row Count": f"{dev_stats.get('train_rows', 8592):,}",
            "Negative Hours": dev_stats.get("train_negative_hours", 287),
            "Prevalence": f"{dev_stats.get('train_negative_prevalence_pct', 3.34):.2f}%",
            "Purpose": "Model fitting & scaling parameter estimation",
        },
        {
            "Split": "VALIDATION",
            "Period (Local Time Europe/Berlin)": "2024-01-01 00:00 to 2024-06-30 23:00",
            "Row Count": f"{dev_stats.get('val_rows', 4367):,}",
            "Negative Hours": dev_stats.get("val_negative_hours", 224),
            "Prevalence": f"{dev_stats.get('val_negative_prevalence_pct', 5.13):.2f}%",
            "Purpose": "Model selection & threshold optimization",
        },
        {
            "Split": "FINAL HOLDOUT",
            "Period (Local Time Europe/Berlin)": "2024-07-01 00:00 to 2024-12-31 23:00",
            "Row Count": f"{holdout_stats.get('total_hours', 4417):,}",
            "Negative Hours": holdout_stats.get("negative_price_hours", 233),
            "Prevalence": f"{holdout_stats.get('negative_price_prevalence_pct', 5.28):.2f}%",
            "Purpose": "Single out-of-sample final evaluation",
        },
    ]
    st.dataframe(pd.DataFrame(split_data), use_container_width=True, hide_index=True)

    st.markdown("#### Selected Primary Models & Governance Summary")
    ov_col1, ov_col2 = st.columns(2)
    with ov_col1:
        with st.container(border=True):
            st.markdown("#### Primary Regression Model")
            st.markdown(
                "- **Algorithm**: LightGBM Regressor (Candidate D: `max_depth=5`, `num_leaves=15`, `learning_rate=0.03`)\n"
                "- **Feature Family**: Core Strict (23 pre-auction predictors)\n"
                "- **Validation MAE**: 18.97 EUR/MWh\n"
                "- **Final Holdout MAE**: **26.87 EUR/MWh** (+17.91% vs 24h persistence)"
            )
    with ov_col2:
        with st.container(border=True):
            st.markdown("#### Primary Classification Model")
            st.markdown(
                "- **Algorithm**: Logistic Regression (`StandardScaler`, `class_weight='balanced'`)\n"
                "- **Operating Threshold**: 0.45 (selected on validation data for recall $\\ge 0.75$)\n"
                "- **Validation PR-AUC**: 0.5914\n"
                "- **Final Holdout Recall**: **84.12%** (196 / 233 negative hours captured; PR-AUC: 0.4538)"
            )

    st.markdown(
        """
        > **Key Methodological Principle**:
        > Model selection and threshold tuning were finalized entirely on VALIDATION data and permanently frozen in Git (`23a34db`).
        > The final holdout was scored exactly once at commit `c29d190`; **no post-holdout tuning was performed**.
        """
    )

# =============================================================================
# TAB 2: PRICE FORECASTING
# =============================================================================
with tab_price:
    st.markdown("### Continuous Price Forecasting (Regression)")
    st.markdown(
        "**Primary Model**: LightGBM Regressor (Candidate D) — Core Strict (23 predictors)"
    )

    reg_metrics = metrics.get("primary_regression", {}).get("holdout_metrics", {})
    bm_metrics = metrics.get("benchmarks", {})

    rf_col1, rf_col2, rf_col3, rf_col4 = st.columns(4)
    rf_col1.metric("MAE", f"{reg_metrics.get('mae', 26.87):.2f} EUR/MWh")
    rf_col2.metric("RMSE", f"{reg_metrics.get('rmse', 45.67):.2f} EUR/MWh")
    rf_col3.metric("Median AE", f"{reg_metrics.get('median_ae', 18.94):.2f} EUR/MWh")
    rf_col4.metric("Bias", f"{reg_metrics.get('bias', -2.31):.2f} EUR/MWh")

    st.markdown("#### Benchmark Comparison (Final Holdout)")
    benchmarks_df = pd.DataFrame(
        [
            {
                "Model / Benchmark": "24h Persistence",
                "Type": "Heuristic Baseline",
                "Holdout MAE (EUR/MWh)": f"{bm_metrics.get('24h_persistence_mae', 32.73):.2f}",
                "Improvement": "Baseline (0.00%)",
            },
            {
                "Model / Benchmark": "168h Persistence (Weekly)",
                "Type": "Heuristic Baseline",
                "Holdout MAE (EUR/MWh)": f"{bm_metrics.get('168h_persistence_mae', 39.05):.2f}",
                "Improvement": "Baseline",
            },
            {
                "Model / Benchmark": "7-Day Lag Mean",
                "Type": "Heuristic Baseline",
                "Holdout MAE (EUR/MWh)": f"{bm_metrics.get('7d_lag_mean_mae', 33.70):.2f}",
                "Improvement": "Baseline",
            },
            {
                "Model / Benchmark": "LightGBM Core Strict (Primary)",
                "Type": "Gradient Boosted Trees",
                "Holdout MAE (EUR/MWh)": f"{reg_metrics.get('mae', 26.87):.2f}",
                "Improvement": f"+{reg_metrics.get('improvement_vs_24h_pct', 17.91):.2f}% vs 24h",
            },
        ]
    )
    st.dataframe(benchmarks_df, use_container_width=True, hide_index=True)

    st.markdown("#### Diagnostic Visualizations")
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        try:
            fig_bm = get_figure_path(
                "final_holdout_regression_benchmark_comparison.png", project_root
            )
            st.image(str(fig_bm), caption="Final Holdout: Regression MAE vs Benchmarks", use_container_width=True)
        except Exception as err:
            st.warning(f"Benchmark figure unavailable: {err}")

    with diag_col2:
        try:
            fig_pred = get_figure_path(
                "final_holdout_regression_actual_vs_predicted.png", project_root
            )
            st.image(str(fig_pred), caption="Final Holdout: Actual vs Predicted Prices", use_container_width=True)
        except Exception as err:
            st.warning(f"Prediction figure unavailable: {err}")

    with st.expander("🔍 View Residual Error Distribution"):
        try:
            fig_err = get_figure_path(
                "final_holdout_regression_error_distribution.png", project_root
            )
            st.image(str(fig_err), caption="Holdout Residual Distribution", use_container_width=True)
        except Exception as err:
            st.warning(f"Residual distribution figure unavailable: {err}")

    st.markdown("#### Out-of-Sample Generalization")
    st.markdown(
        """
        - **Validation MAE**: 18.97 EUR/MWh (H1 2024)
        - **Final Holdout MAE**: 26.87 EUR/MWh (H2 2024)
        - **Absolute Difference**: +7.90 EUR/MWh (+41.65%)

        *Performance weakened on the final unseen period; no post-holdout model changes were made.*
        """
    )

    with st.expander("Supplementary Forecast Extension Results"):
        st.markdown(
            """
            `ARCHIVE_VINTAGE_LIMITATION` | `PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED`

            - **Feature Family**: Forecast Extension (26 predictors: Core Strict + 3 archived load forecast features)
            - **Holdout MAE**: 26.55 EUR/MWh (vs. 26.87 Core Strict)
            - **Improvement vs 24h**: +18.89%

            *Note: These retrospective forecast features are evaluated strictly as supplementary sensitivity analysis and do not replace the primary Core Strict conclusions.*
            """
        )

# =============================================================================
# TAB 3: NEGATIVE-PRICE DETECTION
# =============================================================================
with tab_negative:
    st.markdown("### Negative-Price Risk Detection (Classification)")
    st.markdown(
        "**Primary Model**: Logistic Regression (`StandardScaler`, `class_weight='balanced'`) — Core Strict"
    )

    clf_metrics = metrics.get("primary_classification", {}).get("holdout_metrics", {})
    cm = clf_metrics.get("confusion_matrix", {})

    np_col1, np_col2, np_col3, np_col4 = st.columns(4)
    np_col1.metric("Frozen Threshold", "0.45")
    np_col2.metric("Negative-Price Hours", f"{holdout_stats.get('negative_price_hours', 233)} / {holdout_stats.get('total_hours', 4417)}")
    np_col3.metric("Class Prevalence", f"{holdout_stats.get('negative_price_prevalence_pct', 5.28):.2f}%")
    np_col4.metric("Recall (Sensitivity)", f"{clf_metrics.get('recall', 0.8412) * 100:.2f}%")

    st.markdown("#### Holdout Classification Metrics")
    clf_table = [
        {"Metric": "PR-AUC", "Validation (Frozen)": "0.5914", "Holdout (Measured)": f"{clf_metrics.get('pr_auc', 0.4538):.4f}", "Interpretation": "Primary metric for imbalanced event detection"},
        {"Metric": "ROC-AUC", "Validation (Frozen)": "0.9360", "Holdout (Measured)": f"{clf_metrics.get('roc_auc', 0.9019):.4f}", "Interpretation": "Overall ranking discrimination"},
        {"Metric": "Balanced Accuracy", "Validation (Frozen)": "0.8505", "Holdout (Measured)": f"{clf_metrics.get('balanced_accuracy', 0.8023):.4f}", "Interpretation": "Unweighted average of recall per class"},
        {"Metric": "Recall", "Validation (Frozen)": "0.7857", "Holdout (Measured)": f"{clf_metrics.get('recall', 0.8412) * 100:.2f}%", "Interpretation": "Proportion of negative hours detected (196 / 233)"},
        {"Metric": "Precision", "Validation (Frozen)": "0.3340", "Holdout (Measured)": f"{clf_metrics.get('precision', 0.1653) * 100:.2f}%", "Interpretation": "Proportion of positive alerts that were negative hours"},
        {"Metric": "F1 Score", "Validation (Frozen)": "0.4687", "Holdout (Measured)": f"{clf_metrics.get('f1', 0.2763):.4f}", "Interpretation": "Harmonic mean of precision and recall"},
    ]
    st.dataframe(pd.DataFrame(clf_table), use_container_width=True, hide_index=True)

    st.markdown("#### Confusion Matrix (Threshold = 0.45)")
    cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
    cm_col1.metric("True Negatives (TN)", f"{cm.get('tn', 3194):,}")
    cm_col2.metric("False Positives (FP)", f"{cm.get('fp', 990):,}")
    cm_col3.metric("False Negatives (FN)", f"{cm.get('fn', 37):,}")
    cm_col4.metric("True Positives (TP)", f"{cm.get('tp', 196):,}")

    st.markdown("#### Classification Visualizations")
    cfig_col1, cfig_col2 = st.columns(2)
    with cfig_col1:
        try:
            fig_pr = get_figure_path(
                "final_holdout_classification_pr_curve.png", project_root
            )
            st.image(str(fig_pr), caption="Precision-Recall Curve (Holdout)", use_container_width=True)
        except Exception as err:
            st.warning(f"PR curve figure unavailable: {err}")

    with cfig_col2:
        try:
            fig_cm = get_figure_path(
                "final_holdout_classification_confusion_matrix.png", project_root
            )
            st.image(str(fig_cm), caption="Confusion Matrix (Threshold 0.45)", use_container_width=True)
        except Exception as err:
            st.warning(f"Confusion matrix figure unavailable: {err}")

    st.markdown(
        """
        > **Operational Interpretation**:
        > The operating threshold was selected on validation data to prioritize negative-price event recall.
        > On the final holdout, 196 of 233 negative hours were detected, with a substantial false-positive trade-off
        > (990 false alarms, yielding 16.53% precision). The classifier serves as an operational early-warning screen,
        > not an autonomous trading trigger.
        """
    )

    with st.expander("Supplementary Forecast Extension Results"):
        st.markdown(
            """
            `ARCHIVE_VINTAGE_LIMITATION` | `PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED`

            - **Feature Family**: Forecast Extension (26 predictors)
            - **Operating Threshold**: 0.55
            - **Holdout PR-AUC**: 0.4941
            - **Holdout Recall**: 82.83% (193 / 233)
            - **Holdout Precision**: 20.73%
            - **Holdout F1**: 0.3316
            - **Confusion Matrix**: TN=3,446, FP=738, FN=40, TP=193

            *Note: These supplementary results do not replace the primary Core Strict conclusions.*
            """
        )

# =============================================================================
# TAB 4: METHODOLOGY & GOVERNANCE
# =============================================================================
with tab_governance:
    st.markdown("### Methodology, Data Leakage Controls & Governance")

    st.markdown("#### 1. Pre-Auction Information Boundary")
    st.markdown(
        """
        In the German day-ahead market, the auction closes at **12:00 Europe/Berlin local market time on D-1 (CET/CEST as applicable)**.
        Any feature constructed for delivery day $D$ must strictly respect this cutoff:
        - **Settled Price Lags**: Minimum lag is 24 hours (settled at 12:00 on D-1 for delivery on D).
        - **Actual Load Lags**: Available actual load figures are lagged at least 24 hours.
        - **Calendar & Solar Proxies**: Derived from deterministic astronomical and calendar rules.
        """
    )

    st.markdown("#### 2. Two-Tier Feature Hierarchy")
    st.markdown(
        """
        - **Core Strict (Primary)**: 23 predictors strictly conforming to verifiable pre-auction publication timing.
        - **Forecast Extension (Supplementary)**: 26 predictors including archived day-ahead grid load forecasts.
          Qualified under `ARCHIVE_VINTAGE_LIMITATION` because retrospective archives do not guarantee immutable point-in-time publication vintage.
        """
    )

    st.markdown("#### 3. Audit Trail & Checkpoint Commits")
    audit_data = [
        {"Milestone": "Data Ingestion & Validation", "Artifact": "reports/data_validation_report.md", "Status": "Verified (71 automated tests passing: 62 pipeline + 9 dashboard)"},
        {"Milestone": "Final Model Freeze", "Artifact": "reports/final_model_freeze.md", "Status": "Permanently frozen at commit 23a34db"},
        {"Milestone": "Final Holdout Evaluation", "Artifact": "reports/final_holdout_report.md", "Status": "Scored once at commit c29d190 (no post-holdout tuning)"},
        {"Milestone": "Portfolio Project README", "Artifact": "README.md", "Status": "Committed at 172170a"},
    ]
    st.dataframe(pd.DataFrame(audit_data), use_container_width=True, hide_index=True)

    st.markdown("#### 4. Data Leakage Checklist")
    st.markdown(
        """
        - [x] **Chronological Splitting**: Prohibited random k-fold shuffling.
        - [x] **Pre-Auction Gate Discipline**: Zero lookahead past 12:00 Europe/Berlin local market time on D-1.
        - [x] **Data Isolation**: Preprocessing transformers fitted strictly on development data (`TRAIN + VALIDATION`).
        - [x] **Pre-Holdout Git Freeze**: Architectures and thresholds frozen before opening holdout split.
        - [x] **Zero Post-Holdout Tuning**: `post_holdout_tuning_performed: false`.
        """
    )

# =============================================================================
# TAB 5: PROJECT DETAILS
# =============================================================================
with tab_details:
    st.markdown("### Technical Architecture & Limitations")

    det_col1, det_col2 = st.columns(2)
    with det_col1:
        st.markdown("#### Technical Stack")
        st.markdown(
            """
            - **Language & Runtime**: Python 3.11
            - **Data Manipulation**: pandas, NumPy, PyArrow
            - **Machine Learning**: scikit-learn, LightGBM
            - **Visualization**: Matplotlib
            - **Testing & QA**: pytest (71 automated tests passing: 62 pipeline + 9 dashboard)
            - **Dashboard Presentation**: Streamlit
            - **Version Control**: Git
            """
        )

    with det_col2:
        st.markdown("#### Data Source & Pipeline")
        st.markdown(
            """
            - **Data Source**: SMARD API / Bundesnetzagentur
            - **Bidding Zone**: Germany / Luxembourg (DE-LU)
            - **Coverage**: 2023-01-01 to 2024-12-31 (Hourly)
            - **Primary Regression**: LightGBM Regressor (Candidate D)
            - **Primary Classification**: Logistic Regression (Balanced)
            """
        )

    st.markdown("#### Project Workflow")
    st.code(
        """
        1. Ingestion: SMARD API data retrieval -> data/raw/
        2. Validation: Timestamp continuity, DST verification -> data/processed/
        3. Features: Pre-auction lag generation & temporal splits -> features_*.parquet
        4. Baselines & ML: Heuristic benchmarks, Ridge, Logistic, LightGBM on TRAIN/VAL
        5. Tuning & Refinement: Regression Candidate D selection & classification threshold 0.45
        6. Freeze: Pre-holdout audit trail permanently committed (23a34db)
        7. Holdout: One-time scoring on unseen H2 2024 holdout (c29d190)
        """,
        language="text",
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
        - **Dataset Scope**: The dataset covers calendar years 2023–2024 only; longer-term multi-year structural regime shifts are not modeled.
        - **Out-of-Sample Degradation**: Final holdout performance was weaker than validation performance (regression MAE increased from 18.97 to 26.87 EUR/MWh). The project does not attribute this degradation to a specific cause.
        - **False-Positive Trade-Off**: The negative-price classifier prioritizes operational event recall (84.12%) and consequently produces a substantial number of false positives (16.53% precision).
        - **Strict Boundary Exclusion**: Core Strict intentionally excludes features whose publication timing could violate the pre-auction information boundary.
        - **Retrospective Forecast Data**: The Forecast Extension uses retrospective forecast data and is treated strictly as supplementary sensitivity analysis because immutable historical point-in-time publication vintage is not guaranteed.
        - **Non-Production Scope**: No claim of production readiness or live trading profitability is made; the pipeline is designed for methodological benchmarking and research evaluation.
        """
    )
