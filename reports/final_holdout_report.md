# Final Holdout Evaluation Report

**Project**: Portfolio Project 3 — German Day-Ahead Electricity Price & Negative Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Freeze Commit**: `23a34db`  
**Model Selection Source**: `validation only`  
**Final Fitting Set (TRAIN + VALIDATION)**: `2023-01-08 00:00` to `2024-06-30 23:00 Europe/Berlin` (**12,959 hours**)  
**Final Holdout Split**: `2024-07-01 00:00` to `2024-12-31 23:00 Europe/Berlin` (**4,417 hours**)  
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
| **Total Rows** | 12,959 | 4,417 |
| **Negative-Price Hours** | 511 | 233 |
| **Negative-Price Prevalence** | 3.94% | **5.28%** |
| **Spot Price Mean (EUR/MWh)** | 82.26 | 89.32 |
| **Spot Price Std (EUR/MWh)** | 48.77 | 62.79 |
| **Spot Price Min (EUR/MWh)** | -500.00 | -73.96 |
| **Spot Price Max (EUR/MWh)** | 524.27 | 936.28 |

---

## 3. Primary Regression Evaluation: LightGBM Core Strict

### Model Configuration (Frozen Candidate D):
- **Predictors**: 23 Core Strict features (no forecast features; strictly pre-auction observed).
- **Hyperparameters**: `n_estimators=400`, `learning_rate=0.03`, `num_leaves=15`, `max_depth=5`, `min_child_samples=50`, `subsample=0.80`, `subsample_freq=1`, `colsample_bytree=0.80`, `reg_lambda=0.0`, `random_state=42`.

### Holdout Performance vs. Frozen Benchmarks:

| Model / Benchmark | Holdout MAE (EUR/MWh) | Holdout RMSE (EUR/MWh) | Holdout Median AE (EUR/MWh) | Holdout Bias (EUR/MWh) | Relative Improvement vs Benchmark |
|---|---|---|---|---|---|
| **24h Persistence** | 32.73 | — | — | — | Baseline |
| **168h Persistence** | 39.05 | — | — | — | Baseline |
| **7-Day Lag Mean** | 33.70 | — | — | — | Baseline |
| **LightGBM Core Strict (Primary)** | **26.87** | **45.67** | **18.94** | **-2.31** | **+17.91% vs 24h**<br>+31.18% vs 168h<br>+20.27% vs 7d mean |

### Validation-to-Holdout Comparison:
- **Validation MAE**: 18.97 EUR/MWh
- **Final Holdout MAE**: 26.87 EUR/MWh
- **Difference**: +7.90 EUR/MWh (+41.65%)
- **Generalization Classification**: **substantial degradation**

---

## 4. Supplementary Regression Sensitivity: LightGBM Forecast Extension

> **QUALIFICATION: ARCHIVE_VINTAGE_LIMITATION**  
> This evaluation is provided strictly as a supplementary sensitivity analysis. The 3 forecast extension predictors (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`) reflect retrospective archived vintages rather than strictly verified pre-auction point-in-time bid snapshots. It does not replace the primary Core Strict model.

| Metric | Core Strict (Primary) | Forecast Extension (Supplementary) | Difference (Ext - Core) |
|---|---|---|---|
| **MAE (EUR/MWh)** | **26.87** | 26.55 | -0.32 |
| **RMSE (EUR/MWh)** | 45.67 | **45.27** | -0.40 |
| **Median AE (EUR/MWh)** | **18.94** | 18.51 | -0.43 |
| **Bias (EUR/MWh)** | **-2.31** | -1.70 | +0.61 |
| **Improvement vs 24h (%)** | +17.91% | +18.89% | — |
| **Generalization Status** | substantial degradation | substantial degradation | — |

---

## 5. Primary Classification Evaluation: Logistic Regression Core Strict

### Model Configuration:
- **Architecture**: `Pipeline(StandardScaler, LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))`
- **Operating Decision Threshold**: Permanently fixed at **0.45** before holdout opening.

### Performance & Validation Comparison:

| Metric | Validation (Frozen) | Holdout (Measured) | Absolute Change | Relative Change (%) | Generalization Classification |
|---|---|---|---|---|---|
| **PR-AUC** | 0.5914 | **0.4538** | -0.1376 | -23.27% | moderate degradation |
| **ROC-AUC** | 0.9360 | **0.9019** | -0.0341 | -3.64% | similar performance |
| **Balanced Accuracy** | 0.8505 | **0.8023** | -0.0482 | -5.67% | similar performance |
| **Recall** | 0.7857 | **0.8412** | +0.0555 | +7.06% | improved holdout performance |
| **Precision** | 0.3340 | **0.1653** | -0.1687 | -50.52% | substantial degradation |
| **F1 Score** | 0.4687 | **0.2763** | -0.1924 | -41.06% | substantial degradation |

### Holdout Confusion Matrix (Threshold = 0.45):
- **True Negatives (TN)**: 3,194
- **False Positives (FP)**: 990
- **False Negatives (FN)**: 37
- **True Positives (TP)**: 196

---

## 6. Supplementary Classification Sensitivity: Logistic Regression Forecast Extension

> **QUALIFICATION: PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED**  
> Operating Decision Threshold: **0.55** (frozen). This model is supplementary only and does not replace the primary Core Strict classifier.

| Metric | Core Strict (Primary, th=0.45) | Forecast Extension (Supplementary, th=0.55) |
|---|---|---|
| **PR-AUC** | 0.4538 | 0.4941 |
| **ROC-AUC** | 0.9019 | 0.9118 |
| **Balanced Accuracy** | 0.8023 | 0.8260 |
| **Recall** | 0.8412 | 0.8283 |
| **Precision** | 0.1653 | 0.2073 |
| **F1 Score** | 0.2763 | 0.3316 |
| **Confusion Matrix [TN, FP, FN, TP]** | [3194, 990, 37, 196] | [3446, 738, 40, 193] |

---

## 7. Final Generalization Analysis

Descriptive evaluation based strictly on measured metric differences:

### Regression Generalization:
- **Measured Result**: Validation MAE of 18.97 EUR/MWh compared to Holdout MAE of 26.87 EUR/MWh (+7.90 EUR/MWh, +41.65%).
- **Classification**: **substantial degradation**.
- **Context**: The primary model maintains substantial predictive value over naive benchmarks (17.91% improvement over 24h persistence).

### Classification Generalization:
- **Measured Result**: PR-AUC changed from 0.5914 to 0.4538 (-23.27%). F1 changed from 0.4687 to 0.2763 (-41.06%). Recall changed from 0.7857 to 0.8412 (+7.06%).
- **Classification**: **moderate degradation** (PR-AUC), **substantial degradation** (F1).
- **Prevalence Context**: The holdout negative-price prevalence is 5.28% (233 hours), compared to 5.13% (224 hours) in the validation set, and 3.94% (511 hours) across the combined development set (TRAIN: 287 hours, VALIDATION: 224 hours, DEVELOPMENT: 511 hours).

---

## 8. Diagnostic Figures Generated

The following five figures have been rendered to `reports/figures/`:
1. `final_holdout_regression_actual_vs_predicted.png`: Scatter plot of actual vs predicted spot prices with perfect forecast reference line and error summary.
2. `final_holdout_regression_error_distribution.png`: Histogram and kernel density of prediction errors, indicating mean bias and median error.
3. `final_holdout_regression_benchmark_comparison.png`: Bar chart comparing Primary LightGBM Core Strict MAE against 24h persistence, 168h persistence, and 7-day lag mean benchmarks.
4. `final_holdout_classification_pr_curve.png`: Precision-Recall curve with annotated PR-AUC and marked operating threshold (0.45).
5. `final_holdout_classification_confusion_matrix.png`: Heatmap of the binary confusion matrix at threshold 0.45.
