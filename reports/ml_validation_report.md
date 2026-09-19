# Milestone 6: First Nonlinear ML Models Validation Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**8,592 hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**4,367 hours**)  
**LightGBM Version**: `4.7.0`  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Executive Summary & Model Configurations

In Milestone 6, we introduce nonlinear tree-based gradient boosting models using LightGBM (`lightgbm 4.7.0`) to capture complex, non-linear merit-order dynamics, solar midday depressions, and load-price elasticities.

### Exact Fixed Model Configurations (No Hyperparameter Search)

To prevent subtle test leakage and overfitting, fixed conservative hyperparameters were specified a priori:

```python
# LightGBM Regressor Configuration
DEFAULT_LGBM_REGRESSOR_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# LightGBM Classifier Configuration
DEFAULT_LGBM_CLASSIFIER_PARAMS = {
    "objective": "binary",
    "scale_pos_weight": 28.93728,  # (8,592 - 287) / 287 = 8,305 / 287
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
```

### Training Class-Imbalance Weight Calculation:
The classifier `scale_pos_weight` is derived **strictly from the TRAIN split**:
$$\text{scale\_pos\_weight} = \frac{N_{\text{class\_0}}}{N_{\text{class\_1}}} = \frac{8,592 - 287}{287} = \frac{8,305}{287} \approx 28.93728$$
where:
- class 0 = non-negative price hour ($P \ge 0.00$ EUR/MWh)
- class 1 = negative-price hour ($P < 0.00$ EUR/MWh)

*Validation split prevalence was strictly NOT used to determine training weights.*

---

## 2. Regression Validation Performance

Primary metric: **Mean Absolute Error (MAE)** in EUR/MWh.  
Secondary metrics: **RMSE**, **Median Absolute Error**, **Mean Error / Bias**.  
*Bias definition*: Formally defined as `mean(prediction - actual)`. A positive bias indicates over-predicting relative to settled spot prices.

| Regression Model | Feature Group | Train Rows | Val Rows | MAE (EUR/MWh) | RMSE (EUR/MWh) | Median AE (EUR/MWh) | Bias (EUR/MWh) | Improvement vs 24h (%) | Improvement vs Ridge (%) |
|---|---|---|---|---|---|---|---|---|---|
| **24h Persistence** | Zero-parameter | — | 4,367 | **22.84** | 32.84 | 14.60 | -0.19 | 0.00% | — |
| **Ridge (Core Strict)** | 23 Core Predictors | 8,592 | 4,367 | **21.91** | 28.40 | 17.56 | +13.07 | +4.08% | 0.00% |
| **LightGBM (Core Strict)** | 23 Core Predictors | 8,592 | 4,367 | **19.87** | 25.85 | 15.75 | +5.92 | **+13.02%** | **+9.32%** |
| **Ridge (Forecast Ext\*)** | 26 Predictors | 8,592 | 4,367 | **20.82** | 27.15 | 16.19 | +12.74 | +8.86% | 0.00% |
| **LightGBM (Forecast Ext\*)** | 26 Predictors | 8,592 | 4,367 | **19.80** | 25.92 | 15.53 | +6.42 | **+13.34%** | **+4.91%** |

> *\*PROVENANCE CAVEAT ON FORECAST EXTENSION*:  
> Contains features qualified under `"ARCHIVE_VINTAGE_LIMITATION"` (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`). Because SMARD retrospective archives may reflect updated historical vintages rather than strictly preserved pre-auction point-in-time bids, this result is reported separately and is not equivalent in leakage certainty to the Core Strict pipeline.

---

## 3. Negative-Price Classification Performance

Evaluated at the standard fixed default threshold of **0.50** (threshold optimization is intentionally deferred to Milestone 7):

| Classification Model | Feature Group | Balanced Accuracy | Recall | Precision | F1 Score | PR-AUC | ROC-AUC | Confusion Matrix [TN, FP, FN, TP] |
|---|---|---|---|---|---|---|---|---|
| **Dummy Prior** | `prior` | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0513 | 0.5000 | [4143, 0, 224, 0] |
| **Logistic Regression (Core Strict)** | 23 Core Predictors | 0.8334 | 0.7411 | 0.3502 | 0.4756 | 0.5914 | 0.9360 | [3835, 308, 58, 166] |
| **LightGBM (Core Strict)** | 23 Core Predictors | **0.7046** | **0.4241** | **0.6051** | **0.4987** | **0.5337** | **0.9245** | [4081, 62, 129, 95] |
| **Logistic Regression (Forecast Ext\*)** | 26 Predictors | 0.8475 | 0.7768 | 0.3392 | 0.4722 | 0.6126 | 0.9425 | [3804, 339, 50, 174] |
| **LightGBM (Forecast Ext\*)** | 26 Predictors | **0.7060** | **0.4241** | **0.6552** | **0.5149** | **0.5617** | **0.9292** | [4093, 50, 129, 95] |

---

## 4. Top 10 Feature Importances

Feature importances represent model-internal split and gain attribution in the gradient-boosted decision trees.  
**Note**: Feature importance measures empirical predictive utility within the fitted tree ensemble; it **does not imply physical causality**.

### A. LightGBM Regressor — Core Strict (23 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
| 1 | `price_lag_24h` | 58,774,226.6 | 40.78% | 574 |
| 2 | `price_daily_lag_mean_7d` | 14,896,689.8 | 10.34% | 245 |
| 3 | `day_of_week` | 11,443,434.3 | 7.94% | 502 |
| 4 | `price_lag_168h` | 9,252,206.0 | 6.42% | 612 |
| 5 | `price_lag_48h` | 5,153,500.5 | 3.58% | 552 |
| 6 | `price_lag_144h` | 4,991,639.8 | 3.46% | 478 |
| 7 | `price_lag_72h` | 4,360,962.2 | 3.03% | 508 |
| 8 | `solar_elevation_proxy` | 3,864,715.7 | 2.68% | 306 |
| 9 | `price_lag_96h` | 3,627,505.1 | 2.52% | 444 |
| 10 | `load_actual_lag_168h` | 3,623,895.1 | 2.51% | 378 |

### B. LightGBM Regressor — Forecast Extension (26 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
| 1 | `price_lag_24h` | 59,536,029.7 | 40.98% | 541 |
| 2 | `price_daily_lag_mean_7d` | 15,145,535.8 | 10.43% | 219 |
| 3 | `load_forecast_diff_24h` | 10,794,866.6 | 7.43% | 672 |
| 4 | `price_lag_168h` | 8,392,652.3 | 5.78% | 534 |
| 5 | `day_of_week` | 6,392,923.7 | 4.40% | 291 |
| 6 | `price_lag_144h` | 4,274,411.8 | 2.94% | 385 |
| 7 | `solar_elevation_proxy` | 4,034,217.6 | 2.78% | 291 |
| 8 | `month` | 3,832,030.4 | 2.64% | 324 |
| 9 | `price_lag_48h` | 3,757,574.1 | 2.59% | 448 |
| 10 | `load_forecast_mw` | 3,694,888.5 | 2.54% | 342 |

### C. LightGBM Classifier — Core Strict (23 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
| 1 | `price_daily_lag_mean_7d` | 39,007.3 | 20.64% | 402 |
| 2 | `price_lag_24h` | 24,566.6 | 13.00% | 728 |
| 3 | `month_sin` | 14,764.4 | 7.81% | 191 |
| 4 | `day_of_week` | 14,653.9 | 7.75% | 277 |
| 5 | `price_lag_144h` | 11,911.4 | 6.30% | 598 |
| 6 | `price_lag_168h` | 10,901.2 | 5.77% | 680 |
| 7 | `load_actual_lag_48h` | 9,571.8 | 5.06% | 615 |
| 8 | `price_daily_lag_std_7d` | 8,633.9 | 4.57% | 525 |
| 9 | `load_actual_lag_168h` | 7,937.0 | 4.20% | 520 |
| 10 | `solar_elevation_proxy` | 7,438.5 | 3.94% | 397 |

### D. LightGBM Classifier — Forecast Extension (26 Predictors)
| Rank | Feature Name | Gain Importance | Gain Share (%) | Split Count |
|---|---|---|---|---|
| 1 | `price_daily_lag_mean_7d` | 38,138.1 | 19.79% | 391 |
| 2 | `price_lag_24h` | 23,264.5 | 12.07% | 702 |
| 3 | `load_forecast_mw` | 14,971.3 | 7.77% | 479 |
| 4 | `price_lag_168h` | 14,745.5 | 7.65% | 512 |
| 5 | `month_sin` | 12,068.2 | 6.26% | 162 |
| 6 | `price_lag_144h` | 11,034.5 | 5.73% | 504 |
| 7 | `day_of_week` | 9,156.6 | 4.75% | 130 |
| 8 | `solar_elevation_proxy` | 8,849.4 | 4.59% | 432 |
| 9 | `price_daily_lag_std_7d` | 8,337.3 | 4.33% | 426 |
| 10 | `load_forecast_diff_24h` | 6,834.0 | 3.55% | 578 |

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
