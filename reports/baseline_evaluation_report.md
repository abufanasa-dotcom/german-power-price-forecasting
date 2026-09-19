# Milestone 5: Baseline Models Evaluation Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**8,592 hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**4,367 hours**)  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Class Prevalence (Negative Spot Prices)

Negative prices ($P < 0.00$ EUR/MWh) represent the rare, high-impact tail of wholesale electricity settlements:

| Split | Total Hourly Observations | Negative Price Hours ($P < 0.00$) | Negative Price Prevalence (%) | Non-Negative Hours ($P \ge 0.00$) |
|---|---|---|---|---|
| **TRAIN (2023)** | 8,592 | 287 | **3.34%** | 8,305 |
| **VALIDATION (H1 2024)** | 4,367 | 224 | **5.13%** | 4,143 |

*Observation*: Negative-price prevalence increased markedly from **3.34%** in 2023 training to **5.13%** in H1 2024 validation (+1.79 percentage points).

---

## 2. Regression Baselines Performance

Primary metric: **Mean Absolute Error (MAE)** in EUR/MWh.  
Secondary metrics: **RMSE**, **Median Absolute Error**, **Mean Error / Bias**.  
*(Note: Mean Absolute Percentage Error (MAPE) is excluded due to division by zero/negative price settlement values).*

| Baseline Model | Feature Group | Train Rows | Val Rows | MAE (EUR/MWh) | RMSE (EUR/MWh) | Median AE (EUR/MWh) | Mean Error Bias (EUR/MWh) | Improvement vs 24h (%) | Improvement vs 168h (%) |
|---|---|---|---|---|---|---|---|---|---|
| **24h Persistence** | Zero-parameter | — | 4,367 | **22.84** | 32.84 | 14.60 | -0.19 | 0.00% | +13.73% |
| **168h Weekly Persistence** | Zero-parameter | — | 4,367 | **26.48** | 35.76 | 19.02 | -2.17 | -15.91% | 0.00% |
| **7-Day Historical Daily Mean** | Zero-parameter | — | 4,367 | **22.49** | 29.76 | 16.84 | -1.08 | +1.53% | +15.05% |
| **Ridge Regression (Core Strict)** | 23 Core Predictors | 8,592 | 4,367 | **21.91** | 28.40 | 17.56 | +13.07 | **+4.08%** | **+17.25%** |
| **Ridge Regression (Forecast Ext)** | 26 Predictors* | 8,592 | 4,367 | **20.82** | 27.15 | 16.19 | +12.74 | **+8.86%** | **+21.37%** |

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
| **Dummy Majority** | `most_frequent` | 0.5000 | 0.0000 | 0.0000 | 0.0000 | — | — | [4143, 0, 224, 0] |
| **Dummy Prior** | `prior` | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0513 | 0.5000 | [4143, 0, 224, 0] |
| **Logistic Regression (Core Strict)** | 23 Core Predictors | **0.8334** | **0.7411** | **0.3502** | **0.4756** | **0.5914** | **0.9360** | [3835, 308, 58, 166] |
| **Logistic Regression (Forecast Ext\*)** | 26 Predictors | **0.8475** | **0.7768** | **0.3392** | **0.4722** | **0.6126** | **0.9425** | [3804, 339, 50, 174] |

---

## 4. Key Interpretive Answers

### 1. How strong is 24h persistence?
24h persistence achieves a validation MAE of **22.84 EUR/MWh**. In power markets, day-ahead price dynamics exhibit strong daily autocorrelation, making 24h persistence a surprisingly competitive hurdle that naïve statistical models often fail to beat.

### 2. Does weekly persistence perform better or worse?
Weekly persistence (168h) performs **substantially worse**, with an MAE of **26.48 EUR/MWh** (a **15.9% deterioration** compared to 24h persistence). While weekly seasonality exists (weekday vs. weekend patterns), power prices are heavily driven by short-term weather regime shifts (synoptic wind fronts and cloud cover) that change completely over a 7-day span.

### 3. Does Ridge improve over persistence?
Yes. Ridge regression using Core Strict predictors achieves an MAE of **21.91 EUR/MWh**, delivering a **4.08% improvement over 24h persistence** and a **17.25% improvement over 168h persistence**. The combination of multi-day price lags (24h to 168h), historical load lags, diurnal/annual calendar harmonics, and the deterministic solar elevation proxy enables Ridge to capture both the general price level and the midday solar depression.

### 4. How much incremental validation value does archived load forecast add?
Adding the official Day-Ahead load forecast predictors reduces the validation MAE from **21.91 EUR/MWh** to **20.82 EUR/MWh** (an incremental gain of **1.09 EUR/MWh** or **4.98%**). While load forecasts provide positive signal, their incremental value is modest because lagged load and calendar features already proxy the predictable industrial demand curve.

### 5. How difficult is negative-price classification at threshold 0.50?
At the default 0.50 threshold with balanced class weighting, Logistic Regression Core Strict achieves **83.3% Balanced Accuracy** and **74.1% Recall** (166 out of 224 negative hours identified), but precision is **35.0%** with 308 false alarms. Because negative prices are a minority event (5.13% prevalence), standard balanced-weight logistic regression tends to over-predict negatives. This demonstrates that **probability calibration and threshold tuning are strictly necessary** in subsequent modeling milestones.

### 6. Does the classifier improve materially over dummy baselines?
Yes. The Dummy Majority baseline achieves a Balanced Accuracy of **50.00%**, Recall of **0.00%**, and F1 of **0.0000** (predicting zero negative events). The Dummy Prior baseline achieves a PR-AUC of **0.0513** (equal to class prevalence). In contrast, Logistic Regression Core Strict achieves a PR-AUC of **0.5914** and ROC-AUC of **0.9360**, confirming strong discriminating power far above random chance.

---

## 5. Diagnostic Figures

Generated validation diagnostic figures in `reports/figures/`:
1. [Figure 1: Validation Actual vs Predicted Price](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_01_val_actual_vs_predicted_week.png)
2. [Figure 2: Validation Price Residual Distribution](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_02_val_residual_distribution.png)
3. [Figure 3: Regression MAE Comparison Across Baselines](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_03_regression_mae_comparison.png)
4. [Figure 4: Negative-Price Classification Precision-Recall Curves](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_04_classification_pr_curves.png)
5. [Figure 5: Classification Confusion Matrices](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/baseline_05_classification_confusion_matrices.png)
