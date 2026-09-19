# Milestone 7A: Limited Regression Model Refinement Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**8,592 hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**4,367 hours**)  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Executive Summary & Selection Rule

Milestone 7A evaluated a strictly controlled set of **6 fixed candidate hyperparameter configurations (Candidates A–F)** to test whether modest regularization and tree capacity adjustments improve over the initial Milestone 6 baseline (Candidate A).

### Selection Rule:
- **Primary Selection Metric**: Validation **Mean Absolute Error (MAE)** in EUR/MWh.
- **Secondary Diagnostics**: RMSE, Median Absolute Error, Mean Error / Bias ($\text{mean}(\hat{y} - y)$).
- **Complexity Assessment Rule**: If improvement over Candidate A is less than 1.0% MAE, tuning is classified as producing only **marginal improvement**.

---

## 2. Core Strict Regression Tuning Results (23 Predictors)

| Candidate | n_est | lr | leaves | depth | min_child | subsample | colsample | reg_lambda | MAE (EUR/MWh) | RMSE (EUR/MWh) | Median AE (EUR/MWh) | Bias (EUR/MWh) | vs Cand A (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Candidate A** | 300 | 0.05 | 31 | 6 | 30 | 0.8 | 0.8 | 0.0 | **19.87** | 25.85 | 15.75 | +5.92 | +0.00% |
| **Candidate B** | 400 | 0.03 | 31 | 6 | 30 | 0.8 | 0.8 | 0.0 | **19.90** | 25.97 | 15.61 | +6.07 | -0.17% |
| **Candidate C** | 300 | 0.05 | 15 | 5 | 40 | 0.8 | 0.8 | 0.0 | **19.60** | 25.65 | 15.43 | +5.50 | +1.38% |
| **Candidate D** **(Best)** | 400 | 0.03 | 15 | 5 | 50 | 0.8 | 0.8 | 0.0 | **18.97** | 25.06 | 14.74 | +5.30 | +4.55% |
| **Candidate E** | 300 | 0.05 | 31 | 5 | 60 | 0.75 | 0.75 | 1.0 | **19.30** | 25.39 | 15.28 | +5.94 | +2.85% |
| **Candidate F** | 400 | 0.03 | 31 | 6 | 50 | 0.8 | 0.8 | 1.0 | **19.80** | 25.80 | 15.62 | +5.50 | +0.34% |

### Core Strict Selection Analysis:
- **Selected Best Configuration**: **Candidate D**
- **Validation MAE**: **18.97 EUR/MWh** (vs 19.87 EUR/MWh for Candidate A)
- **Improvement vs Candidate A**: **+4.55%**
- **Improvement vs Ridge Core Strict (21.91 EUR/MWh)**: **+13.45%**
- **Improvement vs 24h Persistence (22.84 EUR/MWh)**: **+16.98%**
- **Tuning Outcome**: **Meaningful improvement (+4.55% MAE reduction vs Candidate A)**

---

## 3. Forecast Extension Regression Tuning Results (26 Predictors*)

> *PROVENANCE CAVEAT ON FORECAST EXTENSION*:  
> Contains features qualified under `"ARCHIVE_VINTAGE_LIMITATION"` (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`). Because SMARD retrospective archives may reflect updated historical vintages rather than strictly preserved pre-auction point-in-time bids, this result is reported separately and is not equivalent in leakage certainty to the Core Strict pipeline.

| Candidate | n_est | lr | leaves | depth | min_child | subsample | colsample | reg_lambda | MAE (EUR/MWh) | RMSE (EUR/MWh) | Median AE (EUR/MWh) | Bias (EUR/MWh) | vs Cand A (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Candidate A** | 300 | 0.05 | 31 | 6 | 30 | 0.8 | 0.8 | 0.0 | **19.80** | 25.92 | 15.53 | +6.42 | +0.00% |
| **Candidate B** | 400 | 0.03 | 31 | 6 | 30 | 0.8 | 0.8 | 0.0 | **19.66** | 25.59 | 15.45 | +6.09 | +0.70% |
| **Candidate C** | 300 | 0.05 | 15 | 5 | 40 | 0.8 | 0.8 | 0.0 | **19.27** | 25.32 | 15.13 | +6.79 | +2.67% |
| **Candidate D** **(Best)** | 400 | 0.03 | 15 | 5 | 50 | 0.8 | 0.8 | 0.0 | **18.98** | 24.89 | 14.91 | +5.77 | +4.16% |
| **Candidate E** | 300 | 0.05 | 31 | 5 | 60 | 0.75 | 0.75 | 1.0 | **19.62** | 25.64 | 15.25 | +7.03 | +0.91% |
| **Candidate F** | 400 | 0.03 | 31 | 6 | 50 | 0.8 | 0.8 | 1.0 | **19.72** | 25.63 | 15.55 | +6.11 | +0.37% |

### Forecast Extension Selection Analysis:
- **Selected Best Configuration**: **Candidate D**
- **Validation MAE**: **18.98 EUR/MWh** (vs 19.80 EUR/MWh for Candidate A)
- **Improvement vs Candidate A**: **+4.16%**
- **Improvement vs Ridge Forecast Extension (20.82 EUR/MWh)**: **+8.87%**
- **Improvement vs 24h Persistence (22.84 EUR/MWh)**: **+16.94%**
- **Tuning Outcome**: **Meaningful improvement (+4.16% MAE reduction vs Candidate A)**

---

## 4. Methodological Findings & Model Selection

1. **Validation Performance Comparison**:
   The lower-capacity Candidate D configuration achieved better validation performance than Candidate A, with a 4.55% MAE reduction for Core Strict and 4.16% for Forecast Extension:
   - **Core Strict Candidate D**: MAE = **18.97 EUR/MWh**
   - **Forecast Extension Candidate D**: MAE = **18.98 EUR/MWh**
2. **Holdout Candidate Selection**:
   Because Core Strict has marginally lower validation MAE and does not rely on `ARCHIVE_VINTAGE_LIMITATION` features, it is the current preferred regression candidate for final holdout evaluation. *(Note: The final holdout split remains strictly unaccessed and will not be evaluated until all modeling milestones are finalized).*
3. **Performance Relative to Linear and Persistence Baselines**:
   Candidate D outperforms both 24h Persistence (+17.0% MAE reduction) and regularized Ridge regression (+13.5% vs Ridge Core Strict, +8.9% vs Ridge Forecast Extension).
4. **No Holdout Compromise**:
   All candidate evaluations and model selections were conducted exclusively on VALIDATION data ($N=4,367$). The final holdout split remains strictly unaccessed.

---

## 5. Diagnostic Figures

Generated validation diagnostic figures in `reports/figures/`:
1. [Figure 1: Validation MAE Comparison Across Candidates A–F](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/regression_tuning_mae_comparison.png)
2. [Figure 2: Validation Price Residual Distributions (Best Models vs Candidate A)](file:///c:/Users/PC/Desktop/Documents/project-3/reports/figures/regression_tuning_best_model_residuals.png)
