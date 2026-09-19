# Milestone 7B: Negative-Price Classification Refinement Report

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Fitting Split (TRAIN)**: `2023-01-08 00:00` to `2023-12-31 23:00 Europe/Berlin` (**8,592 hours**)  
**Evaluation Split (VALIDATION)**: `2024-01-01 00:00` to `2024-06-30 23:00 Europe/Berlin` (**4,367 hours**)  
**Holdout Policy Compliance**: The final model holdout (`2024-07-01` to `2024-12-31`) was **strictly excluded** and neither accessed, fitted, nor scored.

---

## 1. Executive Summary & Model Selection

Negative-price events are rare (5.13% validation prevalence), making standard accuracy and default 0.50 thresholds unsuitable for early-warning operation.

### Selection Principle:
- **Primary Selection Metric**: Validation **PR-AUC** (Area Under the Precision-Recall Curve), which is threshold-independent and directly reflects the precision-recall trade-off under extreme class imbalance.
- **Secondary Context**: ROC-AUC.

### Validation Model Comparison (PR-AUC):

| Model Family | Candidate Model | PR-AUC | ROC-AUC | Bal Acc (t=0.50) | Recall (t=0.50) | Precision (t=0.50) | F1 (t=0.50) | Status |
|---|---|---|---|---|---|---|---|---|
| **Core Strict** | **Logistic Regression** | **0.5914** | 0.9360 | 0.8334 | 0.7411 | 0.3502 | 0.4756 | **Selected Best Core** |
| **Core Strict** | LightGBM Classifier | 0.5337 | 0.9245 | 0.7046 | 0.4241 | 0.6051 | 0.4987 | — |
| **Forecast Ext*** | **Logistic Regression** | **0.6126** | 0.9425 | 0.8475 | 0.7768 | 0.3392 | 0.4722 | **Selected Best Ext*** |
| **Forecast Ext*** | LightGBM Classifier | 0.5617 | 0.9292 | 0.7060 | 0.4241 | 0.6552 | 0.5149 | — |

> **Key Finding**: Across both feature families, **Logistic Regression achieved higher validation PR-AUC than LightGBM** (0.5914 vs 0.5337 in Core Strict; 0.6126 vs 0.5617 in Forecast Extension), indicating a stronger precision-recall trade-off for the negative-price classification task.

---

## 2. Predefined Threshold Selection Objective

This project operates as an **early-warning system** for negative electricity prices. Missing a negative-price event is costly, but false alarms must remain bounded.

### Predefined Selection Rule:
1. Restrict candidate thresholds to those achieving **Recall >= 0.75**.
2. Among those qualifying candidates, select the threshold with the **highest F1 score**.
3. If no threshold achieves Recall >= 0.75, fallback to the threshold with the highest Balanced Accuracy.

---

## 3. Core Strict Threshold Selection (Logistic Regression)

Evaluated on 17 predefined thresholds [0.10 to 0.90]:

| Threshold | Balanced Accuracy | Recall | Precision | F1 Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.8286 | 0.9330 | 0.1546 | **0.2652** | 209 | 1143 | 15 | 3000 |
| 0.15 | 0.8545 | 0.9286 | 0.1860 | **0.3100** | 208 | 910 | 16 | 3233 |
| 0.20 | 0.8616 | 0.9062 | 0.2112 | **0.3426** | 203 | 758 | 21 | 3385 |
| 0.25 | 0.8629 | 0.8839 | 0.2321 | **0.3677** | 198 | 655 | 26 | 3488 |
| 0.30 | 0.8652 | 0.8616 | 0.2619 | **0.4017** | 193 | 544 | 31 | 3599 |
| 0.35 | 0.8614 | 0.8348 | 0.2873 | **0.4274** | 187 | 464 | 37 | 3679 |
| 0.40 | 0.8618 | 0.8214 | 0.3124 | **0.4526** | 184 | 405 | 40 | 3738 |
| 0.45 **[Selected]** | 0.8505 | 0.7857 | 0.3340 | **0.4687** | 176 | 351 | 48 | 3792 |
| 0.50 | 0.8334 | 0.7411 | 0.3502 | **0.4756** | 166 | 308 | 58 | 3835 |
| 0.55 | 0.8218 | 0.7098 | 0.3672 | **0.4840** | 159 | 274 | 65 | 3869 |
| 0.60 | 0.8127 | 0.6830 | 0.3903 | **0.4968** | 153 | 239 | 71 | 3904 |
| 0.65 | 0.7961 | 0.6429 | 0.4068 | **0.4983** | 144 | 210 | 80 | 3933 |
| 0.70 | 0.7773 | 0.5982 | 0.4254 | **0.4972** | 134 | 181 | 90 | 3962 |
| 0.75 | 0.7676 | 0.5714 | 0.4604 | **0.5100** | 128 | 150 | 96 | 3993 |
| 0.80 | 0.7629 | 0.5536 | 0.5188 | **0.5356** | 124 | 115 | 100 | 4028 |
| 0.85 *[Max F1]* | 0.7429 | 0.5045 | 0.5947 | **0.5459** | 113 | 77 | 111 | 4066 |
| 0.90 | 0.7039 | 0.4196 | 0.6573 | **0.5123** | 94 | 49 | 130 | 4094 |

### Core Strict Operating Point:
- **Selected Operating Threshold**: **0.45**
- **Balanced Accuracy**: **0.8505**
- **Recall**: **0.7857** (176 of 224 negative hours detected)
- **Precision**: **0.3340** (351 false alarms)
- **F1 Score**: **0.4687**
- **PR-AUC**: **0.5914**
- **ROC-AUC**: **0.9360**
- **Confusion Matrix**: [TN=3792, FP=351, FN=48, TP=176]
- **Comparison with Unconstrained Max F1**: The unconstrained F1-maximizing threshold is **0.85** (F1: 0.5459, Recall: 0.5045, Precision: 0.5947). It achieves higher precision but misses the Recall >= 0.75 requirement, making threshold **0.45** the preferred operating point for an early-warning alert system.

---

## 4. Forecast Extension Threshold Selection (Logistic Regression*)

> *PROVENANCE CAVEAT ON FORECAST EXTENSION*:  
> Contains features qualified under `"ARCHIVE_VINTAGE_LIMITATION"` (`load_forecast_mw`, `load_forecast_diff_24h`, `load_forecast_daily_peak_ratio`). Because SMARD retrospective archives may reflect updated historical vintages rather than strictly preserved pre-auction point-in-time bids, this result is reported separately and is not equivalent in leakage certainty to the Core Strict pipeline.

| Threshold | Balanced Accuracy | Recall | Precision | F1 Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.8309 | 0.9554 | 0.1497 | **0.2588** | 214 | 1216 | 10 | 2927 |
| 0.15 | 0.8463 | 0.9241 | 0.1775 | **0.2978** | 207 | 959 | 17 | 3184 |
| 0.20 | 0.8475 | 0.8884 | 0.1990 | **0.3252** | 199 | 801 | 25 | 3342 |
| 0.25 | 0.8552 | 0.8750 | 0.2232 | **0.3557** | 196 | 682 | 28 | 3461 |
| 0.30 | 0.8580 | 0.8571 | 0.2471 | **0.3836** | 192 | 585 | 32 | 3558 |
| 0.35 | 0.8561 | 0.8348 | 0.2691 | **0.4070** | 187 | 508 | 37 | 3635 |
| 0.40 | 0.8553 | 0.8170 | 0.2933 | **0.4316** | 183 | 441 | 41 | 3702 |
| 0.45 | 0.8553 | 0.8036 | 0.3186 | **0.4563** | 180 | 385 | 44 | 3758 |
| 0.50 | 0.8475 | 0.7768 | 0.3392 | **0.4722** | 174 | 339 | 50 | 3804 |
| 0.55 **[Selected]** | 0.8513 | 0.7723 | 0.3745 | **0.5044** | 173 | 289 | 51 | 3854 |
| 0.60 | 0.8414 | 0.7455 | 0.3911 | **0.5131** | 167 | 260 | 57 | 3883 |
| 0.65 | 0.8372 | 0.7277 | 0.4245 | **0.5362** | 163 | 221 | 61 | 3922 |
| 0.70 | 0.8275 | 0.7009 | 0.4524 | **0.5499** | 157 | 190 | 67 | 3953 |
| 0.75 | 0.8157 | 0.6652 | 0.5156 | **0.5809** | 149 | 140 | 75 | 4003 |
| 0.80 *[Max F1]* | 0.8019 | 0.6295 | 0.5709 | **0.5987** | 141 | 106 | 83 | 4037 |
| 0.85 | 0.7703 | 0.5580 | 0.6345 | **0.5938** | 125 | 72 | 99 | 4071 |
| 0.90 | 0.7263 | 0.4643 | 0.6842 | **0.5532** | 104 | 48 | 120 | 4095 |

### Forecast Extension Operating Point:
- **Selected Operating Threshold**: **0.55**
- **Balanced Accuracy**: **0.8513**
- **Recall**: **0.7723** (173 of 224 negative hours detected)
- **Precision**: **0.3745** (289 false alarms)
- **F1 Score**: **0.5044**
- **PR-AUC**: **0.6126**
- **ROC-AUC**: **0.9425**
- **Confusion Matrix**: [TN=3854, FP=289, FN=51, TP=173]
- **Comparison with Unconstrained Max F1**: The unconstrained F1-maximizing threshold is **0.80** (F1: 0.5987, Recall: 0.6295, Precision: 0.5709).

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
