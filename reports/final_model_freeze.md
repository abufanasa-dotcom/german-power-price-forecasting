# Final Model Freeze Checkpoint

**Project**: Portfolio Project 3 — German Day-Ahead Wholesale Electricity Price Forecasting  
**Bidding Zone**: Germany-Luxembourg (DE-LU) / German Control Area (DE)  
**Freeze Status**: **PERMANENTLY FROZEN BEFORE HOLDOUT OPENING**  
**Timestamp**: `2026-09-19T17:07:00Z`  
**Git Base Commit**: `03010f9`  

---

## 1. Data Splits

| Split Name | Delivery Start (Local) | Delivery End (Local) | Timezone | Hourly Row Count | Usage / Status |
|---|---|---|---|---|---|
| **TRAIN** | `2023-01-08 00:00` | `2023-12-31 23:00` | Europe/Berlin | **8,592** | Model fitting & preprocessing parameter fitting |
| **VALIDATION** | `2024-01-01 00:00` | `2024-06-30 23:00` | Europe/Berlin | **4,367** | Hyperparameter tuning, candidate comparison & threshold selection |
| **FINAL HOLDOUT** | `2024-07-01 00:00` | `2024-12-31 23:00` | Europe/Berlin | **4,417** | **COMPLETELY UNOPENED** |

> **Explicit Pre-Holdout Declaration**:  
> The final holdout split has **not yet been inspected** for target behavior, class prevalence, regression performance, or classification performance. It remains completely unopened.

---

## 2. Primary Regression Model — FROZEN

* **Feature Family**: **Core Strict** (23 leakage-controlled predictors)
* **Architecture**: **LightGBM Regressor** (`Candidate D`)
* **Selection Source**: **VALIDATION only** (Milestone 7A)
* **Validation Performance**:
  - **MAE**: **`18.97 EUR/MWh`**
  - **RMSE**: `25.06 EUR/MWh`
  - **Median AE**: `14.74 EUR/MWh`
  - **Mean Error / Bias**: `+5.30 EUR/MWh`
* **Frozen Hyperparameter Configuration**:
  ```python
  {
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
  ```
* **Governance Rule**: No further feature selection, hyperparameter tuning, or configuration changes are permitted after holdout opening.

---

## 3. Supplementary Regression Model — FROZEN

* **Feature Family**: **Forecast Extension** (26 predictors)
* **Architecture**: **LightGBM Regressor** (`Candidate D`, same hyperparameters as above)
* **Validation Performance**:
  - **MAE**: **`18.98 EUR/MWh`**
  - **RMSE**: `24.89 EUR/MWh`
  - **Median AE**: `14.91 EUR/MWh`
  - **Mean Error / Bias**: `+5.77 EUR/MWh`
* **Qualification**: `ARCHIVE_VINTAGE_LIMITATION`
* **Governance Rule**: This model is supplementary only and is not the primary leakage-controlled model.

---

## 4. Primary Classification Model — FROZEN

* **Feature Family**: **Core Strict** (23 leakage-controlled predictors)
* **Architecture**: **Logistic Regression** (Pipeline with `StandardScaler` and `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`)
* **Selection Source**: **VALIDATION only** (Milestone 7B)
* **Operating Threshold**: **`0.45`** (Permanently frozen before holdout opening)
* **Validation Performance**:
  - **PR-AUC**: **`0.5914`**
  - **ROC-AUC**: **`0.9360`**
  - **Balanced Accuracy**: **`0.8505`**
  - **Recall**: **`0.7857`** (176 of 224 negative hours captured)
  - **Precision**: **`0.3340`** (351 false alarms)
  - **F1 Score**: **`0.4687`**
  - **Confusion Matrix**: `[TN=3,792, FP=351, FN=48, TP=176]`
* **Governance Rule**: The operating threshold (0.45) is permanently frozen before holdout opening.

---

## 5. Supplementary Classification Model — FROZEN

* **Feature Family**: **Forecast Extension** (26 predictors)
* **Architecture**: **Logistic Regression** (Pipeline with `StandardScaler` and `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`)
* **Operating Threshold**: **`0.55`**
* **Validation Performance**:
  - **PR-AUC**: **`0.6126`**
  - **ROC-AUC**: **`0.9425`**
  - **Balanced Accuracy**: **`0.8513`**
  - **Recall**: **`0.7723`** (173 of 224 negative hours captured)
  - **Precision**: **`0.3745`** (289 false alarms)
  - **F1 Score**: **`0.5044`**
  - **Confusion Matrix**: `[TN=3,854, FP=289, FN=51, TP=173]`
* **Qualification**: `PRE-AUCTION BY PUBLICATION RULE / ARCHIVE-VINTAGE NOT GUARANTEED` (`ARCHIVE_VINTAGE_LIMITATION`)
* **Governance Rule**: This model is supplementary only.

---

## 6. Final Holdout Governance Rules

1. **Sequential Integrity**: The final holdout split will be evaluated only after this freeze checkpoint is safely committed in Git.
2. **Immutable Modeling Decisions**: Holdout evaluation results must **never** be used to:
   - change features or feature definitions
   - change hyperparameters
   - change model architectures
   - change classification thresholds
   - select another model
3. **Primary Decision Basis**: All primary conclusions and headline performance figures will be based strictly on the **Core Strict** models.
4. **Supplementary Status**: Forecast Extension results will be reported exclusively as supplementary sensitivity analyses under explicit vintage caveats.
5. **Finality**: Once holdout results are observed, they represent the final, definitive out-of-sample evaluation of this research project.
6. **Exploratory Boundary**: Any post-holdout experimentation must be explicitly labeled as exploratory and cannot replace or overwrite the frozen final evaluation.
