"""
Baseline Models Module for German Day-Ahead Electricity Market Forecasting.

Provides:
- Regression Baselines:
  1. 24h Persistence: y_hat = price_lag_24h
  2. 168h Weekly Persistence: y_hat = price_lag_168h
  3. Historical-Lag 7-Day Mean: y_hat = price_daily_lag_mean_7d
  4. Ridge (Core Strict): scikit-learn Pipeline with StandardScaler, fit on TRAIN only
  5. Ridge (Official Forecast Extension): scikit-learn Pipeline with StandardScaler, fit on TRAIN only
- Classification Baselines:
  1. Dummy Majority: DummyClassifier(strategy="most_frequent")
  2. Dummy Prior: DummyClassifier(strategy="prior")
  3. Logistic Regression (Core Strict): LogisticRegression(class_weight="balanced") in Pipeline with StandardScaler
  4. Logistic Regression (Official Forecast Extension): LogisticRegression(class_weight="balanced") in Pipeline with StandardScaler
- Standardized Metric Evaluation:
  - Regression: MAE (primary), RMSE, Median AE, Bias (Mean Error). No MAPE.
  - Classification: Balanced Accuracy, Recall, Precision, F1, PR-AUC, Confusion Matrix, ROC-AUC. Default threshold 0.50.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    baseline_mae_24h: Optional[float] = None,
    baseline_mae_168h: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute regression metrics for Day-Ahead electricity prices.
    
    Primary Metric: MAE (EUR/MWh)
    Secondary Metrics: RMSE (EUR/MWh), Median AE (EUR/MWh), Mean Error / Bias (EUR/MWh)
    Note: MAPE is intentionally excluded because electricity prices include 0 and negative values.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    err = y_p - y_t
    abs_err = np.abs(err)
    
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    median_ae = float(np.median(abs_err))
    bias = float(np.mean(err))
    sample_count = len(y_t)
    
    metrics = {
        "sample_count": sample_count,
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "bias": bias,
    }
    
    if baseline_mae_24h is not None and baseline_mae_24h > 0:
        rel_imp_24h = ((baseline_mae_24h - mae) / baseline_mae_24h) * 100.0
        metrics["improvement_vs_24h_pct"] = float(rel_imp_24h)
        
    if baseline_mae_168h is not None and baseline_mae_168h > 0:
        rel_imp_168h = ((baseline_mae_168h - mae) / baseline_mae_168h) * 100.0
        metrics["improvement_vs_168h_pct"] = float(rel_imp_168h)
        
    return metrics


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute binary classification metrics for negative price detection.
    
    Threshold: Default 0.50.
    Metrics: Balanced Accuracy, Recall, Precision, F1, PR-AUC, Confusion Matrix, ROC-AUC.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    metrics = {
        "sample_count": len(y_t),
        "balanced_accuracy": bal_acc,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    
    # Probabilistic ranking metrics
    if y_proba is not None:
        y_prob = np.asarray(y_proba, dtype=float)
        # Ensure single column of positive class probability
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
            
        # Check if true class contains both classes
        if len(np.unique(y_t)) > 1:
            metrics["pr_auc"] = float(average_precision_score(y_t, y_prob))
            metrics["roc_auc"] = float(roc_auc_score(y_t, y_prob))
        else:
            metrics["pr_auc"] = float("nan")
            metrics["roc_auc"] = float("nan")
            
    return metrics


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Construct a Ridge regression pipeline with StandardScaler.
    Preprocessing is fit strictly on training data during pipeline.fit().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha, random_state=42)),
    ])


def build_logistic_pipeline(C: float = 1.0) -> Pipeline:
    """
    Construct a Logistic Regression pipeline with StandardScaler and balanced class weights.
    Preprocessing is fit strictly on training data during pipeline.fit().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])
