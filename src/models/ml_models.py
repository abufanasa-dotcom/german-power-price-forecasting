"""
Machine Learning Models Module for German Day-Ahead Electricity Market Forecasting.

Provides:
- Fixed, conservative LightGBM Regressor configuration for spot price prediction.
- Fixed, conservative LightGBM Classifier configuration for negative price detection.
- Class-imbalance weighting derived strictly from training data.
- Feature importance extraction utilities (gain and split).
"""

from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

# Standard conservative configuration for LightGBM Regressor
DEFAULT_LGBM_REGRESSOR_PARAMS: Dict[str, Any] = {
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

# Standard conservative configuration for LightGBM Classifier
DEFAULT_LGBM_CLASSIFIER_PARAMS: Dict[str, Any] = {
    "objective": "binary",
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


def compute_train_class_weight(y_train: pd.Series) -> float:
    """
    Calculate class-imbalance scale_pos_weight strictly from training labels.
    
    Formula:
        scale_pos_weight = N_class_0 / N_class_1
        where:
        - class 0 = non-negative price hour (price >= 0)
        - class 1 = negative-price hour (price < 0)
    """
    y_arr = np.asarray(y_train, dtype=int)
    n_class_1 = int((y_arr == 1).sum())
    n_class_0 = int((y_arr == 0).sum())
    if n_class_1 == 0:
        raise ValueError("Training target contains zero class 1 (negative-price) instances.")
    return float(n_class_0 / n_class_1)


def build_lgbm_regressor(
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> lgb.LGBMRegressor:
    """
    Construct a LightGBM Regressor with conservative hyperparameters.
    """
    cfg = DEFAULT_LGBM_REGRESSOR_PARAMS.copy()
    if params:
        cfg.update(params)
    cfg["random_state"] = random_state
    return lgb.LGBMRegressor(**cfg)


def build_lgbm_classifier(
    scale_pos_weight: float,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """
    Construct a LightGBM Classifier with class imbalance weighting and conservative hyperparameters.
    """
    cfg = DEFAULT_LGBM_CLASSIFIER_PARAMS.copy()
    if params:
        cfg.update(params)
    cfg["scale_pos_weight"] = scale_pos_weight
    cfg["random_state"] = random_state
    return lgb.LGBMClassifier(**cfg)


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract feature importance by both 'gain' and 'split' from a fitted LightGBM model.
    """
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")
    
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": gain_importance,
        "importance_split": split_importance,
    })
    
    # Sort descending by gain importance
    df_imp = df_imp.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    df_imp["gain_share_pct"] = (df_imp["importance_gain"] / df_imp["importance_gain"].sum()) * 100.0
    return df_imp
