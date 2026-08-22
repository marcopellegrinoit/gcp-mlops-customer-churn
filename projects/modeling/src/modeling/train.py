"""Model training logic: XGBoost fit with SHAP feature importance."""

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from ml_common.evaluate import select_threshold
from sklearn.model_selection import StratifiedKFold


def select_threshold_via_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    fixed_params: dict,
    target_recall: float,
    n_folds: int,
    random_state: int = 42,
) -> float:
    """Pick the decision threshold from out-of-fold CV predictions, never from the test set.

    Refits the already-chosen params (no search) across n_folds stratified folds, collecting
    each fold's held-out predictions into one out-of-fold probability array spanning the whole
    training set, then selects the threshold on that. This keeps threshold selection from ever
    touching the test set, so post-training's reported F1 isn't optimistically biased by having
    also picked the operating point on that same held-out data.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y))

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        # computed per-fold to avoid leaking the full-set class ratio into validation
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        model = xgb.XGBClassifier(
            **params, **fixed_params, scale_pos_weight=scale_pos_weight, random_state=random_state
        )
        model.fit(X_train, y_train, verbose=False)
        oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]

    return select_threshold(y.to_numpy(), oof_proba, target_recall)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    fixed_params: dict,
    random_state: int = 42,
) -> tuple[xgb.XGBClassifier, list[str], dict[str, float]]:
    """Fit XGBoost on X/y and return (model, feature_names, shap_importance).

    scale_pos_weight is derived from y to handle class imbalance without oversampling.
    """
    feat_names = list(X.columns)
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    model_params = {
        **params,
        **fixed_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
    }
    model = xgb.XGBClassifier(**model_params)
    model.fit(X, y, verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_importance: dict[str, float] = {
        name: float(np.abs(vals).mean())
        for name, vals in zip(feat_names, shap_values.T, strict=True)
    }
    return model, feat_names, shap_importance
