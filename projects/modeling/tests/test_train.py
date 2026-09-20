"""Unit tests for train_model (no GCP dependencies)."""

import numpy as np
import pandas as pd
import xgboost as xgb
from modeling.train import select_threshold_via_cv, train_model


def _make_dataset(n=200, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series((X["f0"] + X["f1"] > 1.0).astype(int))
    return X, y


FIXED_PARAMS = {"tree_method": "hist"}
PARAMS = {"max_depth": 3, "n_estimators": 20, "learning_rate": 0.1}


def test_train_model_returns_fitted_classifier():
    X, y = _make_dataset()
    model, _, _ = train_model(X, y, PARAMS, FIXED_PARAMS)
    assert isinstance(model, xgb.XGBClassifier)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_model_returns_feature_names_in_order():
    X, y = _make_dataset(n_features=4)
    _, feat_names, _ = train_model(X, y, PARAMS, FIXED_PARAMS)
    assert feat_names == list(X.columns)


def test_train_model_shap_importance_covers_all_features():
    X, y = _make_dataset(n_features=4)
    _, feat_names, shap_importance = train_model(X, y, PARAMS, FIXED_PARAMS)
    assert set(shap_importance.keys()) == set(feat_names)
    for value in shap_importance.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_train_model_sets_scale_pos_weight_from_class_imbalance():
    X, y = _make_dataset(n=200, seed=1)
    model, _, _ = train_model(X, y, PARAMS, FIXED_PARAMS)
    expected = float((y == 0).sum() / (y == 1).sum())
    assert model.get_params()["scale_pos_weight"] == expected


def test_train_model_is_deterministic_given_random_state():
    X, y = _make_dataset()
    model1, _, shap1 = train_model(X, y, PARAMS, FIXED_PARAMS, random_state=7)
    model2, _, shap2 = train_model(X, y, PARAMS, FIXED_PARAMS, random_state=7)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))
    assert shap1 == shap2


def test_select_threshold_via_cv_returns_valid_float():
    X, y = _make_dataset()
    threshold = select_threshold_via_cv(X, y, PARAMS, FIXED_PARAMS, target_recall=0.8, n_folds=5)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


def test_select_threshold_via_cv_is_deterministic_given_random_state():
    X, y = _make_dataset()
    t1 = select_threshold_via_cv(
        X, y, PARAMS, FIXED_PARAMS, target_recall=0.8, n_folds=5, random_state=7
    )
    t2 = select_threshold_via_cv(
        X, y, PARAMS, FIXED_PARAMS, target_recall=0.8, n_folds=5, random_state=7
    )
    assert t1 == t2
