"""Unit tests for model scoring (no GCP dependencies — model is fit in-memory)."""

import pandas as pd
import xgboost as xgb
from serving.predict import score


def _fit_tiny_model() -> tuple[xgb.XGBClassifier, list[str]]:
    X = pd.DataFrame(
        {
            "avg_transaction_30d": [10.0, 50.0, 5.0, 60.0],
            "member_since_days": [100, 900, 50, 1000],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)
    return model, list(X.columns)


def test_score_returns_probability_per_row():
    model, feature_names = _fit_tiny_model()
    df = pd.DataFrame(
        {
            "avg_transaction_30d": [55.0, 8.0],
            "member_since_days": [950, 80],
            "customer_id": ["d1", "d2"],  # extra column present in the BQ snapshot, not a feature
        }
    )
    proba = score(model, df, feature_names)
    assert len(proba) == 2
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_score_ignores_column_order(monkeypatch=None):
    model, feature_names = _fit_tiny_model()
    df_ordered = pd.DataFrame({"avg_transaction_30d": [55.0], "member_since_days": [950]})
    df_reordered = pd.DataFrame({"member_since_days": [950], "avg_transaction_30d": [55.0]})
    proba_ordered = score(model, df_ordered, feature_names)
    proba_reordered = score(model, df_reordered, feature_names)
    assert proba_ordered[0] == proba_reordered[0]
