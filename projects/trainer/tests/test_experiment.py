"""Unit tests for the training-time score baseline (no GCS/Vertex dependencies)."""

import numpy as np
import pandas as pd
import xgboost as xgb
from ml_common.config import CHURN_PROBABILITY_FIELD
from ml_common.drift import compute_baseline_stats, compute_psi


def _fit_tiny_model() -> tuple[xgb.XGBClassifier, pd.DataFrame]:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "avg_transaction_30d": rng.normal(50, 10, 500),
            "avg_engagement_30d": rng.normal(0, 1, 500),
        }
    )
    y = pd.Series(rng.randint(0, 2, 500))
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3)
    model.fit(X, y)
    return model, X


def test_score_baseline_is_added_alongside_feature_baseline():
    model, X = _fit_tiny_model()

    train_scores = pd.DataFrame({CHURN_PROBABILITY_FIELD: model.predict_proba(X)[:, 1]})
    baseline_stats = compute_baseline_stats(X)
    baseline_stats.update(compute_baseline_stats(train_scores))

    assert "avg_transaction_30d" in baseline_stats
    assert CHURN_PROBABILITY_FIELD in baseline_stats
    assert baseline_stats[CHURN_PROBABILITY_FIELD]["type"] == "numeric"


def test_score_baseline_round_trips_through_compute_psi():
    model, X = _fit_tiny_model()
    train_scores = pd.DataFrame({CHURN_PROBABILITY_FIELD: model.predict_proba(X)[:, 1]})
    baseline_stats = compute_baseline_stats(train_scores)

    # Scoring the same training data again should show ~zero drift against its own baseline
    # (well under the 0.2 production PSI_THRESHOLD) — some residual noise is expected from
    # decile-edge quantization, not a hard zero.
    psi = compute_psi(baseline_stats, train_scores)[CHURN_PROBABILITY_FIELD]

    assert psi < 0.05
