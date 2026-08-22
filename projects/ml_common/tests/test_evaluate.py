"""Unit tests for evaluate stage logic (no GCP dependencies)."""

import numpy as np
import pytest
from ml_common.evaluate import _spearman_rank_correlation, select_threshold

# ---------------------------------------------------------------------------
# select_threshold
# ---------------------------------------------------------------------------


def _make_binary(n_pos=40, n_neg=160, seed=0):
    """Synthetic y_true and perfectly-ordered probabilities."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    # Higher proba for positives so there exist meaningful thresholds
    proba = np.concatenate([rng.uniform(0.6, 1.0, n_pos), rng.uniform(0.0, 0.5, n_neg)])
    return y, proba


def test_select_threshold_achieves_target_recall():
    y, proba = _make_binary()
    threshold = select_threshold(y, proba, target_recall=0.80)
    preds = (proba >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    recall = tp / (tp + fn)
    assert recall >= 0.80


def test_select_threshold_prefers_higher_precision_among_candidates():
    y, proba = _make_binary()
    t1 = select_threshold(y, proba, target_recall=0.80)
    t2 = select_threshold(y, proba, target_recall=0.50)
    # Stricter recall target → lower threshold → typically lower precision;
    # looser target allows a higher threshold with higher precision.
    preds1 = (proba >= t1).astype(int)
    preds2 = (proba >= t2).astype(int)
    tp1 = np.sum((preds1 == 1) & (y == 1))
    tp2 = np.sum((preds2 == 1) & (y == 1))
    prec1 = tp1 / max(preds1.sum(), 1)
    prec2 = tp2 / max(preds2.sum(), 1)
    # The looser-recall variant should have at least as much precision
    assert prec2 >= prec1 - 1e-6


def test_select_threshold_fallback_when_no_candidate_meets_recall():
    """When no threshold achieves the target recall, fall back to max-recall threshold."""
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.4, 0.3, 0.2, 0.1])
    # target_recall=1.0 is impossible to guarantee at threshold > min(proba)
    threshold = select_threshold(y, proba, target_recall=1.0)
    # Fallback: should still return a valid float in [0, 1]
    assert 0.0 <= threshold <= 1.0


def test_select_threshold_returns_float():
    y, proba = _make_binary()
    result = select_threshold(y, proba, target_recall=0.80)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _spearman_rank_correlation
# ---------------------------------------------------------------------------


def test_spearman_perfect_positive_correlation():
    a = {"f1": 3.0, "f2": 2.0, "f3": 1.0}
    b = {"f1": 9.0, "f2": 6.0, "f3": 3.0}
    corr = _spearman_rank_correlation(a, b)
    assert corr == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    a = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
    b = {"f1": 3.0, "f2": 2.0, "f3": 1.0}
    corr = _spearman_rank_correlation(a, b)
    assert corr == pytest.approx(-1.0)


def test_spearman_returns_none_when_baseline_is_none():
    a = {"f1": 1.0, "f2": 2.0}
    assert _spearman_rank_correlation(a, None) is None


def test_spearman_returns_none_when_fewer_than_two_shared_features():
    a = {"f1": 1.0}
    b = {"f1": 2.0}
    assert _spearman_rank_correlation(a, b) is None


def test_spearman_ignores_features_not_in_both_dicts():
    a = {"f1": 3.0, "f2": 2.0, "f3": 1.0, "fx": 99.0}
    b = {"f1": 9.0, "f2": 6.0, "f3": 3.0}
    corr = _spearman_rank_correlation(a, b)
    assert corr == pytest.approx(1.0)


def test_spearman_result_in_valid_range():
    rng = np.random.default_rng(42)
    features = [f"f{i}" for i in range(10)]
    a = {f: float(v) for f, v in zip(features, rng.random(10), strict=True)}
    b = {f: float(v) for f, v in zip(features, rng.random(10), strict=True)}
    corr = _spearman_rank_correlation(a, b)
    assert corr is not None
    assert -1.0 <= corr <= 1.0
