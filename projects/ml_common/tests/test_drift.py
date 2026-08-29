"""Unit tests for PSI drift detection."""

import numpy as np
import pandas as pd
import pytest
from ml_common.drift import compute_baseline_stats, compute_psi, evaluate_drift


@pytest.fixture()
def baseline_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "avg_transaction_30d": rng.normal(loc=50, scale=10, size=1000),
            "membership_tier": rng.choice(
                ["friend", "champion", "guardian"], size=1000, p=[0.6, 0.3, 0.1]
            ),
        }
    )


def test_compute_baseline_stats_numeric_and_categorical(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    assert stats["avg_transaction_30d"]["type"] == "numeric"
    assert len(stats["avg_transaction_30d"]["bin_edges"]) > 1
    assert stats["membership_tier"]["type"] == "categorical"
    assert pytest.approx(sum(stats["membership_tier"]["frequencies"].values()), abs=1e-9) == 1.0


def test_psi_zero_when_distribution_unchanged(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    scores = compute_psi(stats, baseline_df)
    assert scores["avg_transaction_30d"] < 0.05
    assert scores["membership_tier"] < 0.05


def test_psi_high_when_numeric_distribution_shifts(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    shifted = baseline_df.copy()
    shifted["avg_transaction_30d"] = shifted["avg_transaction_30d"] + 100  # large mean shift
    scores = compute_psi(stats, shifted)
    assert scores["avg_transaction_30d"] > 0.2


def test_psi_high_when_categorical_distribution_shifts(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    shifted = baseline_df.copy()
    shifted["membership_tier"] = "guardian"  # collapse to a single category
    scores = compute_psi(stats, shifted)
    assert scores["membership_tier"] > 0.2


def test_psi_handles_unseen_category(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    shifted = baseline_df.copy()
    shifted["membership_tier"] = "brand_new_tier"
    scores = compute_psi(stats, shifted)
    assert scores["membership_tier"] > 0


def test_evaluate_drift_not_detected_when_stable(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    result = evaluate_drift(stats, baseline_df, psi_threshold=0.2)
    assert result["drift_detected"] is False
    assert result["breached_features"] == {}


def test_evaluate_drift_detected_when_shifted(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    shifted = baseline_df.copy()
    shifted["avg_transaction_30d"] = shifted["avg_transaction_30d"] + 100
    result = evaluate_drift(stats, shifted, psi_threshold=0.2)
    assert result["drift_detected"] is True
    assert "avg_transaction_30d" in result["breached_features"]


def test_evaluate_drift_ignores_columns_missing_from_current(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    partial = baseline_df.drop(columns=["membership_tier"])
    result = evaluate_drift(stats, partial, psi_threshold=0.2)
    assert "membership_tier" not in result["feature_psi"]


def test_psi_numeric_handles_constant_baseline_column():
    constant_df = pd.DataFrame({"is_trial_account": [0.0] * 1000})
    stats = compute_baseline_stats(constant_df)
    assert stats["is_trial_account"]["bin_edges"] == [-np.inf, np.inf]
    scores = compute_psi(stats, constant_df)
    assert scores["is_trial_account"] == 0.0
