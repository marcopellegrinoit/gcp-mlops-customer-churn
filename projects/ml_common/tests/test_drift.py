"""Unit tests for PSI drift detection."""

import numpy as np
import pandas as pd
import pytest
from ml_common.drift import (
    compute_baseline_stats,
    compute_psi,
    evaluate_drift,
    psi_noise_floor,
)


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
    # A constant column has one distinct value, so it takes the discrete path and its
    # frequency table is the single value — no bin edges are involved at all.
    constant_df = pd.DataFrame({"is_trial_account": [0.0] * 1000})
    stats = compute_baseline_stats(constant_df)
    assert stats["is_trial_account"] == {
        "type": "discrete",
        "frequencies": {"0.0": 1.0},
        "monitored": False,
    }
    scores = compute_psi(stats, constant_df)
    assert scores["is_trial_account"] == 0.0


def test_psi_numeric_handles_stale_single_edge_baseline():
    # Baselines frozen before the near-constant-column fix can still carry a
    # single-element bin_edges array (e.g. [inf]) in already-registered model artifacts.
    stats = {"days_since_last_successful_payment": {"type": "numeric", "bin_edges": [np.inf]}}
    current = pd.DataFrame({"days_since_last_successful_payment": [0.0] * 1000})
    scores = compute_psi(stats, current)
    assert scores["days_since_last_successful_payment"] == 0.0


@pytest.fixture()
def discrete_df() -> pd.DataFrame:
    """A zero-inflated count column and a 0/1-valued rate column — the real feature shapes.

    Most numeric features customer_features produces look like this rather than like a
    clean Gaussian: renewal_count/contact_requests_last_30d are mostly 0, and the rate
    columns collapse to exactly 0.0 or 1.0 for customers with a single event.
    """
    rng = np.random.RandomState(11)
    return pd.DataFrame(
        {
            "renewal_count": rng.poisson(0.4, 2000).astype(float),
            "payment_failure_rate_30d": rng.choice([0.0, 0.5, 1.0], 2000, p=[0.6, 0.2, 0.2]),
        }
    )


def test_no_drift_against_own_training_data(discrete_df):
    # The regression this suite previously missed: decile binning collapses on a mass
    # point, and assuming the surviving buckets are equal-frequency made a baseline
    # breach against the exact rows it was built from — firing a retrain every night.
    stats = compute_baseline_stats(discrete_df)
    result = evaluate_drift(stats, discrete_df, psi_threshold=0.2)
    assert result["drift_detected"] is False
    assert result["breached_features"] == {}
    assert result["unmonitored_features"] == []


def test_no_drift_against_a_fresh_identically_distributed_sample(discrete_df):
    rng = np.random.RandomState(12)
    fresh = pd.DataFrame(
        {
            "renewal_count": rng.poisson(0.4, 2000).astype(float),
            "payment_failure_rate_30d": rng.choice([0.0, 0.5, 1.0], 2000, p=[0.6, 0.2, 0.2]),
        }
    )
    stats = compute_baseline_stats(discrete_df)
    assert evaluate_drift(stats, fresh, psi_threshold=0.2)["drift_detected"] is False


def test_discrete_column_still_detects_a_real_shift(discrete_df):
    stats = compute_baseline_stats(discrete_df)
    shifted = discrete_df.copy()
    shifted["renewal_count"] = shifted["renewal_count"] + 2  # every customer gains renewals
    result = evaluate_drift(stats, shifted, psi_threshold=0.2)
    assert "renewal_count" in result["breached_features"]


def test_low_cardinality_numeric_uses_a_frequency_table(discrete_df):
    stats = compute_baseline_stats(discrete_df)
    assert stats["renewal_count"]["type"] == "discrete"
    assert "bin_edges" not in stats["renewal_count"]
    # Keys survive a metadata.json round-trip as strings.
    assert set(stats["renewal_count"]["frequencies"]) == {"0.0", "1.0", "2.0", "3.0", "4.0"}


def test_continuous_column_stores_measured_proportions(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    spec = stats["avg_transaction_30d"]
    assert spec["type"] == "numeric"
    assert len(spec["expected_pct"]) == len(spec["bin_edges"]) - 1
    assert pytest.approx(sum(spec["expected_pct"]), abs=1e-9) == 1.0


def test_constant_column_is_reported_unmonitored_not_healthy():
    # PSI on a single-bucket column is structurally 0.0, which reads as "stable" when it
    # really means "no signal" — it must not be silently counted as a passing feature.
    constant_df = pd.DataFrame({"is_trial_account": [0.0] * 1000})
    stats = compute_baseline_stats(constant_df)
    assert stats["is_trial_account"]["monitored"] is False
    result = evaluate_drift(stats, constant_df, psi_threshold=0.2)
    assert result["unmonitored_features"] == ["is_trial_account"]
    assert result["drift_detected"] is False


def test_legacy_baseline_without_proportions_is_unmonitored():
    # Artifacts frozen before expected_pct was stored cannot be evaluated without
    # re-assuming uniformity — the defect itself. They stay out of the breach decision
    # until rebuilt from ml.split_assignments.
    stats = {"avg_transaction_30d": {"type": "numeric", "bin_edges": [-np.inf, 10.0, np.inf]}}
    current = pd.DataFrame({"avg_transaction_30d": [500.0] * 1000})
    result = evaluate_drift(stats, current, psi_threshold=0.2)
    assert result["unmonitored_features"] == ["avg_transaction_30d"]
    assert result["drift_detected"] is False


def test_noise_floor_raises_the_threshold_for_small_samples(baseline_df):
    stats = compute_baseline_stats(baseline_df)
    spec = stats["avg_transaction_30d"]
    assert psi_noise_floor(spec, 50) > psi_noise_floor(spec, 5000)


def test_small_sample_noise_does_not_trigger_drift(baseline_df):
    # 40 rows drawn from the baseline itself: raw PSI is large, but it is all sampling
    # noise, and the per-feature threshold must absorb it.
    stats = compute_baseline_stats(baseline_df)
    tiny = baseline_df.sample(n=40, random_state=3)
    result = evaluate_drift(stats, tiny, psi_threshold=0.2)
    assert result["feature_thresholds"]["avg_transaction_30d"] > 0.2
    assert result["drift_detected"] is False
