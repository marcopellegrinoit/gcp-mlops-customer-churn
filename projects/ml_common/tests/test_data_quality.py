"""Unit tests for the data-quality assertions that gate retraining."""

import numpy as np
import pandas as pd
import pytest
from ml_common.data_quality import check_data_quality
from ml_common.drift import compute_baseline_stats


@pytest.fixture()
def snapshot() -> pd.DataFrame:
    rng = np.random.RandomState(5)
    engagement = rng.beta(2, 5, 1000)
    engagement[rng.rand(1000) < 0.10] = np.nan
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(1000)],
            "avg_engagement_30d": engagement,
            "avg_transaction_30d": rng.normal(50, 10, 1000),
            "membership_tier": rng.choice(["friend", "champion", "guardian"], 1000),
        }
    ).drop(columns=["customer_id"])


@pytest.fixture()
def baseline(snapshot) -> dict:
    return compute_baseline_stats(snapshot)


def _checks(report: dict) -> set[str]:
    return {f["check"] for f in report["failures"]}


def test_healthy_snapshot_passes(snapshot, baseline):
    report = check_data_quality(snapshot, baseline, [1000, 1010, 990], 0)
    assert report["data_quality_failed"] is False
    assert report["failures"] == []
    assert report["row_count"] == 1000


def test_short_load_fails(snapshot, baseline):
    # A partial upstream load moves every distribution, and would otherwise read as drift.
    partial = snapshot.head(200)
    report = check_data_quality(partial, baseline, [1000, 1010, 990], 0)
    assert report["data_quality_failed"] is True
    assert "row_volume" in _checks(report)


def test_no_history_skips_the_row_volume_check(snapshot, baseline):
    report = check_data_quality(snapshot.head(50), baseline, [], 0)
    assert "row_volume" not in _checks(report)


def test_duplicate_customer_keys_fail(snapshot, baseline):
    report = check_data_quality(snapshot, baseline, [1000], 12)
    assert report["data_quality_failed"] is True
    assert "duplicate_keys" in _checks(report)


def test_missing_feature_column_fails(snapshot, baseline):
    report = check_data_quality(snapshot.drop(columns=["avg_transaction_30d"]), baseline, [1000], 0)
    assert report["data_quality_failed"] is True
    assert "missing_columns" in _checks(report)


def test_null_rate_explosion_fails(snapshot, baseline):
    # A broken join, not a population shift: this must halt rather than retrain.
    broken = snapshot.copy()
    broken["avg_engagement_30d"] = np.nan
    report = check_data_quality(broken, baseline, [1000], 0)
    assert report["data_quality_failed"] is True
    assert "null_rate" in _checks(report)


def test_modest_null_rate_movement_passes(snapshot, baseline):
    drifted = snapshot.copy()
    values = drifted["avg_engagement_30d"].to_numpy(copy=True)
    values[:150] = np.nan  # 10% -> ~25%, a real shift but not a broken pipeline
    drifted["avg_engagement_30d"] = values
    report = check_data_quality(drifted, baseline, [1000], 0)
    assert "null_rate" not in _checks(report)


def test_collapsed_column_fails(snapshot, baseline):
    # An upstream default written into every row: PSI is least reliable exactly here.
    collapsed = snapshot.copy()
    collapsed["avg_transaction_30d"] = 42.0
    report = check_data_quality(collapsed, baseline, [1000], 0)
    assert report["data_quality_failed"] is True
    assert "collapsed_column" in _checks(report)


def test_column_that_was_already_constant_is_not_reported_collapsed():
    constant = pd.DataFrame({"is_trial_account": [0.0] * 500})
    baseline = compute_baseline_stats(constant)
    report = check_data_quality(constant, baseline, [500], 0)
    assert report["data_quality_failed"] is False


def test_every_failure_is_reported_not_just_the_first(snapshot, baseline):
    broken = snapshot.head(100).copy()
    broken["avg_engagement_30d"] = np.nan
    broken["avg_transaction_30d"] = 1.0
    report = check_data_quality(broken, baseline, [1000, 1000], 3)
    assert {"row_volume", "duplicate_keys", "null_rate", "collapsed_column"} <= _checks(report)
