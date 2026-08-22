"""Unit tests for the drift detection orchestration stage (no GCP credentials required)."""

import drift_monitor.detect as detect_module
import numpy as np
import pandas as pd
from drift_monitor.detect import run_drift_check
from ml_common.drift import compute_baseline_stats

_DISPLAY_NAME = "daily-churn-scoring"


class _FakeChampion:
    def __init__(
        self, resource_name="projects/p/locations/l/models/123", uri="gs://bucket/artifacts/abc"
    ):
        self.resource_name = resource_name
        self.uri = uri


def _baseline_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({"avg_transaction_30d": rng.normal(50, 10, 500)})


def _run(monkeypatch, metadata, feature_df, fetch_predictions=None):
    monkeypatch.setattr(detect_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(detect_module, "fetch_champion", lambda model_display_name: _FakeChampion())
    monkeypatch.setattr(detect_module, "download_json", lambda uri: metadata)
    monkeypatch.setattr(
        detect_module,
        "fetch_latest_snapshot",
        lambda project_id, table: ("2026-06-18", feature_df),
    )
    if fetch_predictions is not None:
        monkeypatch.setattr(detect_module, "fetch_latest_predictions", fetch_predictions)
    uploaded = {}
    monkeypatch.setattr(
        detect_module, "upload_json", lambda uri, obj: uploaded.update(uri=uri, obj=obj)
    )

    result = run_drift_check(
        project_id="proj",
        region="europe-west1",
        bq_features_table="features.customer_features",
        model_display_name="churn-predictor",
        decision_gcs_uri="gs://bucket/drift/latest.json",
        psi_threshold=0.2,
        batch_predict_display_name=_DISPLAY_NAME,
    )
    return result, uploaded


def test_run_drift_check_returns_no_champion_reason_when_none_registered(monkeypatch):
    monkeypatch.setattr(detect_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(detect_module, "fetch_champion", lambda model_display_name: None)

    uploaded = {}
    monkeypatch.setattr(
        detect_module, "upload_json", lambda uri, obj: uploaded.update(uri=uri, obj=obj)
    )

    result = run_drift_check(
        project_id="proj",
        region="europe-west1",
        bq_features_table="features.customer_features",
        model_display_name="churn-predictor",
        decision_gcs_uri="gs://bucket/drift/latest.json",
        psi_threshold=0.2,
        batch_predict_display_name=_DISPLAY_NAME,
    )

    assert result == {"drift_detected": False, "reason": "no_champion_registered"}
    assert uploaded["uri"] == "gs://bucket/drift/latest.json"
    assert uploaded["obj"] == result


def test_run_drift_check_reports_no_drift_when_distribution_stable(monkeypatch):
    baseline_df = _baseline_df()
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": compute_baseline_stats(baseline_df),
    }

    result, uploaded = _run(
        monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: None
    )

    assert result["drift_detected"] is False
    assert result["champion_model"] == "projects/p/locations/l/models/123"
    assert result["snapshot_date"] == "2026-06-18"
    assert uploaded["obj"] == result


def test_run_drift_check_reports_drift_when_distribution_shifts(monkeypatch):
    baseline_df = _baseline_df()
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": compute_baseline_stats(baseline_df),
    }
    shifted_df = baseline_df.copy()
    shifted_df["avg_transaction_30d"] = shifted_df["avg_transaction_30d"] + 100

    result, _ = _run(monkeypatch, metadata, shifted_df, fetch_predictions=lambda *a, **kw: None)

    assert result["drift_detected"] is True
    assert "avg_transaction_30d" in result["breached_features"]


def test_run_drift_check_adds_score_psi_without_affecting_drift_detected(monkeypatch):
    baseline_df = _baseline_df()
    rng = np.random.RandomState(7)
    train_scores = pd.DataFrame({"churn_probability": rng.uniform(0, 1, 500)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }
    # Same distribution as the baseline -> stable, but present as its own field.
    stable_scores = pd.DataFrame({"churn_probability": rng.uniform(0, 1, 500)})

    result, _ = _run(
        monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: stable_scores
    )

    assert result["drift_detected"] is False
    assert "score_psi" in result
    assert result["score_drift_detected"] is False


def test_run_drift_check_flags_score_drift_when_scores_shift(monkeypatch):
    baseline_df = _baseline_df()
    rng = np.random.RandomState(7)
    train_scores = pd.DataFrame({"churn_probability": rng.uniform(0, 0.3, 500)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }
    shifted_scores = pd.DataFrame({"churn_probability": rng.uniform(0.7, 1.0, 500)})

    result, _ = _run(
        monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: shifted_scores
    )

    # Score drift never flips the feature-only drift_detected flag that gates retraining.
    assert result["drift_detected"] is False
    assert result["score_drift_detected"] is True


def test_run_drift_check_skips_score_check_when_predictions_unavailable(monkeypatch):
    baseline_df = _baseline_df()
    train_scores = pd.DataFrame({"churn_probability": np.random.RandomState(1).uniform(0, 1, 200)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }

    result, _ = _run(monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: None)

    assert "score_psi" not in result
    assert "score_drift_detected" not in result


def test_run_drift_check_skips_score_check_on_empty_predictions(monkeypatch):
    baseline_df = _baseline_df()
    train_scores = pd.DataFrame({"churn_probability": np.random.RandomState(1).uniform(0, 1, 200)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }
    empty = pd.DataFrame({"churn_probability": []})

    result, _ = _run(monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: empty)

    assert "score_psi" not in result


def test_run_drift_check_survives_score_fetch_exception(monkeypatch):
    baseline_df = _baseline_df()
    train_scores = pd.DataFrame({"churn_probability": np.random.RandomState(1).uniform(0, 1, 200)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }

    def _boom(*args, **kwargs):
        raise RuntimeError("BigQuery permission denied")

    result, uploaded = _run(monkeypatch, metadata, baseline_df, fetch_predictions=_boom)

    assert result["drift_detected"] is False
    assert "score_psi" not in result
    assert uploaded["obj"] == result


def test_run_drift_check_survives_compute_psi_failure_after_successful_fetch(monkeypatch):
    # fetch_latest_predictions succeeds but returns data compute_psi can't digest (e.g. a
    # non-numeric value slipping through) — the try/except must cover this too, not just the
    # fetch call, or a downstream failure here would take the feature-PSI result down with it.
    baseline_df = _baseline_df()
    train_scores = pd.DataFrame({"churn_probability": np.random.RandomState(1).uniform(0, 1, 200)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }
    unusable_scores = pd.DataFrame({"churn_probability": ["not-a-number", "also-not-a-number"]})

    result, uploaded = _run(
        monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: unusable_scores
    )

    assert result["drift_detected"] is False
    assert "score_psi" not in result
    assert uploaded["obj"] == result


def test_run_drift_check_skips_score_check_when_scores_are_all_null(monkeypatch):
    baseline_df = _baseline_df()
    train_scores = pd.DataFrame({"churn_probability": np.random.RandomState(1).uniform(0, 1, 200)})
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": {
            **compute_baseline_stats(baseline_df),
            **compute_baseline_stats(train_scores),
        },
    }
    # Rows present, but no usable score in any of them (e.g. a serving-side scoring defect) —
    # must be skipped like the empty case, not scored as a spurious 100%-drifted distribution.
    all_null_scores = pd.DataFrame({"churn_probability": [None, None, None]})

    result, _ = _run(
        monkeypatch, metadata, baseline_df, fetch_predictions=lambda *a, **kw: all_null_scores
    )

    assert "score_psi" not in result
    assert "score_drift_detected" not in result


def test_run_drift_check_skips_score_check_for_pre_change_artifacts(monkeypatch):
    # metadata frozen by a model trained before the score-baseline was added: no
    # churn_probability key at all. fetch_latest_predictions must not even be called.
    baseline_df = _baseline_df()
    metadata = {
        "feature_names": ["avg_transaction_30d"],
        "baseline_stats": compute_baseline_stats(baseline_df),
    }

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("fetch_latest_predictions should not be called")

    result, _ = _run(monkeypatch, metadata, baseline_df, fetch_predictions=_should_not_be_called)

    assert "score_psi" not in result
