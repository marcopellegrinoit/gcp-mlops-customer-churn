"""Unit tests for the drift detection orchestration stage (no GCP credentials required)."""

import drift_monitor.detect as detect_module
import numpy as np
import pandas as pd
from drift_monitor.detect import run_drift_check
from ml_common.drift import compute_baseline_stats


class _FakeChampion:
    def __init__(
        self, resource_name="projects/p/locations/l/models/123", uri="gs://bucket/artifacts/abc"
    ):
        self.resource_name = resource_name
        self.uri = uri


def _baseline_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({"avg_transaction_30d": rng.normal(50, 10, 500)})


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

    monkeypatch.setattr(detect_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(detect_module, "fetch_champion", lambda model_display_name: _FakeChampion())
    monkeypatch.setattr(detect_module, "download_json", lambda uri: metadata)
    monkeypatch.setattr(
        detect_module,
        "fetch_latest_snapshot",
        lambda project_id, table: ("2026-06-18", baseline_df),
    )
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

    monkeypatch.setattr(detect_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(detect_module, "fetch_champion", lambda model_display_name: _FakeChampion())
    monkeypatch.setattr(detect_module, "download_json", lambda uri: metadata)
    monkeypatch.setattr(
        detect_module, "fetch_latest_snapshot", lambda project_id, table: ("2026-06-18", shifted_df)
    )
    monkeypatch.setattr(detect_module, "upload_json", lambda uri, obj: None)

    result = run_drift_check(
        project_id="proj",
        region="europe-west1",
        bq_features_table="features.customer_features",
        model_display_name="churn-predictor",
        decision_gcs_uri="gs://bucket/drift/latest.json",
        psi_threshold=0.2,
    )

    assert result["drift_detected"] is True
    assert "avg_transaction_30d" in result["breached_features"]
