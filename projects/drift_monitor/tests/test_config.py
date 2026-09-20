"""Unit tests for the drift monitor's environment configuration."""

import pytest
from drift_monitor.config import DriftMonitorSettings, get_settings
from pydantic import ValidationError

_REQUIRED = {
    "BQ_PROJECT_ID": "proj",
    "REGION": "europe-west1",
    "BQ_FEATURES_TABLE": "features.customer_features",
    "MODEL_DISPLAY_NAME": "churn-predictor",
    "GCS_BUCKET": "pipeline-metadata",
    "DECISION_BLOB": "drift/latest.json",
    "BATCH_PREDICT_DISPLAY_NAME": "daily-churn-scoring",
}


@pytest.fixture()
def deployed_env(monkeypatch):
    for name, value in _REQUIRED.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_reads_the_deployed_environment(deployed_env):
    settings = get_settings()
    assert settings.project_id == "proj"
    assert settings.decision_gcs_uri == "gs://proj-pipeline-metadata/drift/latest.json"
    assert settings.staging_bucket_uri == "gs://proj-pipeline-metadata"


def test_missing_required_variable_fails_at_startup(deployed_env, monkeypatch):
    # The regression this replaces: os.environ.get() returned None, the job ran to
    # completion against gs://None-None/None, and a silent no-op looked like a quiet night.
    monkeypatch.delenv("GCS_BUCKET")
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="gcs_bucket"):
        get_settings()


def test_tuning_values_are_read_from_the_environment(deployed_env, monkeypatch):
    monkeypatch.setenv("PSI_THRESHOLD", "0.35")
    monkeypatch.setenv("PERSISTENCE_WINDOW", "5")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.psi_threshold == 0.35
    assert settings.persistence_window == 5


def test_unsatisfiable_persistence_rule_is_rejected():
    # More breaches required than there are runs to breach in: drift would never once
    # trigger retraining, and the only symptom would be silence.
    with pytest.raises(ValidationError, match="no run could ever trigger retraining"):
        DriftMonitorSettings(
            project_id="proj",
            region="europe-west1",
            bq_features_table="features.customer_features",
            model_display_name="churn-predictor",
            gcs_bucket="pipeline-metadata",
            decision_blob="drift/latest.json",
            batch_predict_display_name="daily-churn-scoring",
            persistence_window=2,
            persistence_min_breaches=3,
        )
