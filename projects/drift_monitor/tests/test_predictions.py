"""Unit tests for fetching the champion's latest production scores (no GCP credentials required)."""

import drift_monitor.predictions as predictions_module
import pandas as pd
from drift_monitor.predictions import fetch_latest_predictions
from google.cloud.aiplatform_v1.types.job_state import JobState

_CHAMPION = "projects/p/locations/l/models/123"
_DISPLAY_NAME = "daily-churn-scoring"


class _FakeGcaResource:
    def __init__(self, state, model):
        self.state = state
        self.model = model


class _FakeOutputInfo:
    def __init__(self, bigquery_output_dataset, bigquery_output_table):
        self.bigquery_output_dataset = bigquery_output_dataset
        self.bigquery_output_table = bigquery_output_table


class _FakeJob:
    def __init__(self, display_name, state, model, output_info=None):
        self.display_name = display_name
        self.gca_resource = _FakeGcaResource(state, model)
        self.output_info = output_info


class _FakeQueryResult:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


class _FakeBqClient:
    def __init__(self, project=None):
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        return _FakeQueryResult(pd.DataFrame({"churn_probability": [0.1, 0.9]}))


def test_fetch_latest_predictions_returns_none_when_no_matching_job(monkeypatch):
    monkeypatch.setattr(
        predictions_module.aiplatform.BatchPredictionJob, "list", staticmethod(lambda **kwargs: [])
    )
    assert fetch_latest_predictions("proj", _CHAMPION, _DISPLAY_NAME) is None


def test_fetch_latest_predictions_skips_unrelated_evaluation_job(monkeypatch):
    # A post_training evaluate-stage batch predict against the same champion model, but with a
    # different display_name — must not be mistaken for the daily production-scoring job.
    evaluation_job = _FakeJob(
        "churn-training-pipeline-batch-predict",
        JobState.JOB_STATE_SUCCEEDED,
        _CHAMPION,
        _FakeOutputInfo("bq://proj.ml", "predictions_111"),
    )
    daily_job = _FakeJob(
        _DISPLAY_NAME,
        JobState.JOB_STATE_SUCCEEDED,
        _CHAMPION,
        _FakeOutputInfo("bq://proj.ml", "predictions_222"),
    )
    monkeypatch.setattr(
        predictions_module.aiplatform.BatchPredictionJob,
        "list",
        staticmethod(lambda **kwargs: [evaluation_job, daily_job]),
    )
    monkeypatch.setattr(predictions_module.bigquery, "Client", _FakeBqClient)

    result = fetch_latest_predictions("proj", _CHAMPION, _DISPLAY_NAME)

    assert result is not None
    assert list(result["churn_probability"]) == [0.1, 0.9]


def test_fetch_latest_predictions_skips_jobs_for_a_different_model(monkeypatch):
    other_model_job = _FakeJob(
        _DISPLAY_NAME,
        JobState.JOB_STATE_SUCCEEDED,
        "projects/p/locations/l/models/999",
        _FakeOutputInfo("bq://proj.ml", "predictions_333"),
    )
    monkeypatch.setattr(
        predictions_module.aiplatform.BatchPredictionJob,
        "list",
        staticmethod(lambda **kwargs: [other_model_job]),
    )
    assert fetch_latest_predictions("proj", _CHAMPION, _DISPLAY_NAME) is None


def test_fetch_latest_predictions_skips_non_succeeded_jobs(monkeypatch):
    running_job = _FakeJob(
        _DISPLAY_NAME,
        JobState.JOB_STATE_RUNNING,
        _CHAMPION,
        _FakeOutputInfo("bq://proj.ml", "predictions_444"),
    )
    monkeypatch.setattr(
        predictions_module.aiplatform.BatchPredictionJob,
        "list",
        staticmethod(lambda **kwargs: [running_job]),
    )
    assert fetch_latest_predictions("proj", _CHAMPION, _DISPLAY_NAME) is None


def test_fetch_latest_predictions_builds_full_table_from_dataset_and_table(monkeypatch):
    job = _FakeJob(
        _DISPLAY_NAME,
        JobState.JOB_STATE_SUCCEEDED,
        _CHAMPION,
        _FakeOutputInfo("bq://my-proj.ml", "predictions_20260822120000"),
    )
    monkeypatch.setattr(
        predictions_module.aiplatform.BatchPredictionJob,
        "list",
        staticmethod(lambda **kwargs: [job]),
    )
    fake_client = _FakeBqClient()
    monkeypatch.setattr(predictions_module.bigquery, "Client", lambda project=None: fake_client)

    fetch_latest_predictions("my-proj", _CHAMPION, _DISPLAY_NAME)

    assert "my-proj.ml.predictions_20260822120000" in fake_client.queries[0]


def test_fetch_latest_predictions_extracts_prediction_from_json_string(monkeypatch):
    # The model's response lands in the "prediction" column as a JSON-encoded STRING, not a
    # nested STRUCT (confirmed against a real BatchPredictionJob's BigQuery output) — the
    # query must extract it with JSON_VALUE rather than assume dot-access works.
    job = _FakeJob(
        _DISPLAY_NAME,
        JobState.JOB_STATE_SUCCEEDED,
        _CHAMPION,
        _FakeOutputInfo("bq://proj.ml", "predictions_555"),
    )
    monkeypatch.setattr(
        predictions_module.aiplatform.BatchPredictionJob,
        "list",
        staticmethod(lambda **kwargs: [job]),
    )
    fake_client = _FakeBqClient()
    monkeypatch.setattr(predictions_module.bigquery, "Client", lambda project=None: fake_client)

    fetch_latest_predictions("proj", _CHAMPION, _DISPLAY_NAME)

    assert "JSON_VALUE(prediction, '$.churn_probability')" in fake_client.queries[0]
