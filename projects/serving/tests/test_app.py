"""Unit tests for the Vertex AI custom-container prediction contract (FastAPI app)."""

import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from ml_common.contracts import CHURN_PREDICTION_FIELD, CHURN_PROBABILITY_FIELD


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AIP_STORAGE_URI", "gs://bucket/artifacts/abc123")
    monkeypatch.setenv("THRESHOLD", "0.5")
    monkeypatch.setenv("AIP_HEALTH_ROUTE", "/health")
    monkeypatch.setenv("AIP_PREDICT_ROUTE", "/predict")
    monkeypatch.setenv("PROJECT_ID", "test-project")

    # Enough rows to actually split on. Four rows left the trees unable to meet
    # min_child_weight, so the model returned base_score 0.5 for every input — which would
    # have made a "predictions must vary" assertion silently untestable, the same class of
    # blind spot as the constant-prediction defect these tests now cover.
    low = [float(v) for v in range(0, 100)]
    high = [float(v) for v in range(100, 200)]
    X = pd.DataFrame({"avg_transaction_30d": low + high})
    y = pd.Series([0] * len(low) + [1] * len(high))
    model = xgb.XGBClassifier(n_estimators=20, max_depth=2)
    model.fit(X, y)

    from serving import app as app_module

    monkeypatch.setattr(app_module, "load_model", lambda artifact_uri, project_id: model)
    monkeypatch.setattr(
        app_module,
        "download_json",
        lambda uri, project_id: {"feature_names": ["avg_transaction_30d"], "threshold": 0.5},
    )

    with TestClient(app_module.app) as c:
        yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_one_prediction_per_instance(client):
    resp = client.post(
        "/predict",
        json={"instances": [{"avg_transaction_30d": 55.0}, {"avg_transaction_30d": 8.0}]},
    )
    assert resp.status_code == 200
    predictions = resp.json()["predictions"]
    assert len(predictions) == 2
    for pred in predictions:
        assert 0.0 <= pred[CHURN_PROBABILITY_FIELD] <= 1.0
        assert isinstance(pred[CHURN_PREDICTION_FIELD], bool)


def test_positional_array_instances_are_rejected(client):
    # The production defect: with a BigQuery source and no instanceConfig, Vertex sends each
    # row as a positional array. Those match no feature name, every feature reindexes to NaN,
    # and XGBoost returns its all-missing constant for every row — with the job reporting
    # complete success. This must fail loudly instead.
    resp = client.post("/predict", json={"instances": [[55.0], [8.0]]})
    assert resp.status_code == 400
    assert "instanceType" in resp.json()["detail"]


def test_instances_with_unrecognised_keys_are_rejected(client):
    resp = client.post("/predict", json={"instances": [{"some_other_column": 1.0}]})
    assert resp.status_code == 400


def test_named_instances_produce_varied_predictions(client):
    # The counterpart assertion: correctly-named instances must not collapse to one value.
    resp = client.post(
        "/predict",
        json={"instances": [{"avg_transaction_30d": 180.0}, {"avg_transaction_30d": 5.0}]},
    )
    assert resp.status_code == 200
    probs = [p[CHURN_PROBABILITY_FIELD] for p in resp.json()["predictions"]]
    assert probs[0] != probs[1]


def test_null_heavy_but_correctly_named_instances_are_accepted(client):
    # A legitimately null row is valid input and must still be scored — the guard keys on
    # column names, not on all-NaN values, precisely so this case is not rejected.
    resp = client.post("/predict", json={"instances": [{"avg_transaction_30d": None}]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 1


def test_empty_instance_list_is_not_rejected(client):
    resp = client.post("/predict", json={"instances": []})
    assert resp.status_code == 200
    assert resp.json()["predictions"] == []


def test_malformed_body_is_a_client_error_not_a_crash(client):
    # PredictRequest makes the shape part of the endpoint's contract: a body without
    # "instances" used to raise KeyError and surface as a 500, which reads as a broken
    # serving container rather than as a malformed request.
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_response_matches_the_declared_prediction_contract(client):
    # These are the exact field names post_training.evaluate reads back out of the Batch
    # Prediction output and the orchestrator MERGEs into ml.predictions.
    resp = client.post("/predict", json={"instances": [{"avg_transaction_30d": 180.0}]})
    assert set(resp.json()["predictions"][0]) == {
        CHURN_PROBABILITY_FIELD,
        CHURN_PREDICTION_FIELD,
    }
