"""Unit tests for the Vertex AI custom-container prediction contract (FastAPI app)."""

import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from ml_common.config import CHURN_PREDICTION_FIELD, CHURN_PROBABILITY_FIELD


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AIP_STORAGE_URI", "gs://bucket/artifacts/abc123")
    monkeypatch.setenv("THRESHOLD", "0.5")
    monkeypatch.setenv("AIP_HEALTH_ROUTE", "/health")
    monkeypatch.setenv("AIP_PREDICT_ROUTE", "/predict")
    monkeypatch.setenv("PROJECT_ID", "test-project")

    X = pd.DataFrame({"avg_transaction_30d": [10.0, 50.0, 5.0, 60.0]})
    y = pd.Series([0, 1, 0, 1])
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)

    from serving import app as app_module

    monkeypatch.setattr(app_module, "load_model", lambda artifact_uri, project_id: model)
    monkeypatch.setattr(
        app_module,
        "download_json",
        lambda uri, project_id: {"feature_names": ["avg_transaction_30d"]},
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
