"""Unit tests for the fetch_champion stage (no GCP credentials required)."""

import post_training.fetch_champion as fetch_champion_module
from post_training.fetch_champion import run_fetch_champion_stage


class _FakeModel:
    """Stands in for aiplatform.Model, whose `name` is the bare model ID, unprefixed."""

    def __init__(self, name, labels, uri="gs://bucket/artifacts/abc"):
        self.name = name
        self.resource_name = f"projects/p/locations/europe-west1/models/{name}"
        self.labels = labels
        self.uri = uri


def test_run_fetch_champion_stage_returns_empty_when_no_champion(monkeypatch):
    monkeypatch.setattr(fetch_champion_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        fetch_champion_module.aiplatform.Model,
        "list",
        staticmethod(lambda **kwargs: []),
    )
    model_id, shap, threshold = run_fetch_champion_stage("proj", "europe-west1", "churn-predictor")
    assert model_id == ""
    assert shap is None
    assert threshold is None


def test_run_fetch_champion_stage_returns_champion_model_id_shap_and_threshold(monkeypatch):
    champion = _FakeModel(name="123", labels={"role": "champion"})
    rejected = _FakeModel(name="999", labels={"role": "rejected"})

    monkeypatch.setattr(fetch_champion_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        fetch_champion_module.aiplatform.Model,
        "list",
        staticmethod(lambda **kwargs: [rejected, champion]),
    )
    monkeypatch.setattr(
        fetch_champion_module,
        "download_json",
        lambda uri: {"shap_importance": {"feature_a": 0.5}, "threshold": 0.42},
    )
    model_id, shap, threshold = run_fetch_champion_stage("proj", "europe-west1", "churn-predictor")
    assert model_id == "123"
    assert shap == {"feature_a": 0.5}
    assert threshold == 0.42


def test_run_fetch_champion_stage_returns_none_shap_and_threshold_when_missing(monkeypatch):
    champion = _FakeModel(name="123", labels={"role": "champion"})
    monkeypatch.setattr(fetch_champion_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        fetch_champion_module.aiplatform.Model,
        "list",
        staticmethod(lambda **kwargs: [champion]),
    )
    monkeypatch.setattr(fetch_champion_module, "download_json", lambda uri: {})
    _, shap, threshold = run_fetch_champion_stage("proj", "europe-west1", "churn-predictor")
    assert shap is None
    assert threshold is None
