"""Unit tests for the fetch_champion stage (no GCP credentials required)."""

import post_training.fetch_champion as fetch_champion_module
import pytest
from post_training.fetch_champion import run_fetch_champion_stage
from pydantic import ValidationError


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
        lambda uri: {
            "feature_names": ["avg_transaction_30d"],
            "shap_importance": {"feature_a": 0.5},
            "threshold": 0.42,
        },
    )
    model_id, shap, threshold = run_fetch_champion_stage("proj", "europe-west1", "churn-predictor")
    assert model_id == "123"
    assert shap == {"feature_a": 0.5}
    assert threshold == 0.42


def test_champion_with_unreadable_metadata_fails_here(monkeypatch):
    # This stage used to hand back None for a metadata.json missing its threshold, and the
    # pipeline carried the gap two stages further before evaluate crashed converting "" to a
    # float. A registered champion always has a threshold — register_or_reject bakes it into
    # the serving container — so an artifact without one is a broken champion, and the
    # useful place to say so is the stage that read it.
    champion = _FakeModel(name="123", labels={"role": "champion"})
    monkeypatch.setattr(fetch_champion_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        fetch_champion_module.aiplatform.Model,
        "list",
        staticmethod(lambda **kwargs: [champion]),
    )
    monkeypatch.setattr(fetch_champion_module, "download_json", lambda uri: {})

    with pytest.raises(ValidationError, match="threshold"):
        run_fetch_champion_stage("proj", "europe-west1", "churn-predictor")
