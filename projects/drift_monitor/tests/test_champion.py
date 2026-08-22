"""Unit tests for champion lookup (no GCP credentials required)."""

import drift_monitor.champion as champion_module
from drift_monitor.champion import fetch_champion


class _FakeModel:
    def __init__(self, resource_name, labels):
        self.resource_name = resource_name
        self.labels = labels


def test_fetch_champion_returns_none_when_no_models(monkeypatch):
    monkeypatch.setattr(champion_module.aiplatform.Model, "list", staticmethod(lambda **kwargs: []))
    assert fetch_champion("churn-predictor") is None


def test_fetch_champion_skips_non_champion_roles(monkeypatch):
    rejected = _FakeModel("projects/p/locations/l/models/999", {"role": "rejected"})
    champion = _FakeModel("projects/p/locations/l/models/123", {"role": "champion"})
    monkeypatch.setattr(
        champion_module.aiplatform.Model,
        "list",
        staticmethod(lambda **kwargs: [rejected, champion]),
    )
    result = fetch_champion("churn-predictor")
    assert result.resource_name == "projects/p/locations/l/models/123"
