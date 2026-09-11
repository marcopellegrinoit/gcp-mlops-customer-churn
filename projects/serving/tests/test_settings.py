"""Unit tests for the serving container's environment contract."""

import pytest
from pydantic import ValidationError
from serving.settings import Settings

_VERTEX_ENV = {
    "aip_storage_uri": "gs://bucket/artifacts/abc123",
    "aip_health_route": "/health",
    "aip_predict_route": "/predict",
    "threshold": 0.42,
    "project_id": "proj",
}


def test_reads_the_vertex_supplied_environment():
    settings = Settings(**_VERTEX_ENV)
    assert settings.aip_http_port == 8080
    assert settings.threshold == 0.42


def test_threshold_outside_the_probability_range_is_rejected():
    # A threshold above 1 labels every row False and below 0 labels every row True, and
    # either way the batch completes successfully with uniform, meaningless labels.
    with pytest.raises(ValidationError):
        Settings(**(_VERTEX_ENV | {"threshold": 1.5}))


def test_relative_route_is_rejected():
    with pytest.raises(ValidationError, match="must start with"):
        Settings(**(_VERTEX_ENV | {"aip_predict_route": "predict"}))


def test_missing_threshold_fails_at_startup():
    # Set by register_or_reject on the registered model's container spec. Absent means the
    # container was started outside that contract; guessing an operating point would score a
    # whole batch against a threshold no model was ever evaluated at.
    with pytest.raises(ValidationError, match="threshold"):
        Settings(**{k: v for k, v in _VERTEX_ENV.items() if k != "threshold"})
