"""Unit tests for the container entrypoint Vertex AI actually executes.

`python -m serving.serve` is the ENTRYPOINT, and nothing else imports it, so a stale
reference here survives every other test and only surfaces as a model server that exits
1 a second after Vertex boots it — roughly 40 minutes into a retraining pipeline.
"""

import runpy
import sys
from unittest.mock import patch

import pytest
from serving.settings import get_settings

_VERTEX_ENV = {
    "AIP_STORAGE_URI": "gs://bucket/artifacts/abc123",
    "AIP_HEALTH_ROUTE": "/health",
    "AIP_PREDICT_ROUTE": "/predict",
    "THRESHOLD": "0.42",
    "PROJECT_ID": "proj",
}


@pytest.fixture
def vertex_env(monkeypatch):
    """Supply the container contract's env vars and reset the settings cache around it."""
    for name, value in _VERTEX_ENV.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_entrypoint_module_imports():
    # Importing is the whole point: every name serve.py pulls in has to still exist.
    import serving.serve  # noqa: F401


def test_entrypoint_serves_the_app_on_the_vertex_supplied_port(vertex_env):
    # runpy re-executes the module body; drop the cached import so it runs as __main__.
    sys.modules.pop("serving.serve", None)
    with patch("uvicorn.run") as run:
        runpy.run_module("serving.serve", run_name="__main__")
    run.assert_called_once_with("serving.app:app", host="0.0.0.0", port=8080)
