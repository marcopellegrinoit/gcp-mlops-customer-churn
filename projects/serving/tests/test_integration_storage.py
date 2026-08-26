"""Integration tests for storage.py against a fake GCS server via Testcontainers.

Run the full suite (requires Docker):
    pytest -m integration

Skip in CI environments without Docker:
    pytest -m "not integration"
"""

import json
import os

import pytest
from google.cloud import storage
from serving.storage import download_json, download_text
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

_PROJECT = "test-project"
_BUCKET = "test-bucket"


@pytest.fixture(scope="module")
def gcs_client():
    """Spin up a fake-gcs-server container and return a client pointed at it.

    Shared across the module to avoid paying Docker startup cost per test.
    """
    container = DockerContainer("fsouza/fake-gcs-server:1.52.2").with_command(
        "-scheme http -public-host localhost"
    )
    container.with_exposed_ports(4443)
    with container:
        wait_for_logs(container, "server started at", timeout=60)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(4443)

        os.environ["STORAGE_EMULATOR_HOST"] = f"http://{host}:{port}"
        try:
            client = storage.Client(project=_PROJECT)
            client.create_bucket(_BUCKET)
            yield client
        finally:
            del os.environ["STORAGE_EMULATOR_HOST"]


@pytest.mark.integration
class TestDownloadAgainstFakeGcs:
    def test_download_json_returns_parsed_object(self, gcs_client):
        payload = {"customer_id": "c1", "churn_probability": 0.42}
        gcs_client.bucket(_BUCKET).blob("predictions/c1.json").upload_from_string(
            json.dumps(payload), content_type="application/json"
        )

        result = download_json(f"gs://{_BUCKET}/predictions/c1.json", _PROJECT)

        assert result == payload

    def test_download_text_returns_raw_contents(self, gcs_client):
        gcs_client.bucket(_BUCKET).blob("threshold.txt").upload_from_string("0.35")

        result = download_text(f"gs://{_BUCKET}/threshold.txt", _PROJECT)

        assert result == "0.35"
