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
from post_training.storage import download_json, list_blobs, upload_text
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

_PROJECT = "test-project"
_BUCKET = "test-bucket"


@pytest.fixture(scope="module")
def gcs_client():
    """Spin up a fake-gcs-server container and return a client pointed at it.

    Shared across the module to avoid paying Docker startup cost per test. Also sets
    GOOGLE_CLOUD_PROJECT since storage.py's functions call storage.Client() with no
    project argument, unlike serving's — the client needs a project to resolve against
    even when talking to the emulator.
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
        os.environ["GOOGLE_CLOUD_PROJECT"] = _PROJECT
        try:
            client = storage.Client(project=_PROJECT)
            client.create_bucket(_BUCKET)
            yield client
        finally:
            del os.environ["STORAGE_EMULATOR_HOST"]
            del os.environ["GOOGLE_CLOUD_PROJECT"]


@pytest.mark.integration
class TestStorageAgainstFakeGcs:
    def test_download_json_returns_parsed_object(self, gcs_client):
        payload = {"model_version": "champion-v3", "recall": 0.83}
        gcs_client.bucket(_BUCKET).blob("champion/metrics.json").upload_from_string(
            json.dumps(payload), content_type="application/json"
        )

        result = download_json(f"gs://{_BUCKET}/champion/metrics.json")

        assert result == payload

    def test_upload_text_writes_readable_blob(self, gcs_client):
        upload_text(f"gs://{_BUCKET}/reports/summary.txt", "pr_auc_delta=0.03")

        written = gcs_client.bucket(_BUCKET).blob("reports/summary.txt").download_as_text()
        assert written == "pr_auc_delta=0.03"

    def test_list_blobs_returns_every_blob_under_prefix(self, gcs_client):
        gcs_client.bucket(_BUCKET).blob("run-42/part-0.jsonl").upload_from_string("{}")
        gcs_client.bucket(_BUCKET).blob("run-42/part-1.jsonl").upload_from_string("{}")
        gcs_client.bucket(_BUCKET).blob("run-99/part-0.jsonl").upload_from_string("{}")

        names = {blob.name for blob in list_blobs(f"gs://{_BUCKET}/run-42/")}

        assert names == {"run-42/part-0.jsonl", "run-42/part-1.jsonl"}
