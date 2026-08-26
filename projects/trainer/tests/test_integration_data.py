"""Integration tests for data.py's GCS I/O against a fake GCS server via Testcontainers.

export_snapshot()/read_split() (BigQuery, against ml.split_assignments) are NOT covered here:
the local BigQuery emulator (ghcr.io/goccy/bigquery-emulator, verified through 0.8.1) crashes
with an internal WASM panic on any table operation in a dataset literally named "ml" — which is
exactly what SPLIT_ASSIGNMENTS_TABLE hardcodes — regardless of host architecture or how the
dataset is created (tables.insert REST call or a CREATE TABLE DDL query both trip it). That
BigQuery-touching half of this module has no viable local-emulator test target; it remains
covered only by production usage, not by an automated integration test.

Run the full suite (requires Docker):
    pytest -m integration

Skip in CI environments without Docker:
    pytest -m "not integration"
"""

import os

import pytest
from google.cloud import storage
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from trainer.data import download_json, upload_json

_PROJECT = "test-project"
_BUCKET = "test-bucket"


@pytest.fixture(scope="module")
def gcs_client():
    """Spin up a fake-gcs-server container and return a client pointed at it."""
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
class TestJsonRoundTripAgainstFakeGcs:
    def test_upload_json_then_download_json_round_trips(self, gcs_client):
        payload = {"snapshot_date": "2026-08-01", "split": "train"}

        uri = upload_json(payload, _PROJECT, _BUCKET, "refs/train.json")
        result = download_json(uri)

        assert result == payload
        assert uri == f"gs://{_BUCKET}/refs/train.json"
