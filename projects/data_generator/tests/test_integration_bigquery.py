"""Integration tests for run() against a live BigQuery emulator via Testcontainers.

Run the full suite (requires Docker):
    pytest -m integration

Skip in CI environments without Docker:
    pytest -m "not integration"
"""

import os

import pytest
from data_generator.main import Config, run
from google.cloud import bigquery
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

_PROJECT = "test-project"
_DATASET = "test-dataset"
_TABLE = "test-table"

# Must match the fields produced by _generate_event().
_SCHEMA = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
    bigquery.SchemaField("customer_id", "STRING"),
    bigquery.SchemaField("event_type", "STRING"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("membership_tier", "STRING"),
    bigquery.SchemaField("monthly_transaction", "FLOAT64"),
    bigquery.SchemaField("payment_status", "STRING"),
    bigquery.SchemaField("payment_attempts_last_30d", "INT64"),
    bigquery.SchemaField("engagement_score", "FLOAT64"),
    bigquery.SchemaField("member_since_days", "INT64"),
    bigquery.SchemaField("contact_requests_last_30d", "INT64"),
    bigquery.SchemaField("churned", "BOOL"),
    bigquery.SchemaField("channel", "STRING"),
    bigquery.SchemaField("raw_payload", "STRING"),
    bigquery.SchemaField("anomaly_injected", "BOOL"),
]


def _row_count(client: bigquery.Client) -> int:
    """Return current row count for the test table."""
    result = client.query(f"SELECT COUNT(*) AS cnt FROM `{_PROJECT}.{_DATASET}.{_TABLE}`").result()
    return next(iter(result))["cnt"]


@pytest.fixture(scope="module")
def bq_client():
    """Spin up a BigQuery emulator container and return a configured client.

    The container is shared across the entire module to avoid paying Docker
    startup cost per test.  BIGQUERY_EMULATOR_HOST is set for the duration
    and cleaned up in the finally block.
    """
    container = (
        DockerContainer("ghcr.io/goccy/bigquery-emulator:0.8.1")
        .with_command(f"--project={_PROJECT} --dataset={_DATASET}")
        .with_exposed_ports(9050)
    )
    with container:
        wait_for_logs(container, "REST server listening", timeout=60)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(9050)

        os.environ["BIGQUERY_EMULATOR_HOST"] = f"http://{host}:{port}"
        try:
            client = bigquery.Client(project=_PROJECT)
            table_ref = bigquery.Table(f"{_PROJECT}.{_DATASET}.{_TABLE}", schema=_SCHEMA)
            client.create_table(table_ref, exists_ok=True)
            yield client
        finally:
            del os.environ["BIGQUERY_EMULATOR_HOST"]


@pytest.mark.integration
class TestRunAgainstEmulator:
    def test_run_completes_without_error(self, bq_client):
        """run() must not raise when the emulator accepts the streaming insert."""
        cfg = Config(
            project_id=_PROJECT,
            dataset_id=_DATASET,
            table_id=_TABLE,
            batch_size=10,
            anomaly_rate=0.0,
            customer_pool_size=50,
        )
        run(cfg, bq_client)  # must not raise

    def test_run_inserts_correct_row_count(self, bq_client):
        """Row count must increase by exactly batch_size."""
        cfg = Config(
            project_id=_PROJECT,
            dataset_id=_DATASET,
            table_id=_TABLE,
            batch_size=20,
            anomaly_rate=0.0,
            customer_pool_size=50,
        )
        before = _row_count(bq_client)
        run(cfg, bq_client)
        after = _row_count(bq_client)
        assert after - before == cfg.batch_size

    def test_anomaly_rows_are_stored(self, bq_client):
        """Rows inserted with anomaly_rate=1.0 must all have anomaly_injected=TRUE."""
        cfg = Config(
            project_id=_PROJECT,
            dataset_id=_DATASET,
            table_id=_TABLE,
            batch_size=15,
            anomaly_rate=1.0,
            customer_pool_size=50,
        )
        before = _row_count(bq_client)
        run(cfg, bq_client)

        result = bq_client.query(
            f"""
            SELECT COUNT(*) AS cnt
            FROM `{_PROJECT}.{_DATASET}.{_TABLE}`
            WHERE anomaly_injected = TRUE
            """
        ).result()
        anomaly_count = next(iter(result))["cnt"]
        # At minimum the batch we just inserted must all be anomalies
        after = _row_count(bq_client)
        inserted = after - before
        assert anomaly_count >= inserted

    def test_normal_rows_have_valid_engagement_score(self, bq_client):
        """Normal events must land with engagement_score in [0, 1]."""
        cfg = Config(
            project_id=_PROJECT,
            dataset_id=_DATASET,
            table_id=_TABLE,
            batch_size=30,
            anomaly_rate=0.0,
            customer_pool_size=50,
        )
        run(cfg, bq_client)

        result = bq_client.query(
            f"""
            SELECT COUNT(*) AS out_of_range
            FROM `{_PROJECT}.{_DATASET}.{_TABLE}`
            WHERE anomaly_injected = FALSE
              AND (engagement_score < 0 OR engagement_score > 1)
            """
        ).result()
        out_of_range = next(iter(result))["out_of_range"]
        assert out_of_range == 0

    def test_event_ids_are_unique_in_table(self, bq_client):
        """No duplicate event_id values should exist across all inserted rows."""
        result = bq_client.query(
            f"""
            SELECT COUNT(*) AS dupes
            FROM (
                SELECT event_id, COUNT(*) AS cnt
                FROM `{_PROJECT}.{_DATASET}.{_TABLE}`
                GROUP BY event_id
                HAVING cnt > 1
            )
            """
        ).result()
        dupes = next(iter(result))["dupes"]
        assert dupes == 0
