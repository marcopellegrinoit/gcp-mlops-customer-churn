"""Unit tests for run() with a mocked BigQuery client."""

from unittest.mock import MagicMock

import pytest
from data_generator.main import Config, run


@pytest.fixture
def config():
    """Minimal config for fast unit tests."""
    return Config(
        project_id="test-project",
        dataset_id="test-dataset",
        table_id="test-table",
        batch_size=10,
        anomaly_rate=0.0,
        customer_pool_size=50,
    )


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.insert_rows_json.return_value = []  # no errors
    return client


# ---------------------------------------------------------------------------
# BigQuery interaction
# ---------------------------------------------------------------------------


class TestBigQueryInteraction:
    def test_insert_rows_json_called_once(self, config, mock_client):
        run(config, mock_client)
        mock_client.insert_rows_json.assert_called_once()

    def test_insert_uses_correct_table_ref(self, config, mock_client):
        run(config, mock_client)
        table_ref, _ = mock_client.insert_rows_json.call_args[0]
        assert table_ref == config.table_ref

    def test_insert_passes_list_of_dicts(self, config, mock_client):
        run(config, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_raises_on_bq_errors(self, config, mock_client):
        mock_client.insert_rows_json.return_value = [{"errors": [{"reason": "invalid"}]}]
        with pytest.raises(RuntimeError, match="BigQuery streaming insert errors"):
            run(config, mock_client)

    def test_no_exception_on_empty_error_list(self, config, mock_client):
        mock_client.insert_rows_json.return_value = []
        run(config, mock_client)  # must not raise


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


class TestBatchSize:
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50])
    def test_inserts_exactly_batch_size_rows(self, mock_client, batch_size):
        cfg = Config(
            project_id="p",
            dataset_id="d",
            table_id="t",
            batch_size=batch_size,
            customer_pool_size=50,
        )
        run(cfg, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        assert len(rows) == batch_size


# ---------------------------------------------------------------------------
# Anomaly rate
# ---------------------------------------------------------------------------


class TestAnomalyRate:
    def test_anomaly_rate_zero_produces_no_anomalies(self, mock_client):
        cfg = Config(
            project_id="p",
            dataset_id="d",
            table_id="t",
            batch_size=100,
            anomaly_rate=0.0,
            customer_pool_size=50,
        )
        run(cfg, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        assert not any(r["anomaly_injected"] for r in rows)

    def test_anomaly_rate_one_produces_all_anomalies(self, mock_client):
        cfg = Config(
            project_id="p",
            dataset_id="d",
            table_id="t",
            batch_size=50,
            anomaly_rate=1.0,
            customer_pool_size=50,
        )
        run(cfg, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        assert all(r["anomaly_injected"] for r in rows)

    def test_partial_anomaly_rate_produces_mixed_events(self, mock_client):
        cfg = Config(
            project_id="p",
            dataset_id="d",
            table_id="t",
            batch_size=500,
            anomaly_rate=0.5,
            customer_pool_size=50,
        )
        run(cfg, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        anomaly_count = sum(r["anomaly_injected"] for r in rows)
        # With 500 events at 50% rate, 0 or 500 is astronomically unlikely
        assert 0 < anomaly_count < 500


# ---------------------------------------------------------------------------
# Event structure from run()
# ---------------------------------------------------------------------------


class TestEventStructure:
    def test_every_row_has_event_id(self, config, mock_client):
        run(config, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        assert all("event_id" in r for r in rows)

    def test_every_row_has_raw_payload(self, config, mock_client):
        run(config, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        assert all("raw_payload" in r for r in rows)

    def test_event_ids_are_unique(self, config, mock_client):
        run(config, mock_client)
        _, rows = mock_client.insert_rows_json.call_args[0]
        ids = [r["event_id"] for r in rows]
        assert len(ids) == len(set(ids))
