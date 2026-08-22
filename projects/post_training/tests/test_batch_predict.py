"""Unit tests for the batch-prediction source export (no GCP credentials required)."""

import json

import post_training.batch_predict as batch_predict_module
from post_training.batch_predict import create_batch_source_files


class _FakeQueryJob:
    def result(self):
        return None


class _FakeExtractJob:
    def result(self):
        return None


class _FakeClient:
    def __init__(self):
        self.queries = []
        self.extracts = []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        return _FakeQueryJob()

    def extract_table(self, source, destination_uris, job_config=None):
        self.extracts.append((source, destination_uris, job_config))
        return _FakeExtractJob()


_TEST_REF = json.dumps({"snapshot_date": "2026-07-19", "split": "test"})


def test_create_batch_source_files_returns_gcs_wildcard_uri(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(batch_predict_module.bigquery, "Client", lambda project: fake_client)

    gcs_uri = create_batch_source_files(_TEST_REF, "my-project", "gs://bucket/batch-test")

    assert gcs_uri == "gs://bucket/batch-test/test-*.jsonl"
    assert len(fake_client.queries) == 1
    sql, job_config = fake_client.queries[0]
    assert "CREATE TABLE" in sql
    assert "my-project.scratch.test_batch_" in sql
    assert "my-project.ml.split_assignments" in sql
    params = {p.name: str(p.value) for p in job_config.query_parameters}
    assert params == {"snapshot_date": "2026-07-19", "split": "test"}

    assert len(fake_client.extracts) == 1
    source, destination_uris, extract_job_config = fake_client.extracts[0]
    assert source.startswith("my-project.scratch.test_batch_")
    assert destination_uris == gcs_uri
    assert extract_job_config.destination_format == "NEWLINE_DELIMITED_JSON"


def test_create_batch_source_files_strips_trailing_slash_from_prefix(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(batch_predict_module.bigquery, "Client", lambda project: fake_client)

    gcs_uri = create_batch_source_files(_TEST_REF, "my-project", "gs://bucket/batch-test/")

    assert gcs_uri == "gs://bucket/batch-test/test-*.jsonl"


def test_create_batch_source_files_generates_unique_table_names(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(batch_predict_module.bigquery, "Client", lambda project: fake_client)

    create_batch_source_files(_TEST_REF, "my-project", "gs://bucket/batch-test")
    create_batch_source_files(_TEST_REF, "my-project", "gs://bucket/batch-test")

    source1, _, _ = fake_client.extracts[0]
    source2, _, _ = fake_client.extracts[1]
    assert source1 != source2
