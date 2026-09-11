"""Materialize the test split into scratch BigQuery, then export it as GCS JSONL."""

import uuid

from google.cloud import bigquery
from ml_common.config import get_settings
from ml_common.contracts import SplitRef


def create_batch_source_files(test_uri: str, project_id: str, gcs_uri_prefix: str) -> str:
    """Export this run's test split as sharded JSONL under gcs_uri_prefix; return the wildcard URI.

    ModelBatchPredictOp requires instances_format and predictions_format to either both be
    "bigquery" or both be non-bigquery — mixing bigquery_source_input_uri with
    gcs_destination_output_uri_prefix (or vice versa) is rejected by the API. This pipeline keeps
    Batch Prediction's *output* on GCS (see docs/ml-infrastructure.md —
    bigquery_destination_output_uri creates a new, unmanaged, never-expiring dataset per job), so
    the source must be GCS JSONL too.
    BigQuery can only extract from a physical table, not an arbitrary filtered query, so this still
    materializes exactly this run's test rows into scratch.test_batch_<uuid> first — the scratch
    dataset's default_table_expiration_ms (iac/config/bigquery.yaml) cleans that up automatically —
    then extracts it to GCS. The random table suffix avoids name collisions between concurrent
    pipeline runs.
    """
    parsed = SplitRef.model_validate_json(test_uri)
    bq = bigquery.Client(project=project_id)
    full_split_table = f"{project_id}.{get_settings().split_assignments_table}"
    table_ref = f"{project_id}.scratch.test_batch_{uuid.uuid4().hex}"

    bq.query(
        f"""
        CREATE TABLE `{table_ref}` AS
        SELECT * EXCEPT (churned, split, assigned_at)
        FROM `{full_split_table}`
        WHERE snapshot_date = @snapshot_date AND split = @split
        """,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("snapshot_date", "DATE", parsed.snapshot_date),
                bigquery.ScalarQueryParameter("split", "STRING", parsed.split),
            ]
        ),
    ).result()

    destination_uri = f"{gcs_uri_prefix.rstrip('/')}/test-*.jsonl"
    bq.extract_table(
        table_ref,
        destination_uri,
        job_config=bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
        ),
    ).result()

    return destination_uri
