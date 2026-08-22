"""BigQuery and GCS I/O utilities for the training pipeline."""

import json

import pandas as pd
from google.cloud import bigquery, storage
from ml_common.config import SPLIT_ASSIGNMENTS_TABLE, TEST_SIZE


def export_snapshot(
    project_id: str,
    bq_features_table: str,
    snapshot_date: str | None = None,
) -> tuple[str, str]:
    """Freeze a stratified train/test split of the feature snapshot into ml.split_assignments.

    The split is computed once in BigQuery (PERCENT_RANK partitioned by churned, ordered by a
    deterministic hash of customer_id) and persisted as a permanent partition of
    ml.split_assignments — see docs/ml-infrastructure.md for why this replaced an in-memory
    pandas/sklearn split. If snapshot_date is None, the most recent available partition of
    bq_features_table is used. Returns (train_ref, test_ref), small JSON strings consumed by
    read_split().
    """
    bq = bigquery.Client(project=project_id)
    full_features_table = f"{project_id}.{bq_features_table}"
    full_split_table = f"{project_id}.{SPLIT_ASSIGNMENTS_TABLE}"

    if snapshot_date is None:
        row = next(
            bq.query(f"SELECT MAX(snapshot_date) AS d FROM `{full_features_table}`").result()
        )
        snapshot_date = str(row["d"])

    snapshot_date_param = bigquery.ScalarQueryParameter("snapshot_date", "DATE", snapshot_date)

    # DELETE+INSERT rather than MERGE: idempotent re-run for the same snapshot_date (e.g. a
    # retried or drift-triggered retrain) fully replaces that partition's assignment.
    bq.query(
        f"DELETE FROM `{full_split_table}` WHERE snapshot_date = @snapshot_date",
        job_config=bigquery.QueryJobConfig(query_parameters=[snapshot_date_param]),
    ).result()

    # PARTITION BY churned gives an exact stratified split (same class-balance guarantee as
    # sklearn's stratify=), computed once here rather than once per downstream reader.
    bq.query(
        f"""
        INSERT INTO `{full_split_table}`
        SELECT f.* EXCEPT (feature_computed_at, latest_event_ts),
          IF(PERCENT_RANK() OVER (
               PARTITION BY churned ORDER BY FARM_FINGERPRINT(CAST(customer_id AS STRING))
             ) < @train_fraction, 'train', 'test') AS split,
          CURRENT_TIMESTAMP() AS assigned_at
        FROM `{full_features_table}` f
        WHERE snapshot_date = @snapshot_date
        """,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                snapshot_date_param,
                bigquery.ScalarQueryParameter("train_fraction", "FLOAT64", 1 - TEST_SIZE),
            ]
        ),
    ).result()

    return (
        json.dumps({"snapshot_date": snapshot_date, "split": "train"}),
        json.dumps({"snapshot_date": snapshot_date, "split": "test"}),
    )


def read_split(ref: str, project_id: str) -> pd.DataFrame:
    """Load a frozen train/test split from ml.split_assignments given an export_snapshot() ref."""
    parsed = json.loads(ref)
    bq = bigquery.Client(project=project_id)
    full_split_table = f"{project_id}.{SPLIT_ASSIGNMENTS_TABLE}"

    return (
        bq.query(
            f"""
            SELECT * EXCEPT (split, assigned_at)
            FROM `{full_split_table}`
            WHERE snapshot_date = @snapshot_date AND split = @split
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("snapshot_date", "DATE", parsed["snapshot_date"]),
                    bigquery.ScalarQueryParameter("split", "STRING", parsed["split"]),
                ]
            ),
        )
        .result()
        .to_dataframe(
            create_bqstorage_client=True,
            # Force plain dtypes instead of db-dtypes extension types (dbdate/dbtime): downstream
            # consumers don't depend on db-dtypes and can't deserialize those extension types.
            date_dtype=None,
            time_dtype=None,
        )
    )


def upload_json(obj: dict, project_id: str, gcs_bucket: str, blob_name: str) -> str:
    """Write a dict as JSON to GCS and return the gs:// URI."""
    client = storage.Client(project=project_id)
    client.bucket(gcs_bucket).blob(blob_name).upload_from_string(
        json.dumps(obj), content_type="application/json"
    )
    return f"gs://{gcs_bucket}/{blob_name}"


def download_json(gcs_uri: str) -> dict:
    """Read a JSON file from GCS."""
    bucket_name, blob_name = _parse_uri(gcs_uri)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())


def _parse_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, blob_name)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob
