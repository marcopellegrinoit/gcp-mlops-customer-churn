"""BigQuery and GCS I/O utilities for the training pipeline."""

import json

import pandas as pd
from data_contracts import SplitRef
from google.cloud import bigquery, storage
from ml_common.config import get_settings


def export_snapshot(
    project_id: str,
    bq_features_table: str,
    snapshot_date: str | None = None,
) -> tuple[SplitRef, SplitRef]:
    """Freeze a stratified train/test split of the feature snapshot into ml.split_assignments.

    The split is computed once in BigQuery (PERCENT_RANK partitioned by churned, ordered by a
    deterministic hash of customer_id) and persisted as a permanent partition of
    ml.split_assignments — see docs/ml-infrastructure.md for why this replaced an in-memory
    pandas/sklearn split. If snapshot_date is None, the most recent available partition of
    bq_features_table is used. Returns (train_ref, test_ref), the pointers read back by
    read_split().
    """
    settings = get_settings()
    bq = bigquery.Client(project=project_id)
    full_features_table = f"{project_id}.{bq_features_table}"
    full_split_table = f"{project_id}.{settings.split_assignments_table}"

    if snapshot_date is None:
        row = next(
            bq.query(f"SELECT MAX(snapshot_date) AS d FROM `{full_features_table}`").result()
        )
        snapshot_date = str(row["d"])

    snapshot_date_param = bigquery.ScalarQueryParameter("snapshot_date", "DATE", snapshot_date)

    # WRITE_TRUNCATE against the partition decorator (table$YYYYMMDD) rather than DELETE
    # followed by INSERT. Both give the same replace-this-partition semantics on a re-run,
    # but this is a single atomic job: two retraining pipelines racing on the same
    # snapshot_date (drift can trigger one while an earlier one is still running) resolve to
    # last-writer-wins instead of interleaving a DELETE into the other's INSERT and leaving
    # the partition half-written. The lineage record this table exists to be is only
    # trustworthy if it can never be observed partially replaced.
    #
    # PARTITION BY churned gives an exact stratified split (same class-balance guarantee as
    # sklearn's stratify=), computed once here rather than once per downstream reader.
    bq.query(
        f"""
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
                bigquery.ScalarQueryParameter("train_fraction", "FLOAT64", 1 - settings.test_size),
            ],
            destination=f"{full_split_table}${snapshot_date.replace('-', '')}",
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    ).result()

    return (
        SplitRef(snapshot_date=snapshot_date, split="train"),
        SplitRef(snapshot_date=snapshot_date, split="test"),
    )


def read_split(ref: str, project_id: str) -> pd.DataFrame:
    """Load a frozen train/test split from ml.split_assignments given an export_snapshot() ref."""
    parsed = SplitRef.model_validate_json(ref)
    bq = bigquery.Client(project=project_id)
    full_split_table = f"{project_id}.{get_settings().split_assignments_table}"

    return (
        bq.query(
            f"""
            SELECT * EXCEPT (split, assigned_at)
            FROM `{full_split_table}`
            WHERE snapshot_date = @snapshot_date AND split = @split
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("snapshot_date", "DATE", parsed.snapshot_date),
                    bigquery.ScalarQueryParameter("split", "STRING", parsed.split),
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
