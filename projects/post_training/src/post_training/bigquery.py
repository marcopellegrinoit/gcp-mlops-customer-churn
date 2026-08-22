"""BigQuery read helpers for the champion/challenger evaluation stage."""

import json

import pandas as pd
from google.cloud import bigquery
from ml_common.config import SPLIT_ASSIGNMENTS_TABLE


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
        .to_dataframe(create_bqstorage_client=True, date_dtype=None, time_dtype=None)
    )
