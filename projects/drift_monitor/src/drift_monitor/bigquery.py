"""BigQuery I/O for the drift monitor."""

import pandas as pd
from google.cloud import bigquery


def fetch_latest_snapshot(project_id: str, bq_features_table: str) -> tuple[str, pd.DataFrame]:
    """Return (snapshot_date, df) for the most recent partition of the features table."""
    bq = bigquery.Client(project=project_id)
    full_table = f"{project_id}.{bq_features_table}"

    row = next(bq.query(f"SELECT MAX(snapshot_date) AS d FROM `{full_table}`").result())
    snapshot_date = str(row["d"])

    df = bq.query(
        f"""
        SELECT * EXCEPT (feature_computed_at, latest_event_ts)
        FROM `{full_table}`
        WHERE snapshot_date = '{snapshot_date}'
        """
    ).to_dataframe()

    return snapshot_date, df


def fetch_quality_context(
    project_id: str, bq_features_table: str, snapshot_date: str, history_partitions: int
) -> tuple[list[int], int]:
    """Return (recent row counts before this snapshot, duplicate customer_id count in it).

    Both are facts about the feature table itself rather than about any model, so they come
    from BigQuery rather than from the frozen baseline — the row-count history is what makes
    "this load is short" answerable, and it has to exclude the snapshot being judged.
    """
    bq = bigquery.Client(project=project_id)
    full_table = f"{project_id}.{bq_features_table}"
    params = [
        bigquery.ScalarQueryParameter("snapshot_date", "DATE", snapshot_date),
        bigquery.ScalarQueryParameter("history_partitions", "INT64", history_partitions),
    ]

    history = bq.query(
        f"""
        SELECT COUNT(*) AS row_count
        FROM `{full_table}`
        WHERE snapshot_date < @snapshot_date
        GROUP BY snapshot_date
        ORDER BY snapshot_date DESC
        LIMIT @history_partitions
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    ).result()
    recent_row_counts = [int(row["row_count"]) for row in history]

    duplicates = next(
        bq.query(
            f"""
            SELECT COUNT(*) AS duplicate_customers
            FROM (
                SELECT customer_id
                FROM `{full_table}`
                WHERE snapshot_date = @snapshot_date
                GROUP BY customer_id
                HAVING COUNT(*) > 1
            )
            """,
            job_config=bigquery.QueryJobConfig(query_parameters=params[:1]),
        ).result()
    )

    return recent_row_counts, int(duplicates["duplicate_customers"])
