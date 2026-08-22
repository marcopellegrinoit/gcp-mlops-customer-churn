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
