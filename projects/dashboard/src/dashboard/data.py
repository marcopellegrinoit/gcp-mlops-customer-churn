"""BigQuery access for the dashboard: one query per cache window, never one per interaction.

The whole cost model of this app is in this module. Both marts are views scoped to a single
snapshot (churn_risk_current) or to a narrow set of aggregate columns (churn_risk_daily), and
each is pulled into pandas *in full*, once, and cached for the configured TTL. Every filter,
sort, search and drill-down in the UI then runs against that in-memory frame. Pushing those
down to BigQuery instead would make an idle user dragging a slider a billable event, which is
how a free-tier dashboard stops being free.

Two independent guards sit under that: maximum_bytes_billed, so a query that would scan more
than the configured ceiling is rejected by BigQuery before it runs rather than billed; and
the views themselves, which do the partition scoping in SQL.

Both queries are `SELECT *`, so this is also where the dashboard finds out whether the marts
still provide what it reads. Each fetch checks its result against the contract in
dashboard.schema before handing it to the UI — see there for why the snapshot is checked by
column and the trend row by row.
"""

from __future__ import annotations

import logging

import pandas as pd
from google.cloud import bigquery

from dashboard.schema import ChurnRiskRow, require_columns, validate_trend
from dashboard.settings import DashboardSettings

logger = logging.getLogger(__name__)


def build_client(settings: DashboardSettings) -> bigquery.Client:
    """Return a BigQuery client bound to the dashboard's project and data location."""
    return bigquery.Client(project=settings.project_id, location=settings.bq_location)


def _run(client: bigquery.Client, settings: DashboardSettings, sql: str) -> pd.DataFrame:
    """Execute a query under the configured byte ceiling and return it as a DataFrame."""
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=settings.max_bytes_billed,
        # Repeat visits within BigQuery's own 24h cache window cost nothing at all; the
        # process cache above this normally means we never get here, but when a container
        # cold-starts mid-day this makes the first render free as well.
        use_query_cache=True,
    )
    job = client.query(sql, job_config=job_config)
    frame = job.result().to_dataframe()
    logger.info(
        "query complete: %s rows, %s bytes billed",
        len(frame),
        job.total_bytes_billed,
    )
    return frame


def fetch_risk_snapshot(client: bigquery.Client, settings: DashboardSettings) -> pd.DataFrame:
    """Return the newest scored snapshot, one row per customer.

    No LIMIT: the view is already scoped to a single day's partition, and a truncated
    worklist would silently hide the customers past the cut — in a table sorted by risk
    that is a guarantee the dashboard cannot make and a business user cannot detect.
    """
    return require_columns(
        _run(client, settings, f"SELECT * FROM `{settings.project_id}.{settings.risk_table}`"),
        ChurnRiskRow,
        settings.risk_table,
    )


def fetch_risk_trend(client: bigquery.Client, settings: DashboardSettings) -> pd.DataFrame:
    """Return the per-day scoring history behind the trend strip."""
    return validate_trend(
        _run(
            client,
            settings,
            f"SELECT * FROM `{settings.project_id}.{settings.trend_table}` ORDER BY snapshot_date",
        ),
        settings.trend_table,
    )
