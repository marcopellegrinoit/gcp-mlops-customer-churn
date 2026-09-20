from unittest.mock import MagicMock

import pandas as pd
from dashboard.data import fetch_risk_snapshot, fetch_risk_trend
from dashboard.settings import DashboardSettings


def _client(frame: pd.DataFrame) -> MagicMock:
    client = MagicMock()
    job = client.query.return_value
    job.result.return_value.to_dataframe.return_value = frame
    job.total_bytes_billed = 1024
    return client


def test_every_query_carries_the_byte_ceiling():
    """maximum_bytes_billed is the guard that turns a runaway scan into an error, not a bill.

    It is the only defence that still works if a view is rewritten to scan full history, so
    a query path that forgets it is a silent cost regression — nothing fails, the dashboard
    just gets expensive.
    """
    settings = DashboardSettings(BQ_PROJECT_ID="proj", max_bytes_billed=123)

    for fetch in (fetch_risk_snapshot, fetch_risk_trend):
        client = _client(pd.DataFrame())
        fetch(client, settings)
        job_config = client.query.call_args.kwargs["job_config"]
        assert job_config.maximum_bytes_billed == 123
        assert job_config.use_query_cache is True


def test_queries_address_the_configured_tables():
    settings = DashboardSettings(
        BQ_PROJECT_ID="proj", risk_table="features.risk_v2", trend_table="features.trend_v2"
    )

    client = _client(pd.DataFrame())
    fetch_risk_snapshot(client, settings)
    assert "`proj.features.risk_v2`" in client.query.call_args.args[0]

    client = _client(pd.DataFrame())
    fetch_risk_trend(client, settings)
    assert "`proj.features.trend_v2`" in client.query.call_args.args[0]


def test_the_snapshot_query_is_never_truncated():
    """A LIMIT would silently drop customers off the end of a risk-ranked worklist."""
    client = _client(pd.DataFrame())
    fetch_risk_snapshot(client, DashboardSettings(BQ_PROJECT_ID="proj"))
    assert "LIMIT" not in client.query.call_args.args[0].upper()
