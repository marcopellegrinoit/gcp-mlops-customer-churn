"""Headless render tests for the page itself.

app.py is mostly glue, but it is the glue that names columns. A renamed mart column or a
mistyped key produces a KeyError that no unit test on risk.py or charts.py would ever see —
it surfaces as a stack trace in front of a business user. AppTest runs the real script
against a patched BigQuery layer, so those breakages fail here instead.
"""

from importlib.util import find_spec
from unittest.mock import patch

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

SCRIPT = find_spec("dashboard.app").origin


def _trend(days: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2026-09-06", periods=days, freq="D")
    return pd.DataFrame(
        {
            "snapshot_date": dates,
            "customers_scored": [200] * days,
            "predicted_churners": [40, 42, 45, 44, 47],
            "high_risk_customers": [30, 31, 35, 33, 36],
            "mean_churn_probability": [0.31, 0.32, 0.34, 0.33, 0.35],
            "predicted_churn_rate": [0.20, 0.21, 0.225, 0.22, 0.235],
            "model_versions_used": [1] * days,
            "last_predicted_at": dates,
        }
    )


def _run(snapshot: pd.DataFrame, trend: pd.DataFrame) -> AppTest:
    # Patched on dashboard.data, not dashboard.app: AppTest re-executes app.py as a fresh
    # module on every run, so its `from dashboard.data import ...` rebinds from the source
    # module each time and a patch applied to the already-imported dashboard.app is invisible
    # to the script under test.
    with (
        patch("dashboard.data.fetch_risk_snapshot", return_value=snapshot),
        patch("dashboard.data.fetch_risk_trend", return_value=trend),
        patch("dashboard.data.build_client"),
    ):
        harness = AppTest.from_file(SCRIPT, default_timeout=60)
        harness.run()
    return harness


@pytest.fixture(autouse=True)
def _clear_caches():
    """Drop cached snapshots between runs — otherwise each test sees the previous one's.

    Cleared globally rather than per-function: AppTest runs the script as its own module, so
    the cached functions it creates are not the ones reachable through `import dashboard.app`.
    """
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def test_page_renders_without_exception(snapshot):
    harness = _run(snapshot, _trend())
    assert not harness.exception


def test_headline_metrics_match_the_snapshot(snapshot):
    harness = _run(snapshot, _trend())
    labels = [metric.label for metric in harness.metric]

    assert "Customers scored" in labels
    scored = next(m for m in harness.metric if m.label == "Customers scored")
    assert scored.value == f"{len(snapshot):,}"


def test_a_stale_snapshot_is_called_out_not_shown_as_today(snapshot):
    """The failure mode this guards is a frozen pipeline looking like a quiet week."""
    stale = snapshot.assign(snapshot_date=pd.Timestamp("2026-01-01").date())
    harness = _run(stale, _trend())

    assert not harness.exception
    assert harness.warning, "an old snapshot must warn, not render silently as current"
    assert "days old" in harness.warning[0].value


def test_a_fresh_snapshot_does_not_warn():
    fresh_date = pd.Timestamp.now("UTC").date()
    fresh = pd.DataFrame(
        {
            "customer_id": ["cust-0001", "cust-0002"],
            "churn_probability": [0.91, 0.12],
            "churn_prediction": [True, False],
            "model_version": ["projects/p/locations/eu/models/churn-predictor"] * 2,
            "snapshot_date": [fresh_date, fresh_date],
            "membership_tier": ["champion", "supporter"],
            "region": ["north", "south"],
            "preferred_channel": ["email", "web"],
            "member_since_days": [900, 120],
            "monthly_value": [180.0, 12.0],
            "payment_failure_rate_30d": [0.8, 0.0],
            "days_since_last_successful_payment": [95.0, 3.0],
            "contact_requests_last_30d": [5, 0],
            "avg_engagement_30d": [8.0, 71.0],
            "events_last_30d": [2, 28],
            "campaign_participation_rate": [0.02, 0.6],
        }
    )
    harness = _run(fresh, _trend())

    assert not harness.exception
    assert not harness.warning


def test_no_scored_data_yet_explains_itself_rather_than_erroring():
    harness = _run(pd.DataFrame(), pd.DataFrame())

    assert not harness.exception
    assert harness.info, "an empty mart must explain itself, not render a blank page"
    assert "nightly pipeline" in harness.info[0].value


def test_the_page_never_shows_the_observed_churn_label(snapshot):
    """Predicted must not sit next to observed. The mart drops `churned`; this is the belt."""
    harness = _run(snapshot.assign(churned=True), _trend())

    columns = harness.dataframe[0].value.columns
    assert "churned" not in columns
