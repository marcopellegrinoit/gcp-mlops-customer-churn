import pytest
from dashboard.settings import DashboardSettings
from pydantic import ValidationError


def test_project_id_is_required(monkeypatch):
    """A missing project must fail at startup, not render an empty dashboard."""
    # conftest sets BQ_PROJECT_ID process-wide so app.py is importable at collection time;
    # both names have to go for this to be testing what it claims to.
    monkeypatch.delenv("BQ_PROJECT_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)

    with pytest.raises(ValidationError):
        DashboardSettings(_env_file=None)


def test_accepts_either_project_env_var_name():
    assert DashboardSettings(BQ_PROJECT_ID="from-terraform").project_id == "from-terraform"
    assert DashboardSettings(PROJECT_ID="from-docker-run").project_id == "from-docker-run"


def test_rejects_bands_that_leave_medium_empty():
    with pytest.raises(ValidationError, match="medium_risk_threshold must be below"):
        DashboardSettings(BQ_PROJECT_ID="test", high_risk_threshold=0.4, medium_risk_threshold=0.6)
    with pytest.raises(ValidationError, match="medium_risk_threshold must be below"):
        DashboardSettings(BQ_PROJECT_ID="test", high_risk_threshold=0.5, medium_risk_threshold=0.5)


def test_indicator_ratio_must_exceed_one():
    # A ratio of 1.0 makes every customer at or above the cohort median an "indicator",
    # which is half the base and ranks nothing.
    with pytest.raises(ValidationError):
        DashboardSettings(BQ_PROJECT_ID="test", indicator_ratio=1.0)


def test_defaults_keep_the_dashboard_inside_the_free_tier():
    settings = DashboardSettings(BQ_PROJECT_ID="test")
    # The byte ceiling is the backstop that turns a runaway view rewrite into an error
    # instead of a bill, and the TTL is what keeps interactions off BigQuery entirely.
    assert settings.max_bytes_billed == 2 * 1024**3
    assert settings.cache_ttl_seconds >= 600
