import sys
from unittest.mock import patch

import pytest
from dashboard import serve
from pydantic import ValidationError


def _argv(monkeypatch, env: dict[str, str]) -> list[str]:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    captured: list[list[str]] = []
    serve.get_settings.cache_clear()

    with (
        patch("streamlit.web.cli.main", side_effect=lambda: captured.append(list(sys.argv))),
        pytest.raises(SystemExit),
    ):
        serve.main()
    return captured[0]


def test_binds_the_port_cloud_run_assigns(monkeypatch):
    argv = _argv(monkeypatch, {"PORT": "9999"})
    assert argv[argv.index("--server.port") + 1] == "9999"
    assert argv[argv.index("--server.address") + 1] == "0.0.0.0"


def test_defaults_to_cloud_runs_own_default_port(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    argv = _argv(monkeypatch, {})
    assert argv[argv.index("--server.port") + 1] == "8080"


def test_disables_the_checks_that_break_the_websocket_behind_iap(monkeypatch):
    """XSRF and CORS must be off, or the page loads and then hangs with no visible error.

    Cloud Run proxies to the container over plain HTTP behind TLS termination, and IAP sits
    in front of that, so the origin Streamlit checks against never matches. This is safe
    only because no request can reach the container without roles/run.invoker.
    """
    argv = _argv(monkeypatch, {})
    assert argv[argv.index("--server.enableXsrfProtection") + 1] == "false"
    assert argv[argv.index("--server.enableCORS") + 1] == "false"
    assert argv[argv.index("--server.headless") + 1] == "true"


def test_runs_the_app_module_without_importing_it(monkeypatch):
    """The resolved script must be app.py, and resolving it must not execute the page."""
    monkeypatch.delitem(sys.modules, "dashboard.app", raising=False)
    argv = _argv(monkeypatch, {})

    assert argv[:2] == ["streamlit", "run"]
    assert argv[2].endswith("dashboard/app.py")
    # Importing app.py runs main(), which queries BigQuery. Startup must not do that.
    assert "dashboard.app" not in sys.modules


def test_misconfiguration_fails_before_the_server_binds(monkeypatch):
    """A container missing its config must die, not start and serve error pages.

    Streamlit does not execute app.py until the first request, and /_stcore/health is served
    by the server itself — so without an explicit check here a revision missing BQ_PROJECT_ID
    starts, passes Cloud Run's startup probe, takes traffic, and only then breaks for every
    user. Verified against the built image: before this check the container stayed up and
    answered the probe with 200.
    """
    monkeypatch.delenv("BQ_PROJECT_ID", raising=False)
    monkeypatch.delenv("PROJECT_ID", raising=False)
    serve.get_settings.cache_clear()

    started = False

    def _fail_if_reached():
        nonlocal started
        started = True

    with (
        patch("streamlit.web.cli.main", side_effect=_fail_if_reached),
        pytest.raises(ValidationError),
    ):
        serve.main()

    assert not started, "the server must not bind when configuration is invalid"


def test_settings_cache_is_not_left_poisoned(monkeypatch):
    serve.get_settings.cache_clear()
    monkeypatch.setenv("BQ_PROJECT_ID", "valid-project")
    assert serve.get_settings().project_id == "valid-project"
