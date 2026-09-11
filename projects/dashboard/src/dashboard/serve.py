"""Container entrypoint: starts the Streamlit server for Cloud Run.

Streamlit has no importable "serve this app object" API the way an ASGI framework does — it
runs a *script path*. The image installs the package with `--no-editable`, so that path is
inside site-packages and is not knowable at build time; it is resolved from the import
system at startup rather than hardcoded, which also keeps a local `python -m dashboard.serve`
working from a source checkout.

The server flags are not optional polish. Cloud Run terminates TLS and proxies to the
container over plain HTTP, and IAP sits in front of that, so Streamlit's XSRF and CORS
checks see an origin that never matches and reject the WebSocket upgrade the whole app runs
on — the page loads and then hangs with no error shown. Disabling them is safe because
nothing can reach this container except IAP: Cloud Run refuses any request without
roles/run.invoker, and the only principal holding it is IAP's service agent. (The service's
ingress is open, which the direct IAP integration requires — the authorisation boundary is
the IAM binding, not the network. See modules/cloud_run_service/main.tf.)
"""

import logging
import os
import sys
from importlib.util import find_spec

from obs_common.logging import configure_logging

from dashboard.settings import get_settings


def main() -> None:
    """Validate configuration, then launch the Streamlit server on the port Cloud Run assigns."""
    configure_logging(logging.INFO)

    # Settings are parsed *here*, before the server binds, purely so a misconfigured revision
    # cannot go live. Streamlit does not execute app.py until the first request, and its
    # /_stcore/health endpoint is served by the server itself — so without this a container
    # missing BQ_PROJECT_ID starts fine, passes the startup probe, takes traffic, and only
    # then shows every user a redacted error page. Failing now exits non-zero, the probe never
    # passes, and Cloud Run holds the previous revision in place.
    settings = get_settings()
    logging.getLogger(__name__).info(
        "dashboard starting",
        extra={"project_id": settings.project_id, "risk_table": settings.risk_table},
    )

    from streamlit.web import cli as streamlit_cli

    # find_spec, not `import dashboard.app`. app.py calls main() at module scope, which is
    # how Streamlit scripts are written — importing it here to read __file__ would run the
    # whole page (and its BigQuery query) once at container startup, outside the Streamlit
    # runtime, before the server is even up. find_spec resolves the path without executing
    # the module.
    spec = find_spec("dashboard.app")
    if spec is None or spec.origin is None:
        raise RuntimeError("dashboard.app is not importable — the package is installed wrong")

    sys.argv = [
        "streamlit",
        "run",
        spec.origin,
        # Cloud Run injects PORT; the default matches Cloud Run's own default so a bare
        # `docker run -p 8080:8080` behaves the same as the deployed service.
        "--server.port",
        os.environ.get("PORT", "8080"),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        # Cloud Run's request path is already gzipped at the edge; leaving Streamlit's own
        # compression on just spends container CPU re-doing it.
        "--server.enableWebsocketCompression",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()
