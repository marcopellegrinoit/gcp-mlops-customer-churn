"""Cloud Run Job entrypoint: run the PSI drift check once and write the decision to GCS."""

import logging

from obs_common.logging import configure_logging

from drift_monitor.config import get_settings
from drift_monitor.detect import run_drift_check

configure_logging()
log = logging.getLogger(__name__)


def main() -> None:
    """Run the drift check and log the decision; the orchestrator reads the result from GCS."""
    result = run_drift_check(get_settings())
    log.info("Drift check result: %s", result)


if __name__ == "__main__":
    main()
