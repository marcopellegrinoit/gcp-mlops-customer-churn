"""Cloud Run Job entrypoint: run the PSI drift check once and write the decision to GCS."""

import logging

from obs_common.logging import configure_logging

from drift_monitor import config as platform_config
from drift_monitor.detect import run_drift_check

configure_logging()
log = logging.getLogger(__name__)


def main() -> None:
    """Run the drift check and log the decision; the orchestrator reads the result from GCS."""
    decision_gcs_uri = f"gs://{platform_config.PROJECT_ID}-{platform_config.GCS_BUCKET}/{platform_config.DECISION_BLOB}"
    result = run_drift_check(
        project_id=platform_config.PROJECT_ID,
        region=platform_config.REGION,
        bq_features_table=platform_config.BQ_FEATURES_TABLE,
        model_display_name=platform_config.MODEL_DISPLAY_NAME,
        decision_gcs_uri=decision_gcs_uri,
        psi_threshold=platform_config.PSI_THRESHOLD,
        batch_predict_display_name=platform_config.BATCH_PREDICT_DISPLAY_NAME,
    )
    log.info("Drift check result: %s", result)


if __name__ == "__main__":
    main()
