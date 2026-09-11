"""Terminal stage of the training pipeline: report the run outcome.

Deliberately a logging-only sink. This is the seam where a real deployment would
hand the outcome off to something outside the pipeline — publish to Pub/Sub, post
to Slack, send a SendGrid email — but this is a showcase project with no such
consumer to build against, so the stage only writes a structured summary to
stdout, which Vertex AI Pipelines captures in Cloud Logging against the run. The
step stays in the DAG so the outcome event has a single, obvious home if a
consumer is ever added.
"""

import logging

from data_contracts import RegistrationResult, to_json
from ml_common.config import get_settings


def notify(message: RegistrationResult) -> None:
    """Log the training pipeline outcome (promotion or rejection) as the run's final event."""
    logging.info("Training pipeline outcome: %s", _summarize(message))
    logging.info("Outcome event payload: %s", to_json(message))


def _summarize(message: RegistrationResult) -> str:
    """Render the register_or_reject result as a one-line human-readable outcome."""
    if message.promoted:
        return f"challenger PROMOTED to champion as {message.model_version}"

    summary = f"challenger REJECTED ({message.consecutive_rejections} consecutive rejection(s))"
    if message.feature_review_alert:
        summary += (
            f" — feature review recommended: {get_settings().max_consecutive_rejections} or more"
            " consecutive rejections suggest the feature set, not the training data, is the limit"
        )
    return summary
