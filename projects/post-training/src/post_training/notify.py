"""Terminal stage of the training pipeline: report the run outcome.

Deliberately a logging-only sink. This is the seam where a real deployment would
hand the outcome off to something outside the pipeline — publish to Pub/Sub, post
to Slack, send a SendGrid email — but this is a showcase project with no such
consumer to build against, so the stage only writes a structured summary to
stdout, which Vertex AI Pipelines captures in Cloud Logging against the run. The
step stays in the DAG so the outcome event has a single, obvious home if a
consumer is ever added.
"""

import json
import logging

from post_training.config import MAX_CONSECUTIVE_REJECTIONS


def notify(message: dict) -> None:
    """Log the training pipeline outcome (promotion or rejection) as the run's final event."""
    logging.info("Training pipeline outcome: %s", _summarize(message))
    logging.info("Outcome event payload: %s", json.dumps(message, default=str))


def _summarize(message: dict) -> str:
    """Render the register_or_reject result as a one-line human-readable outcome."""
    if message.get("promoted"):
        return f"challenger PROMOTED to champion as {message.get('model_version')}"

    rejections = message.get("consecutive_rejections")
    summary = f"challenger REJECTED ({rejections} consecutive rejection(s))"
    if message.get("feature_review_alert"):
        summary += (
            f" — feature review recommended: {MAX_CONSECUTIVE_REJECTIONS} or more consecutive"
            " rejections suggest the feature set, not the training data, is the limit"
        )
    return summary
