"""Unit tests for the logging-only notify stage (no GCP dependencies)."""

import logging

from post_training.notify import notify


def test_notify_logs_promotion(caplog):
    with caplog.at_level(logging.INFO):
        notify({"promoted": True, "model_version": "projects/p/locations/l/models/123"})
    assert "PROMOTED" in caplog.text
    assert "projects/p/locations/l/models/123" in caplog.text


def test_notify_logs_rejection_with_counter(caplog):
    with caplog.at_level(logging.INFO):
        notify({"promoted": False, "consecutive_rejections": 1, "feature_review_alert": False})
    assert "REJECTED" in caplog.text
    assert "feature review" not in caplog.text


def test_notify_logs_feature_review_alert(caplog):
    with caplog.at_level(logging.INFO):
        notify({"promoted": False, "consecutive_rejections": 3, "feature_review_alert": True})
    assert "feature review recommended" in caplog.text


def test_notify_serializes_full_payload(caplog):
    with caplog.at_level(logging.INFO):
        notify({"promoted": True, "challenger_metrics": {"pr_auc": 0.81}, "threshold": 0.42})
    assert '"pr_auc": 0.81' in caplog.text


def test_notify_tolerates_non_serializable_values(caplog):
    # The payload is whatever register_or_reject produced; a dummy sink must never be the
    # thing that fails a pipeline run that already succeeded.
    with caplog.at_level(logging.INFO):
        notify({"promoted": False, "consecutive_rejections": object()})
    assert "REJECTED" in caplog.text
