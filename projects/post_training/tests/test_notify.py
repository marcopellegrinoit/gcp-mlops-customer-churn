"""Unit tests for the logging-only notify stage (no GCP dependencies)."""

import logging

from ml_common.contracts import ModelMetrics, RegistrationResult
from post_training.notify import notify


def _result(**overrides) -> RegistrationResult:
    fields = dict(promoted=False, consecutive_rejections=0, feature_review_alert=False)
    return RegistrationResult(**(fields | overrides))


def test_notify_logs_promotion(caplog):
    with caplog.at_level(logging.INFO):
        notify(_result(promoted=True, model_version="projects/p/locations/l/models/123"))
    assert "PROMOTED" in caplog.text
    assert "projects/p/locations/l/models/123" in caplog.text


def test_notify_logs_rejection_with_counter(caplog):
    with caplog.at_level(logging.INFO):
        notify(_result(consecutive_rejections=1))
    assert "REJECTED" in caplog.text
    assert "feature review" not in caplog.text


def test_notify_logs_feature_review_alert(caplog):
    with caplog.at_level(logging.INFO):
        notify(_result(consecutive_rejections=3, feature_review_alert=True))
    assert "feature review recommended" in caplog.text


def test_notify_serializes_full_payload(caplog):
    metrics = ModelMetrics(pr_auc=0.81, roc_auc=0.88, f1=0.7, threshold=0.42)
    with caplog.at_level(logging.INFO):
        notify(_result(promoted=True, challenger_metrics=metrics, threshold=0.42))
    assert '"pr_auc": 0.81' in caplog.text


def test_notify_never_sees_a_payload_it_cannot_serialize(caplog):
    # This stage used to take an arbitrary dict and defend against unserializable values,
    # because a dummy sink must never fail a run that already succeeded. RegistrationResult
    # moves that guarantee upstream: the payload is parsed at the stage boundary
    # (post_training.main._notify), so anything reaching here is already a valid, JSON-safe
    # outcome — and a malformed one fails where it was produced instead of at the last step.
    with caplog.at_level(logging.INFO):
        notify(_result(consecutive_rejections=2))
    assert '"consecutive_rejections": 2' in caplog.text
