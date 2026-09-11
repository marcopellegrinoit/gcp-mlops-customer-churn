"""Unit tests for the ml.drift_metrics rows one run appends (no BigQuery required)."""

import drift_monitor.history as history_module
import pandas as pd
import pytest
from ml_common.config import get_settings
from ml_common.contracts import CHURN_PROBABILITY_FIELD, DriftDecision

_RUN_TS = pd.Timestamp("2026-09-11T02:00:00Z")


@pytest.fixture()
def inserted(monkeypatch):
    """Capture what record_run would stream to BigQuery."""
    captured = {}

    class _FakeClient:
        def __init__(self, project=None):
            captured["project"] = project

        def insert_rows_json(self, table, rows):
            captured["table"] = table
            captured["rows"] = rows
            return []

    monkeypatch.setattr(history_module.bigquery, "Client", _FakeClient)
    return captured


def _decision(**overrides) -> DriftDecision:
    fields = dict(
        drift_detected=True,
        feature_psi={"avg_transaction_30d": 0.31, "renewal_count": 0.02},
        feature_thresholds={"avg_transaction_30d": 0.2, "renewal_count": 0.2},
        breached_features={"avg_transaction_30d": 0.31},
        unmonitored_features=["renewal_count"],
    )
    return DriftDecision(**(fields | overrides))


def test_one_row_per_evaluated_feature(inserted):
    history_module.record_run("proj", _RUN_TS, "2026-09-10", "models/123", _decision())

    rows = inserted["rows"]
    assert {row["feature"] for row in rows} == {"avg_transaction_30d", "renewal_count"}
    breached = next(r for r in rows if r["feature"] == "avg_transaction_30d")
    assert breached["breached"] is True
    assert breached["monitored"] is True
    unmonitored = next(r for r in rows if r["feature"] == "renewal_count")
    assert unmonitored["breached"] is False
    assert unmonitored["monitored"] is False


def test_rows_are_json_safe_for_the_streaming_insert(inserted):
    # insert_rows_json serialises what it is handed; a datetime or an enum reaching it
    # fails the whole batch, and losing a run's history silently degrades the persistence
    # rule for every run after it.
    history_module.record_run("proj", _RUN_TS, "2026-09-10", "models/123", _decision())

    row = inserted["rows"][0]
    assert isinstance(row["run_ts"], str)
    assert row["run_ts"].startswith("2026-09-11T02:00:00")
    assert set(row) == {
        "run_ts",
        "snapshot_date",
        "champion_model",
        "feature",
        "psi",
        "threshold",
        "breached",
        "monitored",
    }


def test_score_drift_joins_the_same_time_series(inserted):
    # The score check reports outside feature_psi because it is a model output rather than
    # an input feature, but it belongs in the same history — otherwise it is only ever
    # visible in the decision file, which is overwritten daily.
    decision = _decision(score_psi=0.4, score_threshold=0.25, score_drift_detected=True)
    history_module.record_run("proj", _RUN_TS, "2026-09-10", "models/123", decision)

    score_row = next(r for r in inserted["rows"] if r["feature"] == CHURN_PROBABILITY_FIELD)
    assert score_row["psi"] == 0.4
    assert score_row["breached"] is True


def test_nothing_is_written_when_no_feature_was_evaluated(inserted):
    history_module.record_run(
        "proj", _RUN_TS, "2026-09-10", "models/123", DriftDecision(reason="no_champion_registered")
    )
    assert "rows" not in inserted


def test_target_table_comes_from_settings(inserted):
    history_module.record_run("proj", _RUN_TS, "2026-09-10", "models/123", _decision())
    assert inserted["table"] == f"proj.{get_settings().drift_metrics_table}"


def test_insert_errors_are_raised(inserted, monkeypatch):
    class _FailingClient:
        def __init__(self, project=None):
            pass

        def insert_rows_json(self, table, rows):
            return [{"index": 0, "errors": [{"reason": "invalid"}]}]

    monkeypatch.setattr(history_module.bigquery, "Client", _FailingClient)
    with pytest.raises(RuntimeError, match="Failed to record drift metrics"):
        history_module.record_run("proj", _RUN_TS, "2026-09-10", "models/123", _decision())
