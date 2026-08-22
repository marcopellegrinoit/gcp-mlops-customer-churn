"""Unit tests for the evaluate stage (no GCP credentials required)."""

import json

import numpy as np
import pandas as pd
import post_training.evaluate as evaluate_module
import pytest
from ml_common.config import CHURN_PROBABILITY_FIELD
from post_training.evaluate import read_batch_predictions, run_evaluate_stage


class _FakeBlob:
    def __init__(self, name, lines):
        self.name = name
        self._lines = lines

    def download_as_text(self):
        return "\n".join(json.dumps(line) for line in self._lines)


_TEST_REF = json.dumps({"snapshot_date": "2026-07-19", "split": "test"})


# ---------------------------------------------------------------------------
# read_batch_predictions
# ---------------------------------------------------------------------------


def test_read_batch_predictions_aggregates_across_shards(monkeypatch):
    blobs = [
        _FakeBlob(
            "prediction.results-00000-of-00002",
            [
                {"customer_id": "d1", "prediction": {CHURN_PROBABILITY_FIELD: 0.1}},
            ],
        ),
        _FakeBlob(
            "prediction.results-00001-of-00002",
            [
                {"customer_id": "d2", "prediction": {CHURN_PROBABILITY_FIELD: 0.9}},
            ],
        ),
    ]
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: blobs)

    proba = read_batch_predictions("gs://bucket/output", pd.Series(["d1", "d2"]))
    np.testing.assert_array_almost_equal(proba, [0.1, 0.9])


def test_read_batch_predictions_ignores_error_shards(monkeypatch):
    blobs = [
        _FakeBlob(
            "prediction.results-00000-of-00001",
            [
                {"customer_id": "d1", "prediction": {CHURN_PROBABILITY_FIELD: 0.1}},
            ],
        ),
        _FakeBlob(
            "prediction.errors_stats-00000-of-00001",
            [
                {"customer_id": "d2", "error": {"code": 13, "message": "boom"}},
            ],
        ),
    ]
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: blobs)

    with pytest.raises(ValueError, match="d2"):
        read_batch_predictions("gs://bucket/output", pd.Series(["d1", "d2"]))


def test_read_batch_predictions_aligns_to_input_order_even_when_shuffled(monkeypatch):
    # Output shard order is reversed/shuffled relative to the requested customer_id order —
    # the join must still return probabilities aligned to the *input* row order.
    blobs = [
        _FakeBlob(
            "prediction.results-00000-of-00001",
            [
                {"customer_id": "d3", "prediction": {CHURN_PROBABILITY_FIELD: 0.3}},
                {"customer_id": "d1", "prediction": {CHURN_PROBABILITY_FIELD: 0.1}},
                {"customer_id": "d2", "prediction": {CHURN_PROBABILITY_FIELD: 0.2}},
            ],
        ),
    ]
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: blobs)

    proba = read_batch_predictions("gs://bucket/output", pd.Series(["d1", "d2", "d3"]))
    np.testing.assert_array_almost_equal(proba, [0.1, 0.2, 0.3])


def test_read_batch_predictions_raises_on_missing_customer_id(monkeypatch):
    blobs = [
        _FakeBlob(
            "prediction.results-00000-of-00001",
            [
                {"customer_id": "d1", "prediction": {CHURN_PROBABILITY_FIELD: 0.1}},
            ],
        ),
    ]
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: blobs)

    with pytest.raises(ValueError, match="d2"):
        read_batch_predictions("gs://bucket/output", pd.Series(["d1", "d2"]))


# ---------------------------------------------------------------------------
# run_evaluate_stage
# ---------------------------------------------------------------------------


def _patch_common(monkeypatch, df):
    monkeypatch.setattr(evaluate_module, "read_split", lambda ref, project_id: df)
    monkeypatch.setattr(
        evaluate_module,
        "download_json",
        lambda uri: {"shap_importance": {"feature_a": 1.0}, "threshold": 0.5},
    )


def _test_df():
    return pd.DataFrame(
        {
            "customer_id": ["d1", "d2", "d3", "d4"],
            "avg_transaction_30d": [10.0, 20.0, 30.0, 40.0],
            "membership_tier": ["gold", "silver", "gold", "bronze"],
            "region": ["north", "south", "north", "east"],
            "preferred_channel": ["email", "sms", "email", "mail"],
            "churned": [0, 1, 0, 1],
        }
    )


def _blobs_for(df, churn_probability=0.5):
    return [
        _FakeBlob(
            "prediction.results-00000-of-00001",
            [
                {"customer_id": d, "prediction": {CHURN_PROBABILITY_FIELD: churn_probability}}
                for d in df["customer_id"]
            ],
        ),
    ]


def test_run_evaluate_stage_no_champion_skips_champion_scoring(monkeypatch):
    df = _test_df()
    _patch_common(monkeypatch, df)
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: _blobs_for(df))

    metrics, decision = run_evaluate_stage(
        test_uri=_TEST_REF,
        challenger_uri="gs://bucket/artifacts/abc",
        challenger_predictions_dir="gs://bucket/challenger-output",
        champion_predictions_dir="",
        champion_shap_path="/nonexistent/path/should/not/be/read.json",
        champion_threshold_path="/nonexistent/path/should/not/be/read.txt",
        project_id="proj",
    )
    assert decision["promote"] is True
    assert metrics["champion_metrics"] is None


def test_run_evaluate_stage_with_champion_reads_predictions_and_shap(monkeypatch, tmp_path):
    df = _test_df()
    _patch_common(monkeypatch, df)
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: _blobs_for(df))

    shap_path = tmp_path / "champion_shap.json"
    shap_path.write_text(json.dumps({"feature_a": 0.8}))
    threshold_path = tmp_path / "champion_threshold.txt"
    threshold_path.write_text("0.5")

    metrics, _ = run_evaluate_stage(
        test_uri=_TEST_REF,
        challenger_uri="gs://bucket/artifacts/abc",
        challenger_predictions_dir="gs://bucket/challenger-output",
        champion_predictions_dir="gs://bucket/batch-test/output",
        champion_shap_path=str(shap_path),
        champion_threshold_path=str(threshold_path),
        project_id="proj",
    )
    assert metrics["champion_metrics"] is not None
    assert (
        metrics["shap_rank_correlation"] is None
    )  # only one shared feature, correlation needs >=2


def test_run_evaluate_stage_raises_on_missing_challenger_prediction(monkeypatch):
    df = _test_df()
    _patch_common(monkeypatch, df)
    blobs = [
        _FakeBlob(
            "prediction.results-00000-of-00001",
            [
                {"customer_id": d, "prediction": {CHURN_PROBABILITY_FIELD: 0.5}}
                for d in ["d1", "d2", "d3"]  # d4 missing
            ],
        ),
    ]
    monkeypatch.setattr(evaluate_module, "list_blobs", lambda prefix: blobs)

    with pytest.raises(ValueError, match="d4"):
        run_evaluate_stage(
            test_uri=_TEST_REF,
            challenger_uri="gs://bucket/artifacts/abc",
            challenger_predictions_dir="gs://bucket/challenger-output",
            champion_predictions_dir="",
            champion_shap_path="/nonexistent/path/should/not/be/read.json",
            champion_threshold_path="/nonexistent/path/should/not/be/read.txt",
            project_id="proj",
        )
