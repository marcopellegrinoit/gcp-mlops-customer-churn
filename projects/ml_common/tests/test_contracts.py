"""Unit tests for the cross-container data contracts themselves."""

import json

import numpy as np
import pandas as pd
import pytest
from ml_common.contracts import (
    CHURN_PREDICTION_FIELD,
    CHURN_PROBABILITY_FIELD,
    BatchPredictionRecord,
    ChurnPrediction,
    DriftDecision,
    ModelMetadata,
    NumericBaseline,
    SplitRef,
    to_json,
)
from ml_common.drift import compute_baseline_stats
from pydantic import ValidationError


def _metadata() -> ModelMetadata:
    rng = np.random.RandomState(4)
    X = pd.DataFrame(
        {
            "avg_transaction_30d": rng.normal(50, 10, 500),
            "membership_tier": rng.choice(["friend", "champion"], 500),
        }
    )
    return ModelMetadata(
        feature_names=list(X.columns),
        threshold=0.42,
        baseline_stats=compute_baseline_stats(X),
        shap_importance={"avg_transaction_30d": 0.7},
        categorical_categories={"membership_tier": ["champion", "friend"]},
        training_snapshot_date="2026-09-01",
    )


# ---------------------------------------------------------------------------
# metadata.json serialisation — the artifact four other containers read
# ---------------------------------------------------------------------------


def test_metadata_round_trips_through_json():
    metadata = _metadata()
    assert ModelMetadata.model_validate_json(to_json(metadata)) == metadata


def test_infinite_bin_edges_survive_serialisation():
    # The trap this contract's serialiser exists to avoid: pydantic's own JSON encoder
    # writes non-finite floats as null by default, and decile edges are bounded by ±inf on
    # purpose so out-of-range values land in the outermost bucket. Nulling them would make
    # every rebuilt baseline unreadable.
    metadata = _metadata()
    spec = metadata.baseline_stats["avg_transaction_30d"]
    assert isinstance(spec, NumericBaseline)
    assert spec.bin_edges[0] == -np.inf

    reloaded = ModelMetadata.model_validate_json(to_json(metadata))
    assert reloaded.baseline_stats["avg_transaction_30d"].bin_edges[0] == -np.inf


def test_absent_optional_fields_stay_absent_rather_than_null():
    # Readers on both sides test for key *presence* to detect a legacy artifact or an
    # optional section, so an explicit null would read as "present, and false-y".
    minimal = ModelMetadata(feature_names=["a"], threshold=0.5)
    written = json.loads(to_json(minimal))
    assert "training_snapshot_date" not in written
    assert "categorical_categories" not in written


def test_unknown_fields_from_a_newer_trainer_are_preserved():
    # A newer trainer may add a field before every reader has been redeployed.
    metadata = ModelMetadata.model_validate(
        {"feature_names": ["a"], "threshold": 0.5, "calibration_method": "isotonic"}
    )
    assert json.loads(to_json(metadata))["calibration_method"] == "isotonic"


def test_artifact_without_a_threshold_is_rejected():
    # register_or_reject bakes the threshold into the serving container; an artifact
    # without one cannot be served, and the useful place to say so is where it is read.
    with pytest.raises(ValidationError, match="threshold"):
        ModelMetadata.model_validate({"feature_names": ["a"]})


# ---------------------------------------------------------------------------
# prediction field names — also BigQuery column names and SQL string literals
# ---------------------------------------------------------------------------


def test_prediction_field_constants_match_the_model():
    # SQL in drift_monitor.predictions, the orchestrator's MERGE, and the ml.predictions
    # DDL all address these columns by string. The constants exist so those call sites
    # never retype the literal; this is what keeps them honest.
    assert set(ChurnPrediction.model_fields) == {
        CHURN_PROBABILITY_FIELD,
        CHURN_PREDICTION_FIELD,
    }


def test_probability_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValidationError):
        ChurnPrediction(churn_probability=1.4, churn_prediction=True)


def test_batch_prediction_record_keys_on_customer_id():
    # key_field="customer_id" puts the key under its original name, not under "key" —
    # confirmed against real BatchPredictionJob output, despite what the docs suggest.
    record = BatchPredictionRecord.model_validate_json(
        json.dumps(
            {
                "customer_id": "c1",
                "prediction": {CHURN_PROBABILITY_FIELD: 0.8, CHURN_PREDICTION_FIELD: True},
            }
        )
    )
    assert record.prediction.churn_probability == 0.8


# ---------------------------------------------------------------------------
# the drift decision the orchestrator workflow branches on
# ---------------------------------------------------------------------------


def test_no_champion_decision_omits_every_optional_section():
    # The workflow reads data_quality and score_drift_detected through map.get defaults, so
    # a missing key is a defined state there and must not become an explicit null.
    written = json.loads(to_json(DriftDecision(reason="no_champion_registered")))
    assert written["drift_detected"] is False
    assert written["reason"] == "no_champion_registered"
    assert "score_drift_detected" not in written
    assert "data_quality" not in written


def test_decision_rejects_a_misspelled_field():
    # The decision is assembled across four stages that each mutate it; a typo'd attribute
    # would otherwise be silently accepted and the orchestrator would read the old value.
    decision = DriftDecision()
    with pytest.raises(ValidationError):
        decision.drift_detcted = True


# ---------------------------------------------------------------------------
# the split pointer passed between pipeline stages
# ---------------------------------------------------------------------------


def test_split_ref_round_trips():
    ref = SplitRef(snapshot_date="2026-09-01", split="train")
    assert SplitRef.model_validate_json(to_json(ref)) == ref


def test_split_ref_rejects_an_unknown_side():
    with pytest.raises(ValidationError):
        SplitRef(snapshot_date="2026-09-01", split="validation")
