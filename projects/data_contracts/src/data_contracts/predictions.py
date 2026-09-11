"""The churn prediction record, in every form it travels in.

One shape, four hops: the serving container returns it, Vertex AI Batch Prediction writes
it to JSONL and to BigQuery, post-training reads it back to score the promotion gate, and
the orchestrator MERGEs it into ml.predictions. The BigQuery columns (iac/config/
bigquery.yaml) and the SQL in drift_monitor.predictions address these fields by *name*, so
the field names here are part of the contract — CHURN_PROBABILITY_FIELD and
CHURN_PREDICTION_FIELD exist so those call sites reference them instead of retyping the
literal, and a test pins them to the model's own field names.
"""

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

CHURN_PROBABILITY_FIELD: Final[str] = "churn_probability"
CHURN_PREDICTION_FIELD: Final[str] = "churn_prediction"


class ChurnPrediction(BaseModel):
    """One customer's churn score and the label it yields at the model's decision threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: bool


class PredictRequest(BaseModel):
    """A Vertex AI custom-container predict request body.

    ``instances`` is deliberately untyped past "a list". Vertex sends named objects when the
    job declares ``instanceConfig.instanceType='object'`` and bare positional arrays when it
    does not, and the second case has a specific, expensive failure mode that
    serving.app._reject_unnamed_instances is written to catch and explain. Typing it as
    ``list[dict]`` here would pre-empt that with a generic 422 and lose the explanation.
    """

    model_config = ConfigDict(extra="allow")

    instances: list[Any]


class PredictResponse(BaseModel):
    """A Vertex AI custom-container predict response body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    predictions: list[ChurnPrediction]


class BatchPredictionRecord(BaseModel):
    """One JSONL line of a Batch Prediction job's ``prediction.results-*`` output.

    ``customer_id`` rather than ``key``: the jobs are configured with
    ``key_field="customer_id"``, and the output carries the key under its original field
    name, not the literal ``"key"`` the docs suggest — confirmed against real job output.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    customer_id: str
    prediction: ChurnPrediction
