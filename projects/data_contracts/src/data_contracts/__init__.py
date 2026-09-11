"""Pydantic models for every data schema that crosses a process, container, or storage boundary.

Anything written to GCS, streamed into BigQuery, handed between KFP stages, or returned over
HTTP is described here once and validated at the boundary it crosses. In-process values —
DataFrames, numpy arrays, hyperparameter dicts handed straight to XGBoost — are deliberately
left alone: validating those buys nothing and costs readability.

Tunable *values* are not here. Those live in ml_common.config as environment-driven settings;
this package is about the shape data has to have, which is not something a deployment gets to
change.
"""

from data_contracts.baseline import (
    BaselineSpec,
    CategoricalBaseline,
    DiscreteBaseline,
    NumericBaseline,
    parse_baseline_stats,
)
from data_contracts.drift import (
    DataQualityReport,
    DriftDecision,
    DriftMetricRow,
    NullRateComparison,
    QualityFailure,
)
from data_contracts.evaluation import (
    EvaluationMetrics,
    ModelMetrics,
    PromotionDecision,
    RegistrationResult,
    RejectionState,
)
from data_contracts.metadata import ModelMetadata
from data_contracts.predictions import (
    CHURN_PREDICTION_FIELD,
    CHURN_PROBABILITY_FIELD,
    BatchPredictionRecord,
    ChurnPrediction,
    PredictRequest,
    PredictResponse,
)
from data_contracts.serde import to_json
from data_contracts.splits import SplitRef

__all__ = [
    "CHURN_PREDICTION_FIELD",
    "CHURN_PROBABILITY_FIELD",
    "BaselineSpec",
    "BatchPredictionRecord",
    "CategoricalBaseline",
    "ChurnPrediction",
    "DataQualityReport",
    "DiscreteBaseline",
    "DriftDecision",
    "DriftMetricRow",
    "EvaluationMetrics",
    "ModelMetadata",
    "ModelMetrics",
    "NullRateComparison",
    "NumericBaseline",
    "PredictRequest",
    "PredictResponse",
    "PromotionDecision",
    "QualityFailure",
    "RegistrationResult",
    "RejectionState",
    "SplitRef",
    "parse_baseline_stats",
    "to_json",
]
