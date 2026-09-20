"""The champion/challenger gate's outputs, as they travel between pipeline stages.

Every one of these crosses a container boundary as a JSON file on a KFP artifact path:
post-training's evaluate stage writes the metrics and the decision, register_or_reject
reads them back and writes its own result, and notify reads that. Same image today, but
nothing in the pipeline guarantees that — and the promotion of a model to production
traffic is the last place to discover a key was renamed.
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelMetrics(BaseModel):
    """One model's scores on the held-out test set, at its own pre-selected threshold.

    ``threshold`` is chosen at training time from out-of-fold CV predictions, never from
    the test set these metrics are computed on, so ``f1`` is not optimistically biased by
    having also picked the operating point on it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pr_auc: float
    roc_auc: float
    f1: float
    threshold: float


class EvaluationMetrics(BaseModel):
    """The scored facts the promotion gate is decided from — no verdict of its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    challenger_metrics: ModelMetrics
    # None on the first-ever pipeline run, when nothing has been promoted yet.
    champion_metrics: ModelMetrics | None = None
    # The challenger's threshold, baked into its serving container if it is promoted.
    threshold: float
    # Spearman correlation of challenger and champion SHAP rankings; None without a champion.
    shap_rank_correlation: float | None = None
    challenger_shap_importance: dict[str, float] = Field(default_factory=dict)


class PromotionDecision(BaseModel):
    """Whether the challenger takes over production traffic, and the deltas that decided it.

    Both deltas are None when there is no champion — the challenger is promoted
    unconditionally in that case, so there is nothing to have measured it against.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    promote: bool
    pr_auc_delta: float | None = None
    f1_delta: float | None = None


class RegistrationResult(BaseModel):
    """Outcome of the register-or-reject stage, and the pipeline's terminal event payload.

    ``feature_review_alert`` fires once rejections have accumulated past their limit: a
    challenger that repeatedly cannot beat the champion points at the feature set rather
    than at the training data, which is a human's problem rather than the pipeline's.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    promoted: bool
    consecutive_rejections: int
    feature_review_alert: bool
    # Vertex AI Model resource name; None whenever the challenger was rejected, since
    # rejected challengers are deliberately never registered.
    model_version: str | None = None

    challenger_metrics: ModelMetrics | None = None
    champion_metrics: ModelMetrics | None = None
    threshold: float | None = None
    shap_rank_correlation: float | None = None


class RejectionState(BaseModel):
    """The consecutive-rejection counter, persisted to GCS between pipeline runs.

    A dedicated state file rather than a query over Vertex AI Experiments run history:
    ``ExperimentRun.list()`` resolves every run in the experiment — including the ~100 HPO
    trials each retraining cycle logs — and each resolution costs its own Tensorboard
    time-series lookup, a fan-out that grows every cycle until it trips the aiplatform
    regional quota. This read is O(1) however long the history gets.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    consecutive_rejections: int = 0
