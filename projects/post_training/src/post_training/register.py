"""Model registration and rejection stage."""

import time
import uuid

from google.cloud import aiplatform
from google.cloud import exceptions as gcs_exceptions
from ml_common.config import get_settings
from ml_common.contracts import (
    EvaluationMetrics,
    PromotionDecision,
    RegistrationResult,
    RejectionState,
    to_json,
)

from post_training.storage import download_json, upload_text


def register_or_reject(
    metrics: EvaluationMetrics,
    decision: PromotionDecision,
    challenger_uri: str,
    serving_image_uri: str,
    project_id: str,
    region: str,
    model_display_name: str,
    experiment_name: str,
    consecutive_rejection_key: str,
) -> RegistrationResult:
    """Register the challenger only if it was promoted; log every outcome to Vertex AI Experiments.

    Rejected challengers are never registered in Model Registry — there's no shadow
    testing or compliance requirement that needs them there, so registering them would
    only clutter the registry with versions that will never serve traffic. They're
    logged to Vertex AI Experiments instead, purely for lineage/audit.
    """
    settings = get_settings()
    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name,
        staging_bucket=f"gs://{project_id}-pipeline-metadata",
    )
    promoted = decision.promote
    state_uri = f"gs://{project_id}-pipeline-metadata/post-training-state/{experiment_name}/{consecutive_rejection_key}.json"
    current_count = _get_rejection_count(state_uri)
    new_count = 0 if promoted else current_count + 1

    model_version = None
    if promoted:
        # shap_importance/threshold are read back later via fetch_champion._fetch_champion_shap/
        # _fetch_champion_threshold, which download this same artifact_uri's metadata.json —
        # the Vertex AI Model resource's `metadata` field isn't settable through the SDK's
        # Model.upload()/Model.update() for custom-trained models, so the GCS artifact directory
        # (already written by trainer._upload_artifacts) is the source of truth instead.
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=challenger_uri,
            serving_container_image_uri=serving_image_uri,
            # Vertex AI treats any image that isn't one of its own prebuilt containers as a
            # "third-party image" and rejects prediction jobs against it with 400
            # FailedPrecondition ("must specify PredictRoute and HealthRoute in ContainerSpec")
            # unless both routes are declared at upload time. Model.upload() does not default
            # them. These must match the routes the serving app actually exposes and the port it
            # listens on (projects/serving — settings.aip_http_port defaults to 8080), and mirror
            # the containerSpec the challenger's UnmanagedContainerModel declares so both sides
            # of the champion/challenger comparison boot identically.
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            # THRESHOLD has no per-BatchPredictionJob override in the Vertex AI API — it
            # must be baked into the registered Model's container spec at upload time, so
            # every batch prediction job against this model version applies the same
            # decision threshold that was used to evaluate it during this pipeline run.
            # PROJECT_ID is passed explicitly because the batch-predict service account is
            # attached via short-lived credential exchange, which doesn't reliably expose
            # project info to google-cloud-storage's ADC-based project detection.
            serving_container_environment_variables={
                "THRESHOLD": str(metrics.threshold),
                "PROJECT_ID": project_id,
            },
            labels={"role": "champion"},
        )
        _demote_previous_champion(
            exclude=model.resource_name, model_display_name=model_display_name
        )
        model_version = model.resource_name

    _log_run(
        promoted=promoted,
        new_count=new_count,
        consecutive_rejection_key=consecutive_rejection_key,
        metrics=metrics,
        model_version=model_version,
    )
    upload_text(state_uri, to_json(RejectionState(consecutive_rejections=new_count)))

    return RegistrationResult(
        promoted=promoted,
        consecutive_rejections=new_count,
        feature_review_alert=new_count >= settings.max_consecutive_rejections,
        model_version=model_version,
        challenger_metrics=metrics.challenger_metrics,
        champion_metrics=metrics.champion_metrics,
        threshold=metrics.threshold,
        shap_rank_correlation=metrics.shap_rank_correlation,
    )


def _log_run(
    promoted: bool,
    new_count: int,
    consecutive_rejection_key: str,
    metrics: EvaluationMetrics,
    model_version: str | None,
) -> None:
    """Record this run's promotion outcome as a Vertex AI Experiments run, for lineage/audit."""
    run_name = f"register-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    aiplatform.start_run(run_name)
    aiplatform.log_params(
        {
            "promoted": int(promoted),
            consecutive_rejection_key: new_count,
            "model_version": model_version or "",
        }
    )
    aiplatform.log_metrics({"threshold": metrics.threshold})
    aiplatform.end_run()


def _get_rejection_count(state_uri: str) -> int:
    """Read the consecutive rejection counter from its GCS state file; 0 if never written."""
    try:
        return RejectionState.model_validate(download_json(state_uri)).consecutive_rejections
    except gcs_exceptions.NotFound:
        return 0


def _demote_previous_champion(exclude: str, model_display_name: str) -> None:
    """Set role=retired on every champion version except the newly registered one."""
    for model in aiplatform.Model.list(filter=f'display_name="{model_display_name}"'):
        if model.resource_name == exclude:
            continue
        labels = dict(model.labels or {})
        if labels.get("role") == "champion":
            labels["role"] = "retired"
            model.update(labels=labels)
