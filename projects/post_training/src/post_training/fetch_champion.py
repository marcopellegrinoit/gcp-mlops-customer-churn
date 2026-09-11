"""Look up the current champion model so it can be scored via Vertex AI Batch Prediction."""

from google.cloud import aiplatform
from ml_common.contracts import ModelMetadata

from post_training.storage import download_json


def run_fetch_champion_stage(
    project_id: str, region: str, model_display_name: str
) -> tuple[str, dict | None, float | None]:
    """Return (champion_model_id, champion_shap, champion_threshold).

    champion_model_id is the bare Vertex AI Model ID (not the full resource name):
    ModelGetOp interpolates whatever it receives into
    "projects/{project}/locations/{location}/models/{model_name}", so handing it a full
    resource name produces a doubly-prefixed name and a 400 INVALID_ARGUMENT.

    champion_model_id is "" and the other two are None if no champion exists yet. When a
    champion does exist its metadata.json is validated as a ModelMetadata, so a malformed
    artifact fails here rather than two stages later — champion_threshold used to come back
    None for an artifact missing it, and evaluate then crashed converting "" to a float.

    champion_threshold is the decision threshold the champion was evaluated/promoted
    with (set by register_or_reject from its own out-of-fold CV selection at training
    time) — never recomputed here, so it stays consistent with what's baked into the
    champion's serving container.
    """
    aiplatform.init(
        project=project_id, location=region, staging_bucket=f"gs://{project_id}-pipeline-metadata"
    )
    champion = _fetch_champion(model_display_name)
    if champion is None:
        return "", None, None
    champion_meta = ModelMetadata.model_validate(download_json(f"{champion.uri}/metadata.json"))
    return champion.name, champion_meta.shap_importance, champion_meta.threshold


def _fetch_champion(model_display_name: str):
    """Return the Vertex AI Model with role=champion label, or None if none exists."""
    for model in aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc",
    ):
        if (model.labels or {}).get("role") == "champion":
            return model
    return None
