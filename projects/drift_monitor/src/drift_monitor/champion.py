"""Look up the current champion model so its training-time baseline can be compared against."""

from google.cloud import aiplatform


def fetch_champion(model_display_name: str):
    """Return the Vertex AI Model with role=champion label, or None if none exists."""
    for model in aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc",
    ):
        if (model.labels or {}).get("role") == "champion":
            return model
    return None
