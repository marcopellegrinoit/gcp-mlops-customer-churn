"""The model artifact's metadata.json — the contract between the trainer and everything downstream.

Written once by trainer._upload_artifacts alongside model.ubj, then read by four separate
containers that may be running older or newer code than the one that wrote it: the serving
app (feature names and category mappings), post-training's evaluate and fetch_champion
stages (threshold and SHAP importances), the drift monitor (baseline distributions), and
scripts/rebuild_champion_baseline.py.

Optional fields are optional because a champion registered before they existed is still
serving and its artifact cannot be rewritten without retraining it. ``extra="allow"`` is
the same tolerance pointed the other way: a newer trainer may add a field before every
reader has been redeployed.
"""

from pydantic import BaseModel, ConfigDict, Field

from ml_common.contracts.baseline import BaselineSpec


class ModelMetadata(BaseModel):
    """Everything about a trained model that is not the model binary itself."""

    model_config = ConfigDict(frozen=True, extra="allow", protected_namespaces=())

    feature_names: list[str]
    threshold: float
    baseline_stats: dict[str, BaselineSpec] = Field(default_factory=dict)
    shap_importance: dict[str, float] = Field(default_factory=dict)

    # Absent on artifacts frozen before inference reused training's category->code mapping.
    # select_inference_features falls back to inferring categories from the batch, as it did
    # before the field existed.
    categorical_categories: dict[str, list[str]] | None = None

    # Absent on artifacts frozen before the training partition was recorded.
    # rebuild_champion_baseline infers it from the model's registration date instead.
    training_snapshot_date: str | None = None
