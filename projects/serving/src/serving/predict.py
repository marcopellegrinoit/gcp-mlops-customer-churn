"""Model loading and scoring for the batch-prediction FastAPI app."""

import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import storage
from ml_common.preprocess import select_inference_features


def load_model(artifact_uri: str, project_id: str) -> xgb.XGBClassifier:
    """Download and deserialise an XGBoost model from a GCS artifact directory."""
    bucket_name, prefix = _split_uri(artifact_uri)
    client = storage.Client(project=project_id)
    # enable_categorical is an sklearn-wrapper attribute, not part of the saved booster, so
    # load_model() won't restore it — it must be re-passed here to match training time
    # (modeling.config.XGB_FIXED_PARAMS), or predict_proba rejects the category-dtype columns.
    model = xgb.XGBClassifier(enable_categorical=True)
    with tempfile.NamedTemporaryFile(suffix=".ubj") as tmp:
        client.bucket(bucket_name).blob(f"{prefix}/model.ubj").download_to_filename(tmp.name)
        model.load_model(tmp.name)
    return model


def score(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
    feature_names: list[str],
    categorical_categories: dict[str, list[str]] | None = None,
) -> np.ndarray:
    """Select the training-time feature set from df and return churn probabilities."""
    X = select_inference_features(df, feature_names, categorical_categories)
    return model.predict_proba(X)[:, 1]


def _split_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, prefix)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, prefix = path.partition("/")
    return bucket, prefix
