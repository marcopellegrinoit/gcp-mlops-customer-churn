"""Vertex AI Experiments integration: wraps modeling stages with GCP logging and I/O."""

import json
import logging
import tempfile
import uuid

import modeling.hpo as ml_hpo
import modeling.train as ml_train
import pandas as pd
import xgboost as xgb
from google.cloud import aiplatform, storage
from ml_common.config import CHURN_PROBABILITY_FIELD
from ml_common.drift import compute_baseline_stats
from ml_common.preprocess import categorical_categories, prepare_features
from modeling import config as ml_config

from trainer.data import read_split


def run_hpo_stage(
    train_uri: str,
    experiment_name: str,
    project_id: str,
    region: str,
    n_trials: int,
    n_folds: int,
    random_state: int,
) -> dict:
    """Load training data and run HPO, logging each trial to Vertex AI Experiments."""
    df = read_split(train_uri, project_id)
    X, y = prepare_features(df)

    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name,
        staging_bucket=f"gs://{project_id}-pipeline-metadata",
    )

    run_id = uuid.uuid4().hex[
        :8
    ]  # scope run names to this HPO execution so repeat pipeline runs don't collide

    def _trial_callback(
        trial_number: int, params: dict, mean_pr_auc: float, std_pr_auc: float
    ) -> None:
        logging.info(
            "HPO trial %04d: params=%s cv_pr_auc_mean=%.4f cv_pr_auc_std=%.4f",
            trial_number,
            params,
            mean_pr_auc,
            std_pr_auc,
        )
        aiplatform.start_run(f"hpo-trial-{run_id}-{trial_number:04d}")
        aiplatform.log_params(params)
        aiplatform.log_metrics({"cv_pr_auc_mean": mean_pr_auc, "cv_pr_auc_std": std_pr_auc})
        aiplatform.end_run()

    return ml_hpo.run_hpo(
        X,
        y,
        search_space=ml_config.HPO_SEARCH_SPACE,
        fixed_params=ml_config.XGB_FIXED_PARAMS,
        n_trials=n_trials,
        n_folds=n_folds,
        random_state=random_state,
        trial_callback=_trial_callback,
    )


def run_train_stage(
    train_uri: str,
    params: dict,
    artifact_gcs_prefix: str,
    experiment_name: str,
    project_id: str,
    region: str,
    run_name: str,
    n_folds: int,
    random_state: int,
) -> str:
    """Train on the full dataset, log to Vertex AI Experiments, upload artifacts to GCS.

    Also selects the model's decision threshold via out-of-fold CV (select_threshold_via_cv)
    rather than from the held-out test set, so post-training's evaluate stage never picks the
    operating point on the same data it reports F1 against. Returns the gs:// artifact dir URI.
    """
    df = read_split(train_uri, project_id)
    X, y = prepare_features(df)

    logging.info("Training final model with params=%s", params)
    model, feat_names, shap_importance = ml_train.train_model(
        X, y, params, ml_config.XGB_FIXED_PARAMS, random_state
    )
    threshold = ml_train.select_threshold_via_cv(
        X, y, params, ml_config.XGB_FIXED_PARAMS, ml_config.TARGET_RECALL, n_folds, random_state
    )

    artifact_id = uuid.uuid4().hex  # unique prefix so concurrent runs don't overwrite each other
    artifact_uri = f"{artifact_gcs_prefix}/{artifact_id}"

    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name,
        staging_bucket=f"gs://{project_id}-pipeline-metadata",
    )
    aiplatform.start_run(f"{run_name}-{artifact_id[:8]}")
    aiplatform.log_params(params)
    aiplatform.log_metrics({f"shap_{feat}": val for feat, val in shap_importance.items()})
    aiplatform.log_metrics({"threshold": threshold})
    aiplatform.log_params({"artifact_uri": artifact_uri})
    aiplatform.end_run()

    train_scores = pd.DataFrame({CHURN_PROBABILITY_FIELD: model.predict_proba(X)[:, 1]})
    baseline_stats = compute_baseline_stats(X)
    baseline_stats.update(compute_baseline_stats(train_scores))
    cat_categories = categorical_categories(X)
    _upload_artifacts(
        model, feat_names, shap_importance, threshold, baseline_stats, cat_categories, artifact_uri
    )
    return artifact_uri


def _upload_artifacts(
    model: xgb.XGBClassifier,
    feat_names: list[str],
    shap_importance: dict[str, float],
    threshold: float,
    baseline_stats: dict,
    cat_categories: dict[str, list[str]],
    artifact_uri: str,
) -> None:
    """Save model.ubj and metadata.json to a GCS artifact directory.

    baseline_stats freezes this training run's per-feature distribution, plus the
    training-time churn_probability distribution, so the drift monitor can later
    compare live feature and score traffic against them without needing access to
    the original training data. cat_categories similarly freezes each categorical
    column's training-time category list, so inference reuses training's exact
    category->code mapping instead of re-deriving it from whatever a given batch
    happens to contain (see ml_common.preprocess.select_inference_features).
    """
    bucket_name, prefix = _split_uri(artifact_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=True) as tmp:
        model.save_model(tmp.name)
        bucket.blob(f"{prefix}/model.ubj").upload_from_filename(tmp.name)

    metadata = {
        "feature_names": feat_names,
        "shap_importance": shap_importance,
        "threshold": threshold,
        "baseline_stats": baseline_stats,
        "categorical_categories": cat_categories,
    }
    bucket.blob(f"{prefix}/metadata.json").upload_from_string(
        json.dumps(metadata), content_type="application/json"
    )


def _split_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, prefix)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, prefix = path.partition("/")
    return bucket, prefix
