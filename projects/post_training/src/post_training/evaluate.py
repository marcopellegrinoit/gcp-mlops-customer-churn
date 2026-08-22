"""Champion/challenger evaluation stage: runs the promotion gate over already-scored predictions."""

import json
import pathlib

import numpy as np
import pandas as pd
from ml_common import config as ml_config
from ml_common import evaluate as ml_evaluate
from ml_common.preprocess import prepare_features

from post_training.bigquery import read_split
from post_training.storage import download_json, list_blobs


def run_evaluate_stage(
    test_uri: str,
    challenger_uri: str,
    challenger_predictions_dir: str,
    champion_predictions_dir: str,
    champion_shap_path: str,
    champion_threshold_path: str,
    project_id: str,
) -> tuple[dict, dict]:
    """Run the champion/challenger gate from predictions already scored by their own serving images.

    challenger_predictions_dir and champion_predictions_dir are both Vertex AI Batch Prediction
    output directories — the challenger's from a Batch Prediction job against an
    UnmanagedContainerModel pointing at the freshly-trained artifact, the champion's from one
    against its own registered serving container. Neither is computed in-process here — the gate
    decision is always based on exactly what each model's own serving container produced, with no
    separate scoring pass that could drift from it. champion_predictions_dir is "" when no champion
    exists yet (first-ever pipeline run).

    Each model's decision threshold (challenger_meta["threshold"], champion_threshold_path) was
    selected at training time from out-of-fold CV predictions, never from this test set — so the
    F1 reported here isn't optimistically biased by having also picked the operating point on it.

    Returns (metrics, decision): metrics carries the scored facts (no verdict), decision carries
    just the promote bool and the deltas that drove it.
    """
    df = read_split(test_uri, project_id)
    _, y = prepare_features(df)

    challenger_meta = download_json(f"{challenger_uri}/metadata.json")
    challenger_proba = read_batch_predictions(challenger_predictions_dir, df["customer_id"])

    if not champion_predictions_dir:
        metrics = ml_evaluate.compute_full_metrics(
            y_true=y.to_numpy(),
            challenger_proba=challenger_proba,
            challenger_threshold=challenger_meta["threshold"],
            challenger_shap=challenger_meta["shap_importance"],
        )
    else:
        champion_proba = read_batch_predictions(champion_predictions_dir, df["customer_id"])
        champion_shap = json.loads(pathlib.Path(champion_shap_path).read_text().strip() or "null")
        champion_threshold = float(pathlib.Path(champion_threshold_path).read_text())

        metrics = ml_evaluate.compute_full_metrics(
            y_true=y.to_numpy(),
            challenger_proba=challenger_proba,
            challenger_threshold=challenger_meta["threshold"],
            challenger_shap=challenger_meta["shap_importance"],
            champion_proba=champion_proba,
            champion_threshold=champion_threshold,
            champion_shap=champion_shap,
        )

    decision = ml_evaluate.decide(
        metrics,
        pr_auc_min_delta=ml_config.PR_AUC_MIN_DELTA,
        f1_min_delta=ml_config.F1_MIN_DELTA,
    )
    return metrics, decision


def read_batch_predictions(gcs_output_directory: str, customer_ids: pd.Series) -> np.ndarray:
    """Re-join churn_probability from every prediction.results-* shard onto customer_ids' order.

    Batch Prediction shards and parallelizes its output, so the result is not guaranteed to be
    in the same row order as the input instances — re-joining by customer_id is mandatory. The
    batch-predict jobs are configured with key_field="customer_id" (pipeline.py), so each output
    record carries the customer_id under its original field name ("customer_id") rather than nested
    in an echoed "instance" object — customer_id is also excluded from what's actually sent to the
    model as a result. (Despite what the key_field docs say, the output field is not literally
    named "key" — confirmed against actual BatchPredictionJob output.)
    """
    proba_by_customer_id = {}
    for blob in list_blobs(gcs_output_directory):
        # Custom-container (UnmanagedContainerModel) batch prediction writes JSONL-formatted
        # results as "prediction.results-XXXXX-of-YYYYY" with no file extension — unlike
        # AutoML/managed-model batch prediction, which writes "predictions_*.jsonl". Matching on
        # ".jsonl" here silently matched zero files and produced an empty proba_by_customer_id.
        if not blob.name.rsplit("/", 1)[-1].startswith("prediction.results-"):
            continue
        for line in blob.download_as_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            proba_by_customer_id[record["customer_id"]] = record["prediction"][
                ml_config.CHURN_PROBABILITY_FIELD
            ]

    return _proba_in_customer_id_order(proba_by_customer_id, customer_ids)


def _proba_in_customer_id_order(proba_by_customer_id: dict, customer_ids: pd.Series) -> np.ndarray:
    """Reindex a customer_id -> probability mapping to customer_ids' order; raise if any missing."""
    proba = customer_ids.map(proba_by_customer_id)
    if proba.isna().any():
        missing = customer_ids[proba.isna()].tolist()
        raise ValueError(f"no probability found for customer_ids: {missing}")
    return proba.to_numpy()
