"""Drift detection: compare the latest feature snapshot against the champion's baseline."""

import logging

from google.cloud import aiplatform
from ml_common.config import CHURN_PROBABILITY_FIELD
from ml_common.drift import compute_psi, evaluate_drift
from ml_common.preprocess import select_inference_features

from drift_monitor.bigquery import fetch_latest_snapshot
from drift_monitor.champion import fetch_champion
from drift_monitor.predictions import fetch_latest_predictions
from drift_monitor.storage import download_json, upload_json

log = logging.getLogger(__name__)


def run_drift_check(
    project_id: str,
    region: str,
    bq_features_table: str,
    model_display_name: str,
    decision_gcs_uri: str,
    psi_threshold: float,
    batch_predict_display_name: str,
) -> dict:
    """Run the PSI drift check against the current champion and write the decision to GCS.

    No champion registered yet (first-ever pipeline run) is reported as no drift —
    there is nothing to compare the live feature snapshot against.
    """
    aiplatform.init(
        project=project_id, location=region, staging_bucket=f"gs://{project_id}-pipeline-metadata"
    )

    champion = fetch_champion(model_display_name)
    if champion is None:
        result = {"drift_detected": False, "reason": "no_champion_registered"}
        upload_json(decision_gcs_uri, result)
        return result

    metadata = download_json(f"{champion.uri}/metadata.json")
    snapshot_date, df = fetch_latest_snapshot(project_id, bq_features_table)
    current = select_inference_features(df, metadata["feature_names"])

    result = evaluate_drift(metadata["baseline_stats"], current, psi_threshold)
    result["champion_model"] = champion.resource_name
    result["snapshot_date"] = snapshot_date

    _add_score_drift(
        result, metadata, champion, project_id, batch_predict_display_name, psi_threshold
    )

    upload_json(decision_gcs_uri, result)
    return result


def _add_score_drift(
    result: dict,
    metadata: dict,
    champion,
    project_id: str,
    batch_predict_display_name: str,
    psi_threshold: float,
) -> None:
    """Add score_psi/score_drift_detected to result in place; monitoring-only, never raises.

    Reported separately from evaluate_drift's drift_detected/breached_features, which stay
    feature-only and keep driving retraining — a score-side failure here (BigQuery permission
    hiccup, malformed BatchPredictionJob output_info, Vertex API flakiness) must degrade to "no
    score signal this run," not take down the feature-PSI result that gates retraining.
    """
    if CHURN_PROBABILITY_FIELD not in metadata["baseline_stats"]:
        return  # artifact predates this check (trained before the score baseline was added)

    try:
        predictions = fetch_latest_predictions(
            project_id, champion.resource_name, batch_predict_display_name
        )
        if predictions is None:
            return
        predictions = predictions.dropna(subset=[CHURN_PROBABILITY_FIELD])
        if predictions.empty:
            return  # rows present but no usable scores (e.g. a scoring defect) -> no signal

        score_psi = compute_psi(metadata["baseline_stats"], predictions)[CHURN_PROBABILITY_FIELD]
        result["score_psi"] = score_psi
        result["score_drift_detected"] = score_psi > psi_threshold
    except Exception:
        log.exception("Score-drift check failed; continuing with feature-PSI result only.")
