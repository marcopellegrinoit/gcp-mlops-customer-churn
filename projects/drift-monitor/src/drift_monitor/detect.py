"""Drift detection: compare the latest feature snapshot against the champion's baseline."""

from google.cloud import aiplatform
from ml_common.drift import evaluate_drift
from ml_common.preprocess import select_inference_features

from drift_monitor.bigquery import fetch_latest_snapshot
from drift_monitor.champion import fetch_champion
from drift_monitor.storage import download_json, upload_json


def run_drift_check(
    project_id: str,
    region: str,
    bq_features_table: str,
    model_display_name: str,
    decision_gcs_uri: str,
    psi_threshold: float,
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

    upload_json(decision_gcs_uri, result)
    return result
