"""Drift detection: compare the latest feature snapshot against the champion's baseline."""

import logging

import pandas as pd
from google.cloud import aiplatform
from ml_common.config import CHURN_PROBABILITY_FIELD
from ml_common.drift import compute_psi, evaluate_drift, psi_noise_floor
from ml_common.preprocess import select_inference_features

from drift_monitor.bigquery import fetch_latest_snapshot
from drift_monitor.champion import fetch_champion
from drift_monitor.history import prior_breach_counts, record_run
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
    persistence_window: int,
    persistence_min_breaches: int,
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
    current = select_inference_features(
        df, metadata["feature_names"], metadata.get("categorical_categories")
    )

    result = evaluate_drift(metadata["baseline_stats"], current, psi_threshold)
    result["champion_model"] = champion.resource_name
    result["snapshot_date"] = snapshot_date

    _add_score_drift(
        result, metadata, champion, project_id, batch_predict_display_name, psi_threshold
    )

    run_ts = pd.Timestamp.now(tz="UTC")
    _apply_persistence_rule(
        result, project_id, champion.resource_name, persistence_window, persistence_min_breaches
    )
    _record_history(result, project_id, run_ts, snapshot_date, champion.resource_name)

    upload_json(decision_gcs_uri, result)
    return result


def _apply_persistence_rule(
    result: dict,
    project_id: str,
    champion_model: str,
    persistence_window: int,
    persistence_min_breaches: int,
) -> None:
    """Require a breach to repeat across runs before it gates retraining; edits result in place.

    A single night's breach is not evidence of a distribution shift — a partial upstream
    load or an unlucky sample produces one too, and retraining on it is what turns a noisy
    detector into a nightly retrain loop. Requiring N breaches within the last M runs also
    absorbs the multiple-comparison risk of testing every feature independently every day:
    at a 1-in-100 per-feature false-alarm rate, the same feature misfiring twice in three
    runs is a ~3-in-10,000 event.

    result["run_drift_detected"] keeps this run's raw verdict for the record;
    result["drift_detected"] becomes the persistent verdict the orchestrator acts on.
    """
    run_breaches = result["breached_features"]
    result["run_drift_detected"] = bool(run_breaches)
    result["persistence_window"] = persistence_window
    result["persistence_min_breaches"] = persistence_min_breaches

    try:
        prior = prior_breach_counts(project_id, champion_model, persistence_window - 1)
    except Exception:
        # No history (first run after deploy) or an unreadable table must not fail the check
        # or silently fall back to single-run triggering. Counting zero prior breaches is the
        # fail-safe direction: a real shift still fires once it has repeated enough times.
        log.exception("Could not read drift history; treating this run as the first breach.")
        prior = {}

    consecutive = {feature: prior.get(feature, 0) + 1 for feature in run_breaches}
    persistent = {
        feature: run_breaches[feature]
        for feature, count in consecutive.items()
        if count >= persistence_min_breaches
    }

    result["breach_counts"] = consecutive
    result["drift_detected"] = bool(persistent)
    result["persistent_breaches"] = persistent

    if run_breaches and not persistent:
        log.info(
            "Breach(es) %s have not yet repeated %d times in the last %d runs; not retraining.",
            sorted(run_breaches),
            persistence_min_breaches,
            persistence_window,
        )


def _record_history(
    result: dict, project_id: str, run_ts: pd.Timestamp, snapshot_date: str, champion_model: str
) -> None:
    """Append this run's per-feature PSI to ml.drift_metrics; never fails the check."""
    try:
        record_run(project_id, run_ts, snapshot_date, champion_model, result)
    except Exception:
        # Losing one run's history degrades the persistence rule for subsequent runs (a
        # breach takes one extra run to accumulate) but must not discard a decision that
        # has already been computed correctly.
        log.exception("Could not record drift metrics for this run.")


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
    score_spec = metadata["baseline_stats"].get(CHURN_PROBABILITY_FIELD)
    if score_spec is None:
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

        score_psi = compute_psi({CHURN_PROBABILITY_FIELD: score_spec}, predictions)[
            CHURN_PROBABILITY_FIELD
        ]
        # Same two-part threshold the feature check uses: a score shift must clear both the
        # configured effect size and the sampling noise of this batch's size.
        score_threshold = max(psi_threshold, psi_noise_floor(score_spec, len(predictions)))
        result["score_psi"] = score_psi
        result["score_threshold"] = score_threshold
        result["score_drift_detected"] = score_psi > score_threshold
    except Exception:
        log.exception("Score-drift check failed; continuing with feature-PSI result only.")
