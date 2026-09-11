"""Drift detection: compare the latest feature snapshot against the champion's baseline."""

import logging

import pandas as pd
from google.cloud import aiplatform
from ml_common.contracts import (
    CHURN_PROBABILITY_FIELD,
    DataQualityReport,
    DriftDecision,
    ModelMetadata,
    QualityFailure,
)
from ml_common.data_quality import check_data_quality
from ml_common.drift import compute_psi, evaluate_drift, psi_noise_floor
from ml_common.preprocess import select_inference_features

from drift_monitor.bigquery import fetch_latest_snapshot, fetch_quality_context
from drift_monitor.champion import fetch_champion
from drift_monitor.config import DriftMonitorSettings
from drift_monitor.history import prior_breach_counts, record_run
from drift_monitor.predictions import fetch_latest_predictions
from drift_monitor.storage import download_json, upload_decision

log = logging.getLogger(__name__)


def run_drift_check(settings: DriftMonitorSettings) -> DriftDecision:
    """Run the PSI drift check against the current champion and write the decision to GCS.

    No champion registered yet (first-ever pipeline run) is reported as no drift —
    there is nothing to compare the live feature snapshot against.
    """
    aiplatform.init(
        project=settings.project_id,
        location=settings.region,
        staging_bucket=settings.staging_bucket_uri,
    )

    champion = fetch_champion(settings.model_display_name)
    if champion is None:
        decision = DriftDecision(drift_detected=False, reason="no_champion_registered")
        upload_decision(settings.decision_gcs_uri, decision)
        return decision

    metadata = ModelMetadata.model_validate(download_json(f"{champion.uri}/metadata.json"))
    snapshot_date, df = fetch_latest_snapshot(settings.project_id, settings.bq_features_table)
    current = select_inference_features(df, metadata.feature_names, metadata.categorical_categories)

    decision = evaluate_drift(metadata.baseline_stats, current, settings.psi_threshold)
    decision.champion_model = champion.resource_name
    decision.snapshot_date = snapshot_date

    _add_score_drift(decision, metadata, champion, settings)

    run_ts = pd.Timestamp.now(tz="UTC")
    _apply_persistence_rule(decision, champion.resource_name, settings)
    _apply_data_quality_gate(decision, current, metadata, snapshot_date, settings)
    _record_history(decision, settings.project_id, run_ts, snapshot_date, champion.resource_name)

    upload_decision(settings.decision_gcs_uri, decision)
    return decision


def _apply_data_quality_gate(
    decision: DriftDecision,
    current: pd.DataFrame,
    metadata: ModelMetadata,
    snapshot_date: str,
    settings: DriftMonitorSettings,
) -> None:
    """Suppress the retrain decision if the snapshot itself is broken; edits decision in place.

    Runs after the drift verdict rather than before it so the PSI numbers are still computed
    and recorded — they are useful evidence when diagnosing the defect — but it overrides
    drift_detected, because retraining on a corrupted snapshot bakes the corruption into the
    model. The promotion gate offers no protection here either: the challenger is evaluated
    against a test split drawn from the same bad snapshot, so it can score well and be
    promoted on the strength of the defect.

    A failure to run the check is itself treated as a failure to clear it. This is the one
    place in this job that fails closed: every other degradation (missing history, a
    score-side error) biases toward not retraining anyway, and so does this one.
    """
    try:
        recent_row_counts, duplicate_key_count = fetch_quality_context(
            settings.project_id,
            settings.bq_features_table,
            snapshot_date,
            settings.quality_history_partitions,
        )
        quality = check_data_quality(
            current, metadata.baseline_stats, recent_row_counts, duplicate_key_count
        )
    except Exception:
        log.exception("Data-quality check could not run; suppressing retraining for this run.")
        quality = DataQualityReport(
            data_quality_failed=True,
            failures=[
                QualityFailure(check="check_failed", detail="the data-quality check itself errored")
            ],
            row_count=len(current),
        )

    decision.data_quality = quality
    if not quality.data_quality_failed:
        return

    log.error(
        "Data-quality assertions failed; not retraining: %s",
        "; ".join(f.detail for f in quality.failures),
    )
    decision.retrain_suppressed_by_data_quality = decision.drift_detected
    decision.drift_detected = False


def _apply_persistence_rule(
    decision: DriftDecision, champion_model: str, settings: DriftMonitorSettings
) -> None:
    """Require a breach to repeat across runs before it gates retraining; edits decision in place.

    A single night's breach is not evidence of a distribution shift — a partial upstream
    load or an unlucky sample produces one too, and retraining on it is what turns a noisy
    detector into a nightly retrain loop. Requiring N breaches within the last M runs also
    absorbs the multiple-comparison risk of testing every feature independently every day:
    at a 1-in-100 per-feature false-alarm rate, the same feature misfiring twice in three
    runs is a ~3-in-10,000 event.

    decision.run_drift_detected keeps this run's raw verdict for the record;
    decision.drift_detected becomes the persistent verdict the orchestrator acts on.
    """
    run_breaches = decision.breached_features
    decision.run_drift_detected = bool(run_breaches)
    decision.persistence_window = settings.persistence_window
    decision.persistence_min_breaches = settings.persistence_min_breaches

    try:
        prior = prior_breach_counts(
            settings.project_id, champion_model, settings.persistence_window - 1
        )
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
        if count >= settings.persistence_min_breaches
    }

    decision.breach_counts = consecutive
    decision.drift_detected = bool(persistent)
    decision.persistent_breaches = persistent

    if run_breaches and not persistent:
        log.info(
            "Breach(es) %s have not yet repeated %d times in the last %d runs; not retraining.",
            sorted(run_breaches),
            settings.persistence_min_breaches,
            settings.persistence_window,
        )


def _record_history(
    decision: DriftDecision,
    project_id: str,
    run_ts: pd.Timestamp,
    snapshot_date: str,
    champion_model: str,
) -> None:
    """Append this run's per-feature PSI to ml.drift_metrics; never fails the check."""
    try:
        record_run(project_id, run_ts, snapshot_date, champion_model, decision)
    except Exception:
        # Losing one run's history degrades the persistence rule for subsequent runs (a
        # breach takes one extra run to accumulate) but must not discard a decision that
        # has already been computed correctly.
        log.exception("Could not record drift metrics for this run.")


def _add_score_drift(
    decision: DriftDecision,
    metadata: ModelMetadata,
    champion,
    settings: DriftMonitorSettings,
) -> None:
    """Add score_psi/score_drift_detected to decision in place; monitoring-only, never raises.

    Reported separately from evaluate_drift's drift_detected/breached_features, which stay
    feature-only and keep driving retraining — a score-side failure here (BigQuery permission
    hiccup, malformed BatchPredictionJob output_info, Vertex API flakiness) must degrade to "no
    score signal this run," not take down the feature-PSI result that gates retraining.
    """
    score_spec = metadata.baseline_stats.get(CHURN_PROBABILITY_FIELD)
    if score_spec is None:
        return  # artifact predates this check (trained before the score baseline was added)

    try:
        predictions = fetch_latest_predictions(
            settings.project_id, champion.resource_name, settings.batch_predict_display_name
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
        score_threshold = max(settings.psi_threshold, psi_noise_floor(score_spec, len(predictions)))
        decision.score_psi = score_psi
        decision.score_threshold = score_threshold
        decision.score_drift_detected = score_psi > score_threshold
    except Exception:
        log.exception("Score-drift check failed; continuing with feature-PSI result only.")
