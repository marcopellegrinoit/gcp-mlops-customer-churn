"""PSI history in BigQuery: the record the persistence rule reads and writes.

A single night's PSI breach is a weak signal — it can be a one-off sampling artefact, a
partial upstream load, or a genuine shift. Retraining on it directly is what turns a
noisy detector into a nightly retrain loop. Persisting every run's per-feature PSI here
lets the monitor require a breach to repeat before acting, and makes the history
queryable ("is this new, or has it been breaching for six weeks?") in a way the
overwritten-daily GCS decision file cannot express.
"""

import logging

import pandas as pd
from data_contracts import CHURN_PROBABILITY_FIELD, DriftDecision, DriftMetricRow
from google.cloud import bigquery
from ml_common.config import get_settings

log = logging.getLogger(__name__)


def record_run(
    project_id: str,
    run_ts: pd.Timestamp,
    snapshot_date: str,
    champion_model: str,
    decision: DriftDecision,
) -> None:
    """Append one row per evaluated feature for this run."""
    rows = [
        DriftMetricRow(
            run_ts=run_ts,
            snapshot_date=snapshot_date,
            champion_model=champion_model,
            feature=feature,
            psi=psi,
            threshold=decision.feature_thresholds[feature],
            breached=feature in decision.breached_features,
            monitored=feature not in decision.unmonitored_features,
        )
        for feature, psi in decision.feature_psi.items()
    ]
    # The score check reports itself outside feature_psi (it is a model output, not an input
    # feature), but it belongs in the same time series — otherwise score drift is only ever
    # visible in the overwritten-daily decision file.
    if decision.score_psi is not None:
        rows.append(
            DriftMetricRow(
                run_ts=run_ts,
                snapshot_date=snapshot_date,
                champion_model=champion_model,
                feature=CHURN_PROBABILITY_FIELD,
                psi=decision.score_psi,
                threshold=decision.score_threshold,
                breached=decision.score_drift_detected,
                monitored=True,
            )
        )

    if not rows:
        return

    bq = bigquery.Client(project=project_id)
    table = f"{project_id}.{get_settings().drift_metrics_table}"
    errors = bq.insert_rows_json(table, [row.model_dump(mode="json") for row in rows])
    if errors:
        raise RuntimeError(f"Failed to record drift metrics: {errors}")


def prior_breach_counts(project_id: str, champion_model: str, prior_runs: int) -> dict[str, int]:
    """Return how many of the last `prior_runs` *previous* runs each feature breached in.

    Deliberately excludes the run in progress — the caller adds it in memory — so the
    persistence rule never depends on reading back a row it just streamed in.

    Scoped to one champion: promoting a new model freezes a new baseline, so PSI measured
    against the previous champion says nothing about the current one and must not carry
    over into the persistence count.
    """
    if prior_runs <= 0:
        return {}

    bq = bigquery.Client(project=project_id)
    full_table = f"{project_id}.{get_settings().drift_metrics_table}"

    query = f"""
        WITH recent_runs AS (
            SELECT DISTINCT run_ts
            FROM `{full_table}`
            WHERE champion_model = @champion_model
            ORDER BY run_ts DESC
            LIMIT @prior_runs
        )
        SELECT feature, COUNTIF(breached) AS breach_count
        FROM `{full_table}`
        WHERE champion_model = @champion_model
          AND run_ts IN (SELECT run_ts FROM recent_runs)
        GROUP BY feature
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("champion_model", "STRING", champion_model),
            bigquery.ScalarQueryParameter("prior_runs", "INT64", prior_runs),
        ]
    )
    rows = bq.query(query, job_config=job_config).result()
    return {row["feature"]: int(row["breach_count"]) for row in rows}
