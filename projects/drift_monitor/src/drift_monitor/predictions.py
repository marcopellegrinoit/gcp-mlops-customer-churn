"""Read back the champion's live production scores for score-distribution drift monitoring."""

import pandas as pd
from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform_v1.types.job_state import JobState
from ml_common.config import CHURN_PROBABILITY_FIELD


def fetch_latest_predictions(
    project_id: str, champion_resource_name: str, batch_predict_display_name: str
) -> pd.DataFrame | None:
    """Return churn_probability scores from the champion's most recent completed daily-scoring job.

    Filtering by model alone isn't enough: post_training's evaluate stage also runs batch
    prediction against the champion (to score it alongside the challenger for the promotion
    gate), so display_name disambiguates the daily production-scoring job from an evaluation
    run. Reads gca_resource directly rather than the wrapping SDK properties where available,
    since BatchPredictionJob.state re-fetches the job from the API on every access.
    """
    jobs = aiplatform.BatchPredictionJob.list(order_by="create_time desc")
    job = next(
        (
            j
            for j in jobs
            if j.gca_resource.state == JobState.JOB_STATE_SUCCEEDED
            and j.display_name == batch_predict_display_name
            and j.gca_resource.model == champion_resource_name
        ),
        None,
    )
    if job is None:
        return None

    dataset = job.output_info.bigquery_output_dataset.removeprefix("bq://")
    table = job.output_info.bigquery_output_table

    # The model's response is nested under a "prediction" struct rather than flattened to
    # top-level columns — confirmed for this same custom container's GCS-JSONL batch-predict
    # output in post_training.evaluate.read_batch_predictions ("record['prediction'][...]",
    # comment: "confirmed against actual BatchPredictionJob output"). Vertex echoes a custom
    # container's per-instance response object the same way regardless of output format, so
    # BigQuery output is expected to nest it under a `prediction` RECORD column too — still
    # worth confirming against a real job before relying on this (see docs/observability.md).
    bq = bigquery.Client(project=project_id)
    return bq.query(
        f"SELECT prediction.{CHURN_PROBABILITY_FIELD} AS {CHURN_PROBABILITY_FIELD} "
        f"FROM `{dataset}.{table}`"
    ).to_dataframe()
