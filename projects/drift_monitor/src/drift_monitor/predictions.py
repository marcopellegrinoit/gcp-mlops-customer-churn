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

    # Confirmed against a real BatchPredictionJob's BigQuery output: for this custom
    # container, the "prediction" column holds the raw predict response body as a
    # JSON-encoded STRING, not a nested STRUCT/RECORD — hence JSON_VALUE rather than
    # dot-access (same fix applied to the orchestrator workflow's MERGE query, which hit
    # this same "Cannot access field ... on a value with type STRING" error).
    bq = bigquery.Client(project=project_id)
    return bq.query(
        f"SELECT CAST(JSON_VALUE(prediction, '$.{CHURN_PROBABILITY_FIELD}') AS FLOAT64) "
        f"AS {CHURN_PROBABILITY_FIELD} FROM `{dataset}.{table}`"
    ).to_dataframe()
