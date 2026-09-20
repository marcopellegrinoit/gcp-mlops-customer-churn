"""Submit the pipeline template staged in Artifact Registry as a one-off Vertex AI PipelineJob.

For manual/dev runs against whatever container images are tagged ``latest`` — this does not
compile, build, or push anything.

Run with:
    PROJECT_ID=<project> uv run scripts/submit_dev_pipeline.py

uv reads the inline metadata below and builds an isolated environment for this script
on the fly — it has no pyproject.toml and is not part of the repo's uv workspace.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = ["google-cloud-aiplatform[pipelines]>=1.70,<2"]
# ///

import argparse
import os

from google.cloud import aiplatform


def submit_pipeline(
    project_id: str,
    region: str,
    pipeline_repo: str,
    pipeline_name: str,
    pipeline_root: str,
    artifact_gcs_prefix: str,
    batch_test_gcs_prefix: str,
    serving_image_uri: str,
    service_account: str,
    batch_predict_service_account: str,
) -> aiplatform.PipelineJob:
    """Submit the ``latest``-tagged pipeline template from Artifact Registry as a PipelineJob."""
    aiplatform.init(
        project=project_id, location=region, staging_bucket=f"gs://{project_id}-pipeline-metadata"
    )

    job = aiplatform.PipelineJob(
        display_name="churn-training-dev",
        template_path=f"https://{region}-kfp.pkg.dev/{project_id}/{pipeline_repo}/{pipeline_name}/latest",
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "artifact_gcs_prefix": artifact_gcs_prefix,
            "batch_test_gcs_prefix": batch_test_gcs_prefix,
            "serving_image_uri": serving_image_uri,
            "batch_predict_service_account": batch_predict_service_account,
        },
    )
    job.submit(service_account=service_account)
    return job


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Submit the latest staged training pipeline template for a manual/dev run"
    )
    _parser.add_argument(
        "--project-id",
        default=os.environ.get("PROJECT_ID"),
        required=os.environ.get("PROJECT_ID") is None,
    )
    _parser.add_argument("--region", default=os.environ.get("REGION", "europe-west1"))
    _parser.add_argument("--pipeline-repo", default="pipeline-templates")
    _parser.add_argument("--pipeline-name", default="churn-training-pipeline")
    _parser.add_argument(
        "--service-account",
        default=os.environ.get("PIPELINE_SERVICE_ACCOUNT"),
        help="Defaults to vertex-ai-pipeline-sa@<project-id>.iam.gserviceaccount.com",
    )
    _parser.add_argument(
        "--batch-predict-service-account",
        default=os.environ.get("BATCH_PREDICT_SERVICE_ACCOUNT"),
        help="Defaults to vertex-batch-predict-sa@<project-id>.iam.gserviceaccount.com",
    )
    _parser.add_argument(
        "--gcs-subdir",
        default="dev",
        help=(
            "Subdirectory under the training-data/pipeline-metadata buckets, "
            "to isolate dev runs from prod artifacts"
        ),
    )
    _args = _parser.parse_args()

    _service_account = (
        _args.service_account or f"vertex-ai-pipeline-sa@{_args.project_id}.iam.gserviceaccount.com"
    )
    _batch_predict_service_account = (
        _args.batch_predict_service_account
        or f"vertex-batch-predict-sa@{_args.project_id}.iam.gserviceaccount.com"
    )
    _training_data_bucket = f"gs://{_args.project_id}-training-data/{_args.gcs_subdir}"

    submit_pipeline(
        project_id=_args.project_id,
        region=_args.region,
        pipeline_repo=_args.pipeline_repo,
        pipeline_name=_args.pipeline_name,
        pipeline_root=f"gs://{_args.project_id}-pipeline-metadata/{_args.gcs_subdir}",
        artifact_gcs_prefix=f"{_training_data_bucket}/artifacts",
        batch_test_gcs_prefix=f"{_training_data_bucket}/batch-test",
        serving_image_uri=f"{_args.region}-docker.pkg.dev/{_args.project_id}/containers/serving:latest",
        service_account=_service_account,
        batch_predict_service_account=_batch_predict_service_account,
    )
