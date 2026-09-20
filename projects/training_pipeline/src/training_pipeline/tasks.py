"""KFP v2 component (task) definitions for the churn training pipeline.

Components are nested inside build_tasks() so the trainer/post-training container
image URIs are captured in each component's closure and resolved at compile time,
not baked in as constants.
"""

import sys
import types

from google_cloud_pipeline_components.types.artifact_types import UnmanagedContainerModel
from kfp import dsl
from kfp.dsl import Artifact, Input, Metrics, Output
from modeling.config import get_settings as get_modeling_settings

_PYTHON_BASE_IMAGE = f"python:{sys.version_info.major}.{sys.version_info.minor}-slim"
_MODELING = get_modeling_settings()


def build_tasks(trainer_image: str, post_training_image: str) -> types.SimpleNamespace:
    """Return the pipeline's component callables bound to the given container images.

    data_split/hpo/train run on trainer_image (needs optuna/shap for HPO and training).
    fetch_champion_name/prep_test_batch_data/evaluate/register_or_reject/notify run on
    post_training_image, which never loads or scores a model itself; it only runs the
    gate math over predictions it's handed, so there is no separate in-process scoring
    pass that could silently drift from production.
    """

    @dsl.container_component
    def data_split(
        project_id: str,
        bq_features_table: str,
        train_uri: Output[Artifact],
        test_uri: Output[Artifact],
        snapshot_date: str = "",
    ):
        """Freeze a stratified train/test split of the feature snapshot into ml.split_assignments.

        train_uri/test_uri carry small JSON references (snapshot_date + split) into that table,
        resolved by trainer.data.read_split — not GCS URIs. See docs/ml-infrastructure.md.
        """
        return dsl.ContainerSpec(
            image=trainer_image,
            command=["python", "-m", "trainer.main", "data_split"],
            args=[
                "--project-id",
                project_id,
                "--bq-features-table",
                bq_features_table,
                "--snapshot-date",
                snapshot_date,
                "--output-train-uri",
                train_uri.path,
                "--output-test-uri",
                test_uri.path,
            ],
        )

    @dsl.container_component
    def hpo(
        project_id: str,
        region: str,
        experiment_name: str,
        train_uri: Input[Artifact],
        params: Output[Artifact],
        n_trials: int = _MODELING.hpo_n_trials,
        n_folds: int = _MODELING.hpo_n_folds,
    ):
        """Run Bayesian HPO search and write the best hyperparameter dict to params."""
        return dsl.ContainerSpec(
            image=trainer_image,
            command=["python", "-m", "trainer.main", "hpo"],
            args=[
                "--project-id",
                project_id,
                "--region",
                region,
                "--experiment-name",
                experiment_name,
                "--train-uri",
                train_uri.path,
                "--n-trials",
                n_trials,
                "--n-folds",
                n_folds,
                "--output-params",
                params.path,
            ],
        )

    @dsl.container_component
    def train(
        project_id: str,
        region: str,
        experiment_name: str,
        train_uri: Input[Artifact],
        params: Input[Artifact],
        artifact_gcs_prefix: str,
        model_uri: Output[Artifact],
        run_name: str = "challenger",
        n_folds: int = _MODELING.hpo_n_folds,
    ):
        """Train on the full dataset with the best HPO params and upload the artifact to GCS.

        n_folds also drives the out-of-fold CV pass used to select this model's decision
        threshold (see modeling.train.select_threshold_via_cv) — kept independent of evaluate's
        held-out test set so the gate's reported F1 isn't biased by picking the threshold there.
        """
        return dsl.ContainerSpec(
            image=trainer_image,
            command=["python", "-m", "trainer.main", "train"],
            args=[
                "--project-id",
                project_id,
                "--region",
                region,
                "--experiment-name",
                experiment_name,
                "--train-uri",
                train_uri.path,
                "--params",
                params.path,
                "--artifact-gcs-prefix",
                artifact_gcs_prefix,
                "--run-name",
                run_name,
                "--n-folds",
                n_folds,
                "--output-model-uri",
                model_uri.path,
            ],
        )

    @dsl.container_component
    def fetch_champion_name(
        project_id: str,
        region: str,
        model_display_name: str,
        champion_model_id: dsl.OutputPath(str),
        champion_shap: Output[Artifact],
        champion_threshold: Output[Artifact],
    ):
        """Look up the current champion model, if any, so it can be batch-scored downstream."""
        return dsl.ContainerSpec(
            image=post_training_image,
            command=["python", "-m", "post_training.main", "fetch_champion"],
            args=[
                "--project-id",
                project_id,
                "--region",
                region,
                "--model-display-name",
                model_display_name,
                "--output-champion-model-id",
                champion_model_id,
                "--output-champion-shap",
                champion_shap.path,
                "--output-champion-threshold",
                champion_threshold.path,
            ],
        )

    @dsl.container_component
    def prep_test_batch_data(
        project_id: str,
        test_uri: Input[Artifact],
        batch_test_gcs_prefix: str,
        gcs_test_uri: dsl.OutputPath(str),
    ):
        """Materialize the test split into scratch BigQuery, then export it as GCS JSONL.

        ModelBatchPredictOp requires instances_format and predictions_format to both be "bigquery"
        or both be non-bigquery — this pipeline keeps Batch Prediction's output on GCS (see
        docs/ml-infrastructure.md), so the source must be GCS JSONL too. This writes exactly this
        run's test rows into the self-expiring scratch.test_batch_<uuid> table first (BigQuery can
        only extract from a physical table, not an arbitrary filtered query), then extracts that
        table to GCS under batch_test_gcs_prefix.
        """
        return dsl.ContainerSpec(
            image=post_training_image,
            command=["python", "-m", "post_training.main", "prep_test_batch_data"],
            args=[
                "--project-id",
                project_id,
                "--test-uri",
                test_uri.path,
                "--gcs-uri-prefix",
                batch_test_gcs_prefix,
                "--output-gcs-test-uri",
                gcs_test_uri,
            ],
        )

    @dsl.component(base_image=_PYTHON_BASE_IMAGE)
    def no_champion_placeholder() -> str:
        """Stand in for the champion predictions directory when no champion exists yet."""
        return ""

    @dsl.component(base_image=_PYTHON_BASE_IMAGE)
    def extract_uri(artifact: Input[Artifact]) -> str:
        """Return the GCS URI backing a KFP artifact (only resolvable inside component code)."""
        return artifact.uri

    @dsl.component(base_image=_PYTHON_BASE_IMAGE)
    def read_artifact_text(artifact: Input[Artifact]) -> str:
        """Return the text KFP synced into an artifact's local file.

        Unlike extract_uri, this is for artifacts whose backing file content (not artifact.uri)
        is the real payload — e.g. train's model_uri, whose .path holds the trainer's own
        manually-uploaded gs:// artifact directory rather than KFP's system-assigned location.
        """
        import pathlib

        return pathlib.Path(artifact.path).read_text()

    @dsl.component(
        base_image=_PYTHON_BASE_IMAGE,
        packages_to_install=["google-cloud-storage"],
    )
    def read_challenger_threshold(challenger_artifact_uri: str) -> str:
        """Read the challenger's CV-selected decision threshold out of its metadata.json."""
        import json

        from google.cloud import storage

        bucket_name, blob_name = challenger_artifact_uri.removeprefix("gs://").split("/", 1)
        blob = storage.Client().bucket(bucket_name).blob(f"{blob_name}/metadata.json")
        return str(json.loads(blob.download_as_text())["threshold"])

    @dsl.component(
        base_image=_PYTHON_BASE_IMAGE,
        packages_to_install=["google-cloud-pipeline-components>=2.10,<3"],
    )
    def build_unmanaged_container_model(
        challenger_artifact_uri: str,
        serving_image: str,
        threshold: str,
        project_id: str,
        unmanaged_model: Output[UnmanagedContainerModel],
    ):
        """Describe the challenger as an UnmanagedContainerModel so it can be Batch-Predicted.

        The challenger isn't registered in Model Registry yet (it may still be rejected), so
        it has no Model resource for ModelBatchPredictOp to target. An UnmanagedContainerModel
        carries the same information (artifact + serving container spec) without requiring
        registration — Vertex AI boots serving_image exactly as it would for a registered model.
        threshold is the same CV-selected value the container would get if this challenger were
        promoted (see read_challenger_threshold), so this batch-eval run boots it identically to
        production even though evaluate itself only reads churn_probability from the result.
        PROJECT_ID is passed explicitly because the batch-predict service account is attached
        via short-lived credential exchange, which doesn't reliably expose project info to
        google-cloud-storage's ADC-based project detection inside the container.
        unmanaged_model.uri must be set to challenger_artifact_uri explicitly: Vertex AI's
        launcher (google_cloud_pipeline_components.container.v1.batch_prediction_job.remote_runner
        .insert_artifact_into_payload) always overwrites unmanagedContainerModel.artifactUri with
        this artifact's own KFP-assigned .uri, and Vertex AI then derives the container's
        AIP_STORAGE_URI from that artifactUri — silently ignoring the AIP_STORAGE_URI entry below
        if .uri is left at its default (a pipeline-metadata staging path with no model files, that
        the batch-predict SA can't read anyway).
        """
        unmanaged_model.uri = challenger_artifact_uri
        unmanaged_model.metadata["containerSpec"] = {
            "imageUri": serving_image,
            "predictRoute": "/predict",
            "healthRoute": "/health",
            "ports": [{"containerPort": 8080}],
            "env": [
                {"name": "AIP_STORAGE_URI", "value": challenger_artifact_uri},
                {"name": "THRESHOLD", "value": threshold},
                {"name": "PROJECT_ID", "value": project_id},
            ],
        }
        unmanaged_model.metadata["predictSchemata"] = {}

    @dsl.container_component
    def evaluate(
        project_id: str,
        test_uri: Input[Artifact],
        challenger_uri: Input[Artifact],
        challenger_predictions_dir: str,
        champion_predictions_dir: str,
        champion_shap: Input[Artifact],
        champion_threshold: Input[Artifact],
        metrics: Output[Artifact],
        decision: Output[Artifact],
        kfp_metrics: Output[Metrics],
    ):
        """Score the champion/challenger gate and split the result into facts vs. verdict.

        metrics carries the scored facts (challenger/champion metrics, threshold,
        SHAP rank correlation) with no promotion verdict, so it's inspectable on its
        own. decision carries just the promote bool and the deltas that drove it.
        kfp_metrics duplicates the scalar values from both into the pipeline run's
        metrics tab — it's display-only and nothing downstream reads it.
        """
        return dsl.ContainerSpec(
            image=post_training_image,
            command=["python", "-m", "post_training.main", "evaluate"],
            args=[
                "--project-id",
                project_id,
                "--test-uri",
                test_uri.path,
                "--challenger-uri",
                challenger_uri.path,
                "--challenger-predictions-dir",
                challenger_predictions_dir,
                "--champion-predictions-dir",
                champion_predictions_dir,
                "--champion-shap",
                champion_shap.path,
                "--champion-threshold",
                champion_threshold.path,
                "--output-metrics",
                metrics.path,
                "--output-decision",
                decision.path,
                "--output-kfp-metrics",
                kfp_metrics.path,
            ],
        )

    @dsl.container_component
    def register_or_reject(
        challenger_uri: Input[Artifact],
        metrics: Input[Artifact],
        decision: Input[Artifact],
        serving_image_uri: str,
        project_id: str,
        region: str,
        model_display_name: str,
        experiment_name: str,
        consecutive_rejection_key: str,
        result: Output[Artifact],
    ):
        """Promote the challenger to Vertex AI Model Registry, or log a rejection to Experiments."""
        return dsl.ContainerSpec(
            image=post_training_image,
            command=["python", "-m", "post_training.main", "register_or_reject"],
            args=[
                "--metrics",
                metrics.path,
                "--decision",
                decision.path,
                "--challenger-uri",
                challenger_uri.path,
                "--serving-image-uri",
                serving_image_uri,
                "--project-id",
                project_id,
                "--region",
                region,
                "--model-display-name",
                model_display_name,
                "--experiment-name",
                experiment_name,
                "--consecutive-rejection-key",
                consecutive_rejection_key,
                "--output-result",
                result.path,
            ],
        )

    @dsl.container_component
    def notify(
        message: Input[Artifact],
    ):
        """Log the pipeline outcome (promotion or rejection) as the run's terminal event.

        A logging-only sink, not a real notification channel — see post_training.notify.
        """
        return dsl.ContainerSpec(
            image=post_training_image,
            command=["python", "-m", "post_training.main", "notify"],
            args=[
                "--message",
                message.path,
            ],
        )

    return types.SimpleNamespace(
        data_split=data_split,
        hpo=hpo,
        train=train,
        fetch_champion_name=fetch_champion_name,
        prep_test_batch_data=prep_test_batch_data,
        no_champion_placeholder=no_champion_placeholder,
        extract_uri=extract_uri,
        read_artifact_text=read_artifact_text,
        read_challenger_threshold=read_challenger_threshold,
        build_unmanaged_container_model=build_unmanaged_container_model,
        evaluate=evaluate,
        register_or_reject=register_or_reject,
        notify=notify,
    )
