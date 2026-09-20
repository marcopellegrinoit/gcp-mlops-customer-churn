r"""KFP v2 pipeline definition for the churn training pipeline.

Each stage runs the same trainer container image with a different CLI subcommand.
All inter-stage values (GCS URIs, JSON blobs) are passed via KFP-managed artifact files:
  - The producing stage writes to artifact.path (a local temp file provided by KFP).
  - The consuming stage receives a local temp path and calls _read() to get the value.

Compile from CI/CD (after the trainer image is built and pushed):
    python -m training_pipeline.pipeline \
        --trainer-image <region>-docker.pkg.dev/<project>/containers/trainer \
        --post-training-image <region>-docker.pkg.dev/<project>/containers/post-training \
        --serving-image <region>-docker.pkg.dev/<project>/containers/serving \
        --output pipeline.yaml
"""

import argparse

from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model import ModelGetOp
from kfp import compiler, dsl
from ml_common.config import get_settings as get_ml_settings
from modeling.config import get_settings as get_modeling_settings

from training_pipeline import resources
from training_pipeline.settings_env import settings_env
from training_pipeline.tasks import build_tasks


def build_pipeline(trainer_image: str, post_training_image: str, serving_image: str):
    """Return a configured KFP pipeline callable bound to the given container images.

    The challenger and champion are both scored via Vertex AI Batch Prediction against
    the exact serving_image container that will run in production — the challenger via an
    UnmanagedContainerModel (it isn't registered yet), the champion via its own registered
    Model.
    """
    tasks = build_tasks(trainer_image, post_training_image)
    ml_settings = get_ml_settings()
    modeling_settings = get_modeling_settings()
    # Every stage reads ml_common.config; only the trainer stages read modeling.config.
    shared_env = settings_env(ml_settings)
    trainer_env = shared_env | settings_env(modeling_settings)

    def pin(task, env: dict[str, str]):
        """Freeze the settings this pipeline was compiled with onto a task's container."""
        for name, value in env.items():
            task.set_env_variable(name, value)
        return task

    @dsl.pipeline(name="churn-training-pipeline")
    def churn_training_pipeline(
        project_id: str,
        artifact_gcs_prefix: str,
        batch_test_gcs_prefix: str,
        serving_image_uri: str,
        batch_predict_service_account: str,
        bq_features_table: str = ml_settings.bq_features_table,
        model_display_name: str = ml_settings.model_display_name,
        experiment_name: str = ml_settings.model_display_name,
        consecutive_rejection_key: str = ml_settings.consecutive_rejection_key,
        snapshot_date: str = "",
        region: str = "europe-west1",
        n_trials: int = modeling_settings.hpo_n_trials,
        n_folds: int = modeling_settings.hpo_n_folds,
    ):
        """Pipeline: export → HPO → train → batch-predict → evaluate → register → notify."""
        # Freeze a stratified train/test split of the BigQuery features table into
        # ml.split_assignments. Caching disabled: the whole point of a drift-triggered retrain
        # is to re-split whatever the feature table currently holds, which KFP's cache key
        # (bq_features_table's literal string, not its contents) can't see has changed.
        data_split_task = pin(
            tasks.data_split(
                project_id=project_id,
                bq_features_table=bq_features_table,
                snapshot_date=snapshot_date,
            )
            .set_display_name("data-split")
            .set_caching_options(False),
            trainer_env,
        )

        # Search for the best hyperparameters via cross-validated trials.
        hpo_task = pin(
            tasks.hpo(
                project_id=project_id,
                region=region,
                experiment_name=experiment_name,
                train_uri=data_split_task.outputs["train_uri"],
                n_trials=n_trials,
                n_folds=n_folds,
            )
            .set_cpu_limit(resources.HPO_CPU_LIMIT)
            .set_memory_limit(resources.HPO_MEMORY_LIMIT)
            .set_display_name("hpo-challenger"),
            trainer_env,
        )

        # Train the challenger model on the full training set using the tuned params.
        train_task = pin(
            tasks.train(
                project_id=project_id,
                region=region,
                experiment_name=experiment_name,
                train_uri=data_split_task.outputs["train_uri"],
                params=hpo_task.outputs["params"],
                artifact_gcs_prefix=artifact_gcs_prefix,
                n_folds=n_folds,
            )
            .set_cpu_limit(resources.TRAIN_CPU_LIMIT)
            .set_memory_limit(resources.TRAIN_MEMORY_LIMIT)
            .set_display_name("train-challenger"),
            trainer_env,
        )

        # Look up the current champion (if any). Caching disabled: a cache hit here would
        # return whichever model was champion the last time this ran, even if a different
        # pipeline run has since promoted a new one — silently gating the challenger against
        # a stale champion instead of the real current one.
        fetch_champion_task = pin(
            tasks.fetch_champion_name(
                project_id=project_id,
                region=region,
                model_display_name=model_display_name,
            )
            .set_display_name("fetch-champion-name")
            .set_caching_options(False),
            shared_env,
        )

        # Materialize the test set and export it as GCS JSONL for batch scoring of both challenger
        # and champion — see tasks.prep_test_batch_data for why a physical materialization (and GCS,
        # not BigQuery, as the Batch Prediction source) is needed.
        prep_test_batch_data_task = pin(
            tasks.prep_test_batch_data(
                project_id=project_id,
                test_uri=data_split_task.outputs["test_uri"],
                batch_test_gcs_prefix=batch_test_gcs_prefix,
            ).set_display_name("prep-test-batch-data"),
            shared_env,
        )

        # Score the champion through its own registered serving container via Vertex AI Batch
        # Prediction, rather than loading it in-process: each registered model permanently
        # carries the serving image it was registered with, so an old champion may depend on
        # different library versions than this pipeline's evaluator image. Skipped entirely
        # on the very first pipeline run, when no champion exists yet. Both batch prediction
        # jobs below run as batch_predict_service_account — a narrower SA scoped to just the
        # training-data bucket the model artifacts and batch-test instances live in, rather
        # than the pipeline's own broader SA or the system-generated default (which has no
        # bucket access at all).
        with dsl.If(
            condition=fetch_champion_task.outputs["champion_model_id"] != "",
            name="has-champion",
        ):
            # Fetch the champion as a Vertex Model artifact (required input type for
            # ModelBatchPredictOp). model_name must be the bare model ID — ModelGetOp builds the
            # full resource name itself from project/location/model_name, so a full resource name
            # here yields a doubly-prefixed name and a 400 INVALID_ARGUMENT from GetModel.
            model_get_task = ModelGetOp(
                project=project_id,
                model_name=fetch_champion_task.outputs["champion_model_id"],
                location=region,
            ).set_display_name("get-champion-model")
            # Run the champion's own registered serving container against the test instances.
            batch_predict_task = (
                ModelBatchPredictOp(
                    job_display_name="champion-test",
                    model=model_get_task.outputs["model"],
                    location=region,
                    instances_format="jsonl",
                    gcs_source_uris=[prep_test_batch_data_task.outputs["gcs_test_uri"]],
                    key_field="customer_id",
                    predictions_format="jsonl",
                    gcs_destination_output_uri_prefix=batch_test_gcs_prefix,
                    machine_type=resources.BATCH_PREDICT_MACHINE_TYPE,
                    service_account=batch_predict_service_account,
                )
                .set_display_name("batch-test-champion")
                .set_retry(
                    resources.BATCH_PREDICT_RETRY_COUNT,
                    backoff_duration=resources.BATCH_PREDICT_RETRY_BACKOFF_DURATION,
                )
            )
            # Pull the batch job's output directory out as a plain string for the evaluate step.
            predictions_dir_if = (
                tasks.extract_uri(artifact=batch_predict_task.outputs["gcs_output_directory"])
                .set_display_name("extract-champion-test-predictions-uri")
                .output
            )
        with dsl.Else(name="no-champion"):
            # No champion yet (first-ever pipeline run) — nothing to score.
            predictions_dir_else = (
                tasks.no_champion_placeholder().set_display_name("no-champion-placeholder").output
            )

        # Resolve to whichever branch actually ran.
        champion_predictions_dir = dsl.OneOf(predictions_dir_if, predictions_dir_else)

        # Score the challenger through the same Batch Prediction mechanism as the champion above,
        # just against an UnmanagedContainerModel instead of a registered Model (the challenger
        # isn't registered yet, so it has no Model resource to point ModelBatchPredictOp at).
        # Runs in parallel with the champion branch; neither depends on the other.
        challenger_artifact_uri = (
            tasks.read_artifact_text(artifact=train_task.outputs["model_uri"])
            .set_display_name("extract-challenger-artifact-uri")
            .output
        )
        challenger_threshold = (
            tasks.read_challenger_threshold(challenger_artifact_uri=challenger_artifact_uri)
            .set_display_name("read-challenger-threshold")
            .output
        )
        build_unmanaged_model_task = tasks.build_unmanaged_container_model(
            challenger_artifact_uri=challenger_artifact_uri,
            serving_image=serving_image,
            threshold=challenger_threshold,
            project_id=project_id,
        ).set_display_name("build-challenger-serving")
        challenger_batch_predict_task = (
            ModelBatchPredictOp(
                job_display_name="challenger-test",
                unmanaged_container_model=build_unmanaged_model_task.outputs["unmanaged_model"],
                location=region,
                instances_format="jsonl",
                gcs_source_uris=[prep_test_batch_data_task.outputs["gcs_test_uri"]],
                key_field="customer_id",
                predictions_format="jsonl",
                gcs_destination_output_uri_prefix=batch_test_gcs_prefix,
                machine_type=resources.BATCH_PREDICT_MACHINE_TYPE,
                service_account=batch_predict_service_account,
            )
            .set_display_name("batch-test-challenger")
            .set_retry(
                resources.BATCH_PREDICT_RETRY_COUNT,
                backoff_duration=resources.BATCH_PREDICT_RETRY_BACKOFF_DURATION,
            )
        )
        challenger_predictions_dir = (
            tasks.extract_uri(
                artifact=challenger_batch_predict_task.outputs["gcs_output_directory"]
            )
            .set_display_name("extract-challenger-predictions-uri")
            .output
        )

        # Evaluate the challenger against the current champion using each one's own serving-image
        # predictions — no separate in-process scoring pass that could drift from production.
        # pin(): the promotion gate's thresholds are read from ML_PR_AUC_MIN_DELTA /
        # ML_F1_MIN_DELTA inside this container, so the gate a run was judged by is fixed at
        # compile time and visible in the pipeline spec rather than baked into the image.
        evaluate_task = pin(
            tasks.evaluate(
                project_id=project_id,
                test_uri=data_split_task.outputs["test_uri"],
                challenger_uri=train_task.outputs["model_uri"],
                challenger_predictions_dir=challenger_predictions_dir,
                champion_predictions_dir=champion_predictions_dir,
                champion_shap=fetch_champion_task.outputs["champion_shap"],
                champion_threshold=fetch_champion_task.outputs["champion_threshold"],
            ).set_display_name("evaluate-challenger"),
            shared_env,
        )

        # Promote the challenger to champion or reject it based on the evaluation decision.
        register_task = pin(
            tasks.register_or_reject(
                metrics=evaluate_task.outputs["metrics"],
                decision=evaluate_task.outputs["decision"],
                challenger_uri=train_task.outputs["model_uri"],
                serving_image_uri=serving_image_uri,
                project_id=project_id,
                region=region,
                model_display_name=model_display_name,
                experiment_name=experiment_name,
                consecutive_rejection_key=consecutive_rejection_key,
            ).set_display_name("register-or-reject"),
            shared_env,
        )

        # Report the final outcome. Logging-only: the DAG keeps a single terminal step where a
        # real deployment would hand the event off to a consumer, but nothing is published.
        pin(
            tasks.notify(message=register_task.outputs["result"]).set_display_name("notify"),
            shared_env,
        )

    return churn_training_pipeline


def compile_pipeline(
    trainer_image: str, post_training_image: str, serving_image: str, output_path: str
) -> None:
    """Compile the training pipeline to a YAML package file for Vertex AI Pipelines."""
    pipeline = build_pipeline(trainer_image, post_training_image, serving_image)
    compiler.Compiler().compile(pipeline, output_path)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Compile the churn training pipeline to YAML")
    _parser.add_argument(
        "--trainer-image", required=True, help="Docker image URI for the trainer container"
    )
    _parser.add_argument(
        "--post-training-image",
        required=True,
        help="Docker image URI for the post-training container",
    )
    _parser.add_argument(
        "--serving-image", required=True, help="Docker image URI for the serving container"
    )
    _parser.add_argument(
        "--output", default="pipeline.yaml", help="Output path for the compiled pipeline YAML"
    )
    _args = _parser.parse_args()
    compile_pipeline(
        _args.trainer_image, _args.post_training_image, _args.serving_image, _args.output
    )
