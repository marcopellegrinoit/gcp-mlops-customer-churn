"""Unit tests for the KFP pipeline definition (no GCP credentials required)."""

import pathlib

import yaml
from training_pipeline.pipeline import build_pipeline, compile_pipeline

FAKE_TRAINER_IMAGE = "europe-docker.pkg.dev/my-project/containers/trainer:latest"
FAKE_POST_TRAINING_IMAGE = "europe-docker.pkg.dev/my-project/containers/post-training:latest"
FAKE_SERVING_IMAGE = "europe-docker.pkg.dev/my-project/containers/serving:latest"


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


def test_build_pipeline_returns_callable():
    pipeline = build_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE)
    assert callable(pipeline)


def test_build_pipeline_different_images_are_independent():
    p1 = build_pipeline("trainer-a:1", "post-training-a:1", "serving-a:1")
    p2 = build_pipeline("trainer-b:2", "post-training-b:2", "serving-b:2")
    assert p1 is not p2


# ---------------------------------------------------------------------------
# compile_pipeline
# ---------------------------------------------------------------------------


def test_compile_pipeline_creates_yaml_file(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    assert pathlib.Path(output).exists()


def test_compiled_yaml_is_valid(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    doc = yaml.safe_load(pathlib.Path(output).read_text())
    assert isinstance(doc, dict)


def test_compiled_yaml_contains_pipeline_name(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    content = pathlib.Path(output).read_text()
    assert "churn-training-pipeline" in content


def test_compiled_yaml_references_trainer_post_training_and_serving_images(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    content = pathlib.Path(output).read_text()
    assert FAKE_TRAINER_IMAGE in content
    assert FAKE_POST_TRAINING_IMAGE in content
    assert FAKE_SERVING_IMAGE in content


def test_compiled_yaml_declares_batch_predict_service_account_pipeline_parameter(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    doc = yaml.safe_load(pathlib.Path(output).read_text())
    root_params = doc["root"]["inputDefinitions"]["parameters"]
    assert "batch_predict_service_account" in root_params
    assert root_params["batch_predict_service_account"]["parameterType"] == "STRING"


def test_compiled_yaml_batch_predict_tasks_take_service_account_from_pipeline_input(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    doc = yaml.safe_load(pathlib.Path(output).read_text())
    tasks = doc["root"]["dag"]["tasks"]
    for task_name in ("batch-test-champion", "batch-test-challenger"):
        # Nested under the champion's conditional branch component for batch-test-champion,
        # so look it up by display name across all component task specs instead of by key.
        matches = [t for t in tasks.values() if t.get("taskInfo", {}).get("name") == task_name] or [
            t
            for comp in doc.get("components", {}).values()
            for t in comp.get("dag", {}).get("tasks", {}).values()
            if t.get("taskInfo", {}).get("name") == task_name
        ]
        assert matches, f"task '{task_name}' not found in compiled pipeline YAML"
        component_input = matches[0]["inputs"]["parameters"]["service_account"]
        assert (
            "componentInputParameter" in component_input or "taskOutputParameter" in component_input
        )
        # The batch-predict tasks must take the narrower batch_predict_service_account
        # input, not the pipeline's own (broader) service_account parameter.
        assert component_input.get("componentInputParameter") in (
            "pipelinechannel--batch_predict_service_account",
            "batch_predict_service_account",
        )


def test_compiled_yaml_contains_all_stages(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    content = pathlib.Path(output).read_text()
    for stage in (
        "data-split",
        "hpo",
        "train",
        "fetch-champion",
        "prep-test-batch-data",
        "build-unmanaged-container-model",
        "evaluate",
        "register-or-reject",
        "notify",
    ):
        assert stage in content, f"stage '{stage}' not found in compiled pipeline YAML"


def test_compiled_yaml_contains_champion_batch_predict_conditional_branch(tmp_path):
    output = str(tmp_path / "pipeline.yaml")
    compile_pipeline(FAKE_TRAINER_IMAGE, FAKE_POST_TRAINING_IMAGE, FAKE_SERVING_IMAGE, output)
    content = pathlib.Path(output).read_text()
    assert "has-champion" in content
    assert "google_cloud_pipeline_components" in content
    assert "oneof" in content.lower()
