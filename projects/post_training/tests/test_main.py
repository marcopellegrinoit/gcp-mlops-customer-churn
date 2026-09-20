"""Unit tests for the post-training CLI dispatcher (no GCP dependencies)."""

import json
import pathlib

import pytest
from post_training.main import _build_parser, _read, _write


def test_write_creates_file_and_read_returns_content(tmp_path):
    path = str(tmp_path / "out.txt")
    _write(path, "gs://bucket/decision.json")
    assert _read(path) == "gs://bucket/decision.json"


def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "deep" / "out.txt")
    _write(path, "hello")
    assert pathlib.Path(path).exists()


def test_write_read_json_roundtrip(tmp_path):
    payload = {"promote": True}
    path = str(tmp_path / "decision.json")
    _write(path, json.dumps(payload))
    assert json.loads(_read(path)) == payload


def test_parser_has_all_subcommands():
    parser = _build_parser()
    subparser_names = set()
    for action in parser._subparsers._actions:
        if hasattr(action, "_name_parser_map"):
            subparser_names = set(action._name_parser_map.keys())
    assert subparser_names == {
        "fetch_champion",
        "prep_test_batch_data",
        "evaluate",
        "register_or_reject",
        "notify",
    }


def test_evaluate_requires_champion_predictions_dir():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "--project-id",
                "proj",
                "--test-uri",
                "gs://bucket/test.parquet",
                "--challenger-uri",
                "gs://bucket/artifacts/abc",
                "--challenger-predictions-dir",
                "gs://bucket/challenger-output",
                # --champion-predictions-dir deliberately omitted
                "--champion-shap",
                "/tmp/champion_shap.json",
                "--output-metrics",
                "/tmp/metrics.json",
                "--output-decision",
                "/tmp/decision.json",
            ]
        )


def test_evaluate_parses_all_required_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "--project-id",
            "proj",
            "--test-uri",
            "gs://bucket/test.parquet",
            "--challenger-uri",
            "gs://bucket/artifacts/abc",
            "--challenger-predictions-dir",
            "gs://bucket/challenger-output",
            "--champion-predictions-dir",
            "gs://bucket/batch-test/output",
            "--champion-shap",
            "/tmp/champion_shap.json",
            "--champion-threshold",
            "/tmp/champion_threshold.txt",
            "--output-metrics",
            "/tmp/metrics.json",
            "--output-decision",
            "/tmp/decision.json",
            "--output-kfp-metrics",
            "/tmp/kfp_metrics.json",
        ]
    )
    assert args.champion_predictions_dir == "gs://bucket/batch-test/output"
    assert args.challenger_predictions_dir == "gs://bucket/challenger-output"
    assert args.project_id == "proj"


def test_fetch_champion_requires_model_display_name():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fetch_champion",
                "--project-id",
                "proj",
                "--region",
                "europe-west1",
                "--output-champion-model-id",
                "/tmp/model_id.txt",
                "--output-champion-shap",
                "/tmp/shap.json",
            ]
        )


def test_fetch_champion_requires_project_id():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fetch_champion",
                "--region",
                "europe-west1",
                "--model-display-name",
                "churn-predictor",
                "--output-champion-model-id",
                "/tmp/model_id.txt",
                "--output-champion-shap",
                "/tmp/shap.json",
                "--output-champion-threshold",
                "/tmp/threshold.txt",
            ]
        )


def test_fetch_champion_parses_all_required_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "fetch_champion",
            "--project-id",
            "proj",
            "--region",
            "europe-west1",
            "--model-display-name",
            "churn-predictor",
            "--output-champion-model-id",
            "/tmp/model_id.txt",
            "--output-champion-shap",
            "/tmp/shap.json",
            "--output-champion-threshold",
            "/tmp/threshold.txt",
        ]
    )
    assert args.project_id == "proj"
    assert args.region == "europe-west1"


def test_prep_test_batch_data_requires_project_id():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prep_test_batch_data",
                # --project-id deliberately omitted
                "--test-uri",
                "gs://bucket/test.parquet",
                "--gcs-uri-prefix",
                "gs://bucket/batch-test",
                "--output-gcs-test-uri",
                "/tmp/gcs_test_uri.txt",
            ]
        )


def test_prep_test_batch_data_parses_all_required_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "prep_test_batch_data",
            "--project-id",
            "proj",
            "--test-uri",
            "gs://bucket/test.parquet",
            "--gcs-uri-prefix",
            "gs://bucket/batch-test",
            "--output-gcs-test-uri",
            "/tmp/gcs_test_uri.txt",
        ]
    )
    assert args.project_id == "proj"
    assert args.gcs_uri_prefix == "gs://bucket/batch-test"


def test_register_or_reject_parses_all_required_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "register_or_reject",
            "--metrics",
            "/tmp/metrics.json",
            "--decision",
            "/tmp/decision.json",
            "--challenger-uri",
            "gs://bucket/artifacts/abc",
            "--serving-image-uri",
            "europe-docker.pkg.dev/proj/repo/serving:latest",
            "--project-id",
            "proj",
            "--region",
            "europe-west1",
            "--model-display-name",
            "churn-predictor",
            "--experiment-name",
            "churn-predictor-experiment",
            "--consecutive-rejection-key",
            "consecutive_rejections",
            "--output-result",
            "/tmp/result.json",
        ]
    )
    assert args.model_display_name == "churn-predictor"
    assert args.consecutive_rejection_key == "consecutive_rejections"
    assert args.project_id == "proj"
    assert args.region == "europe-west1"


def test_register_or_reject_requires_project_id():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "register_or_reject",
                "--metrics",
                "/tmp/metrics.json",
                "--decision",
                "/tmp/decision.json",
                "--challenger-uri",
                "gs://bucket/artifacts/abc",
                "--serving-image-uri",
                "europe-docker.pkg.dev/proj/repo/serving:latest",
                # --project-id deliberately omitted
                "--region",
                "europe-west1",
                "--model-display-name",
                "churn-predictor",
                "--experiment-name",
                "churn-predictor-experiment",
                "--consecutive-rejection-key",
                "consecutive_rejections",
                "--output-result",
                "/tmp/result.json",
            ]
        )


def test_notify_requires_message():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["notify"])


def test_notify_parses_all_required_args():
    parser = _build_parser()
    args = parser.parse_args(["notify", "--message", "/tmp/message.json"])
    assert args.message == "/tmp/message.json"


def test_notify_takes_no_project_id():
    # The stage only logs the outcome — it talks to no GCP service, so it takes no project.
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["notify", "--message", "/tmp/message.json", "--project-id", "proj"])
