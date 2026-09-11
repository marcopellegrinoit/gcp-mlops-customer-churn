"""Unit tests for the trainer CLI dispatcher (no GCP dependencies)."""

import json
import pathlib

from trainer.main import _build_parser, _read, _write

# ---------------------------------------------------------------------------
# _write / _read round-trip
# ---------------------------------------------------------------------------


def test_write_creates_file_and_read_returns_content(tmp_path):
    path = str(tmp_path / "out.txt")
    _write(path, "gs://bucket/train.parquet")
    assert _read(path) == "gs://bucket/train.parquet"


def test_write_creates_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "deep" / "out.txt")
    _write(path, "hello")
    assert pathlib.Path(path).exists()


def test_write_read_json_roundtrip(tmp_path):
    payload = {"max_depth": 5, "learning_rate": 0.05}
    path = str(tmp_path / "params.json")
    _write(path, json.dumps(payload))
    assert json.loads(_read(path)) == payload


# ---------------------------------------------------------------------------
# Parser: subcommand presence and required arguments
# ---------------------------------------------------------------------------


def test_parser_has_all_three_subcommands():
    parser = _build_parser()
    subparser_names = set()
    for action in parser._subparsers._actions:
        if hasattr(action, "_name_parser_map"):
            subparser_names = set(action._name_parser_map.keys())
    assert subparser_names == {"data_split", "hpo", "train"}


def test_data_split_requires_table(tmp_path):
    parser = _build_parser()
    out_train = str(tmp_path / "train_uri.txt")
    out_test = str(tmp_path / "test_uri.txt")
    args = parser.parse_args(
        [
            "data_split",
            "--project-id",
            "my-project",
            "--bq-features-table",
            "features.customer_features",
            "--output-train-uri",
            out_train,
            "--output-test-uri",
            out_test,
        ]
    )
    assert args.bq_features_table == "features.customer_features"
    assert args.snapshot_date == ""


def test_hpo_defaults_from_modeling_settings():
    from modeling.config import get_settings

    parser = _build_parser()
    args = parser.parse_args(
        [
            "hpo",
            "--project-id",
            "my-project",
            "--region",
            "europe-west1",
            "--experiment-name",
            "churn-predictor",
            "--train-uri",
            "gs://bucket/train.parquet",
            "--output-params",
            "/tmp/params.json",
        ]
    )
    assert args.n_trials == get_settings().hpo_n_trials
    assert args.n_folds == get_settings().hpo_n_folds


def test_train_run_name_default():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train",
            "--project-id",
            "my-project",
            "--region",
            "europe-west1",
            "--experiment-name",
            "churn-predictor",
            "--train-uri",
            "gs://bucket/train.parquet",
            "--params",
            "/tmp/params.json",
            "--artifact-gcs-prefix",
            "gs://bucket/artifacts",
            "--output-model-uri",
            "/tmp/model_uri.txt",
        ]
    )
    assert args.run_name == "challenger"
