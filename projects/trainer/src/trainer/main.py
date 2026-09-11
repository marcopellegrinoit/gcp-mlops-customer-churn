"""CLI dispatcher for the heavy training pipeline stages (data_split, hpo, train).

The lightweight evaluate/register_or_reject/notify stages run in the separate
`post-training` container — see projects/post_training/src/post_training/main.py —
so this image never installs their GCP-only dependencies.

Each stage is invoked as an independent container command by the KFP pipeline.
All inter-stage values (GCS URIs and JSON blobs) travel through KFP-managed artifact
files: outputs are written to a path via _write(); inputs are read from a path via
_read(). Plain pipeline parameters (n_trials, model_display_name, etc.) are passed
as direct CLI argument values and used without file I/O.
"""

import argparse
import json
import pathlib

from ml_common.config import get_settings as get_ml_settings
from ml_common.contracts import to_json
from modeling.config import get_settings as get_modeling_settings
from obs_common.logging import configure_logging

configure_logging()


def main() -> None:
    """Dispatch to the appropriate pipeline stage based on the subcommand."""
    args = _build_parser().parse_args()
    args.func(args)


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------


def _data_split(args: argparse.Namespace) -> None:
    """Handle the data_split subcommand."""
    from trainer.data import export_snapshot  # deferred so unused stages don't load heavy deps

    train_ref, test_ref = export_snapshot(
        project_id=args.project_id,
        bq_features_table=args.bq_features_table,
        snapshot_date=args.snapshot_date or None,
    )
    _write(args.output_train_uri, to_json(train_ref))
    _write(args.output_test_uri, to_json(test_ref))


def _hpo(args: argparse.Namespace) -> None:
    """Handle the hpo subcommand."""
    from trainer.experiment import run_hpo_stage

    best_params = run_hpo_stage(
        train_uri=_read(args.train_uri),
        experiment_name=args.experiment_name,
        project_id=args.project_id,
        region=args.region,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        random_state=get_ml_settings().random_state,
    )
    _write(args.output_params, json.dumps(best_params))


def _train(args: argparse.Namespace) -> None:
    """Handle the train subcommand."""
    from trainer.experiment import run_train_stage

    params = json.loads(_read(args.params))
    model_uri = run_train_stage(
        train_uri=_read(args.train_uri),
        params=params,
        artifact_gcs_prefix=args.artifact_gcs_prefix,
        experiment_name=args.experiment_name,
        project_id=args.project_id,
        region=args.region,
        run_name=args.run_name,
        n_folds=args.n_folds,
        random_state=get_ml_settings().random_state,
    )
    _write(args.output_model_uri, model_uri)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all pipeline stage subcommands."""
    parser = argparse.ArgumentParser(description="Churn training pipeline stages")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("data_split")
    p.add_argument("--project-id", required=True)
    p.add_argument("--snapshot-date", default="")
    p.add_argument("--bq-features-table", required=True)
    p.add_argument("--output-train-uri", required=True)
    p.add_argument("--output-test-uri", required=True)
    p.set_defaults(func=_data_split)

    p = sub.add_parser("hpo")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--train-uri", required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--n-trials", type=int, default=get_modeling_settings().hpo_n_trials)
    p.add_argument("--n-folds", type=int, default=get_modeling_settings().hpo_n_folds)
    p.add_argument("--output-params", required=True)
    p.set_defaults(func=_hpo)

    p = sub.add_parser("train")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--train-uri", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--artifact-gcs-prefix", required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--run-name", default="challenger")
    p.add_argument("--n-folds", type=int, default=get_modeling_settings().hpo_n_folds)
    p.add_argument("--output-model-uri", required=True)
    p.set_defaults(func=_train)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: str, content: str) -> None:
    """Write stage output to a file; creates parent directories as needed."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def _read(path: str) -> str:
    """Read stage input from a file."""
    return pathlib.Path(path).read_text()


if __name__ == "__main__":
    main()
