"""CLI dispatcher for the post-training champion/challenger gate stages.

evaluate/register_or_reject/notify run here, in their own image, separate from
the actual prediction-serving container: none of them need optuna/shap (so they
stay out of the trainer image) and none of them serve live predictions (so they
stay out of the serving image). This container has no default long-running mode —
each KFP step invokes it as a one-shot CLI subcommand.

All inter-stage values (GCS URIs and JSON blobs) travel through KFP-managed artifact
files: outputs are written to a path via _write(); inputs are read from a path via
_read().
"""

import argparse
import json
import pathlib

from data_contracts import (
    EvaluationMetrics,
    PromotionDecision,
    RegistrationResult,
    to_json,
)
from obs_common.logging import configure_logging

configure_logging()


def main() -> None:
    """Dispatch to the appropriate pipeline stage based on the subcommand."""
    args = _build_parser().parse_args()
    args.func(args)


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------


def _fetch_champion(args: argparse.Namespace) -> None:
    """Handle the fetch_champion subcommand."""
    from post_training.fetch_champion import run_fetch_champion_stage

    champion_model_id, champion_shap, champion_threshold = run_fetch_champion_stage(
        project_id=args.project_id,
        region=args.region,
        model_display_name=args.model_display_name,
    )
    _write(args.output_champion_model_id, champion_model_id)
    _write(
        args.output_champion_shap, json.dumps(champion_shap) if champion_shap is not None else ""
    )
    _write(
        args.output_champion_threshold,
        str(champion_threshold) if champion_threshold is not None else "",
    )


def _prep_test_batch_data(args: argparse.Namespace) -> None:
    """Handle the prep_test_batch_data subcommand."""
    from post_training.batch_predict import create_batch_source_files

    gcs_test_uri = create_batch_source_files(
        test_uri=_read(args.test_uri),
        project_id=args.project_id,
        gcs_uri_prefix=args.gcs_uri_prefix,
    )
    _write(args.output_gcs_test_uri, gcs_test_uri)


def _evaluate(args: argparse.Namespace) -> None:
    """Handle the evaluate subcommand."""
    from ml_common.evaluate import flatten_for_kfp_metrics

    from post_training.evaluate import run_evaluate_stage

    metrics, decision = run_evaluate_stage(
        test_uri=_read(args.test_uri),
        challenger_uri=_read(args.challenger_uri),
        challenger_predictions_dir=args.challenger_predictions_dir,
        champion_predictions_dir=args.champion_predictions_dir,
        champion_shap_path=args.champion_shap,
        champion_threshold_path=args.champion_threshold,
        project_id=args.project_id,
    )
    _write(args.output_metrics, to_json(metrics))
    _write(args.output_decision, to_json(decision))
    _write(args.output_kfp_metrics, json.dumps(flatten_for_kfp_metrics(metrics, decision)))


def _register_or_reject(args: argparse.Namespace) -> None:
    """Handle the register_or_reject subcommand."""
    from post_training.register import register_or_reject

    result = register_or_reject(
        metrics=EvaluationMetrics.model_validate_json(_read(args.metrics)),
        decision=PromotionDecision.model_validate_json(_read(args.decision)),
        challenger_uri=_read(args.challenger_uri),
        serving_image_uri=args.serving_image_uri,
        project_id=args.project_id,
        region=args.region,
        model_display_name=args.model_display_name,
        experiment_name=args.experiment_name,
        consecutive_rejection_key=args.consecutive_rejection_key,
    )
    _write(args.output_result, to_json(result))


def _notify(args: argparse.Namespace) -> None:
    """Handle the notify subcommand."""
    from post_training.notify import notify

    notify(message=RegistrationResult.model_validate_json(_read(args.message)))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all post-training-stage subcommands."""
    parser = argparse.ArgumentParser(description="Churn post-training-stage pipeline commands")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("fetch_champion")
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--model-display-name", required=True)
    p.add_argument("--output-champion-model-id", required=True)
    p.add_argument("--output-champion-shap", required=True)
    p.add_argument("--output-champion-threshold", required=True)
    p.set_defaults(func=_fetch_champion)

    p = sub.add_parser("prep_test_batch_data")
    p.add_argument("--project-id", required=True)
    p.add_argument("--test-uri", required=True)
    p.add_argument("--gcs-uri-prefix", required=True)
    p.add_argument("--output-gcs-test-uri", required=True)
    p.set_defaults(func=_prep_test_batch_data)

    p = sub.add_parser("evaluate")
    p.add_argument("--project-id", required=True)
    p.add_argument("--test-uri", required=True)
    p.add_argument("--challenger-uri", required=True)
    p.add_argument("--challenger-predictions-dir", required=True)
    p.add_argument("--champion-predictions-dir", required=True)
    p.add_argument("--champion-shap", required=True)
    p.add_argument("--champion-threshold", required=True)
    p.add_argument("--output-metrics", required=True)
    p.add_argument("--output-decision", required=True)
    p.add_argument("--output-kfp-metrics", required=True)
    p.set_defaults(func=_evaluate)

    p = sub.add_parser("register_or_reject")
    p.add_argument("--metrics", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--challenger-uri", required=True)
    p.add_argument("--serving-image-uri", required=True)
    p.add_argument("--project-id", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--model-display-name", required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--consecutive-rejection-key", required=True)
    p.add_argument("--output-result", required=True)
    p.set_defaults(func=_register_or_reject)

    p = sub.add_parser("notify")
    p.add_argument("--message", required=True)
    p.set_defaults(func=_notify)

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
