"""Recompute the current champion's drift baseline in place, without retraining it.

Baselines frozen before per-bucket proportions were stored carry only decile bin edges.
Evaluating one means re-assuming those buckets are equal-frequency, which they are not
whenever tied quantiles collapsed — the defect that made the drift monitor breach against
its own training data and trigger a retrain every night. ``ml_common.drift`` now reports
such baselines as unmonitored rather than acting on them, so an un-migrated champion is
simply not watched until this script runs.

Waiting for the next promotion is not a migration path: the promotion gate requires the
challenger to beat the champion on PR-AUC *and* F1, and a challenger trained on an
undrifted distribution generally does not, so the stale baseline can persist indefinitely.

``ml.split_assignments`` is a permanent, never-expiring record of the exact rows each
model trained on, so the correct baseline is recoverable exactly — no approximation, and
no retraining.

Run with:
    PROJECT_ID=<project> uv run scripts/rebuild_champion_baseline.py --snapshot-date 2026-09-01

uv reads the inline metadata below and builds an isolated environment for this script
on the fly — it has no pyproject.toml and is not part of the repo's uv workspace.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "google-cloud-aiplatform>=1.70,<2",
#   "google-cloud-bigquery>=3,<4",
#   "google-cloud-storage>=3,<4",
#   "data-contracts",
#   "ml-common",
#   "pandas>=2",
#   "xgboost>=2,<4",
#   "db-dtypes>=1.7.1,<2",
# ]
#
# [tool.uv.sources]
# data-contracts = { path = "../projects/data_contracts", editable = true }
# ml-common = { path = "../projects/ml_common", editable = true }
# ///

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb
from data_contracts import (
    CHURN_PROBABILITY_FIELD,
    BaselineSpec,
    ModelMetadata,
    to_json,
)
from google.cloud import aiplatform, bigquery, storage
from ml_common.config import get_settings
from ml_common.drift import compute_baseline_stats
from ml_common.preprocess import prepare_features

SETTINGS = get_settings()


def fetch_champion(model_display_name: str):
    """Return the Vertex AI Model labelled role=champion, or None."""
    for model in aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"', order_by="create_time desc"
    ):
        if (model.labels or {}).get("role") == "champion":
            return model
    return None


def read_training_split(project_id: str, snapshot_date: str) -> pd.DataFrame:
    """Load the exact rows this champion trained on from the frozen split record."""
    bq = bigquery.Client(project=project_id)
    return (
        bq.query(
            f"""
            SELECT * EXCEPT (split, assigned_at)
            FROM `{project_id}.{SETTINGS.split_assignments_table}`
            WHERE snapshot_date = @snapshot_date AND split = 'train'
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("snapshot_date", "DATE", snapshot_date)
                ]
            ),
        )
        .result()
        .to_dataframe(create_bqstorage_client=True, date_dtype=None, time_dtype=None)
    )


def resolve_snapshot_date(project_id: str, champion, metadata: ModelMetadata) -> str:
    """Determine which split_assignments partition the champion actually trained on.

    Models trained after training_snapshot_date was added to metadata.json say so directly.
    For older ones it is inferred as the most recent partition at or before the model's
    registration date — every training run writes a partition, rejected challengers
    included, so the newest partition overall is often some later run's, not this one's.
    """
    if metadata.training_snapshot_date:
        return metadata.training_snapshot_date

    bq = bigquery.Client(project=project_id)
    row = next(
        bq.query(
            f"""
            SELECT MAX(snapshot_date) AS d
            FROM `{project_id}.{SETTINGS.split_assignments_table}`
            WHERE snapshot_date <= DATE(@registered_at)
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "registered_at", "TIMESTAMP", champion.create_time
                    )
                ]
            ),
        ).result()
    )
    if row["d"] is None:
        raise SystemExit(
            "Could not infer the champion's training snapshot; pass --snapshot-date explicitly."
        )
    print(
        "  (inferred from the model's registration date — metadata has no training_snapshot_date)"
    )
    return str(row["d"])


def read_metadata(artifact_uri: str) -> ModelMetadata:
    """Download and validate an artifact directory's metadata.json."""
    bucket_name, _, prefix = artifact_uri.removeprefix("gs://").partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/metadata.json")
    return ModelMetadata.model_validate_json(blob.download_as_text())


def write_metadata(artifact_uri: str, metadata: ModelMetadata) -> None:
    """Overwrite an artifact directory's metadata.json."""
    bucket_name, _, prefix = artifact_uri.removeprefix("gs://").partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/metadata.json")
    blob.upload_from_string(to_json(metadata), content_type="application/json")


def _rebuild_score_baseline(
    artifact_uri: str, X: pd.DataFrame, previous_spec: BaselineSpec
) -> BaselineSpec:
    """Recompute the training-time churn_probability distribution by rescoring the model.

    The score baseline is the distribution of the model's own predictions on its training
    data, so unlike the feature baselines it cannot be derived from the split table alone —
    it needs the model. Loading model.ubj and rescoring the exact rows and column order the
    trainer used reproduces it, which keeps score-drift monitoring alive; carrying the old
    spec across instead would leave it unmonitored, because it predates stored proportions.

    The reproduction is verified before it is trusted: the new decile edges must match the
    frozen ones. They were computed from the same model on the same rows, so a mismatch
    means something differs (a categorical mapping, a column order) and the safe response is
    to keep the old spec and let the next retrain replace it.
    """
    bucket_name, _, prefix = artifact_uri.removeprefix("gs://").partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/model.ubj")

    with tempfile.NamedTemporaryFile(suffix=".ubj") as tmp:
        blob.download_to_filename(tmp.name)
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(tmp.name)

    scores = pd.DataFrame({CHURN_PROBABILITY_FIELD: model.predict_proba(X)[:, 1]})
    rebuilt = compute_baseline_stats(scores)[CHURN_PROBABILITY_FIELD]

    frozen_edges = getattr(previous_spec, "bin_edges", None)
    if rebuilt.type != "numeric" or not frozen_edges:
        return rebuilt  # nothing comparable to verify against; the recomputed spec is still right

    if not np.allclose(rebuilt.bin_edges, frozen_edges, rtol=1e-3, atol=1e-4, equal_nan=True):
        print(
            "  WARNING: rescored churn_probability does not reproduce the frozen bin edges; "
            "keeping the existing spec (score drift stays unmonitored until the next retrain)."
        )
        return previous_spec

    print("  churn_probability rescored and verified against the frozen edges.")
    return rebuilt


def rebuild(project_id: str, region: str, snapshot_date: str | None, dry_run: bool) -> None:
    """Recompute baseline_stats from the frozen training split and rewrite metadata.json."""
    aiplatform.init(project=project_id, location=region)

    champion = fetch_champion(SETTINGS.model_display_name)
    if champion is None:
        raise SystemExit(
            f"No model labelled role=champion found for {SETTINGS.model_display_name}."
        )

    metadata = read_metadata(champion.uri)
    snapshot_date = snapshot_date or resolve_snapshot_date(project_id, champion, metadata)
    print(f"Champion:       {champion.resource_name}")
    print(f"Artifact URI:   {champion.uri}")
    print(f"Training split: {SETTINGS.split_assignments_table} @ {snapshot_date}")

    df = read_training_split(project_id, snapshot_date)
    if df.empty:
        raise SystemExit(f"No 'train' rows at snapshot_date={snapshot_date}.")

    X, _ = prepare_features(df)
    # Reproduce the training-time column set exactly rather than trusting whatever columns
    # the split table happens to carry: a baseline keyed on different columns than the model
    # was trained on would not line up with what the drift monitor reindexes to at check time.
    X = X[metadata.feature_names]

    baseline_stats = compute_baseline_stats(X)

    previous_score_spec = metadata.baseline_stats.get(CHURN_PROBABILITY_FIELD)
    if previous_score_spec is not None:
        baseline_stats[CHURN_PROBABILITY_FIELD] = _rebuild_score_baseline(
            champion.uri, X, previous_score_spec
        )

    unmonitored = sorted(c for c, spec in baseline_stats.items() if not spec.is_monitored)
    print(f"\nRebuilt {len(baseline_stats)} feature baselines from {len(X)} training rows.")
    for col, spec in sorted(baseline_stats.items()):
        print(f"  {col:38s} {spec.type:12s} buckets={spec.value_cardinality}")
    if unmonitored:
        print(f"\nNo usable drift signal (will be reported unmonitored): {', '.join(unmonitored)}")

    if dry_run:
        print("\n--dry-run: metadata.json not written.")
        return

    write_metadata(champion.uri, metadata.model_copy(update={"baseline_stats": baseline_stats}))
    print(f"\nWrote {champion.uri}/metadata.json")


def main() -> None:
    """Parse arguments and rebuild the champion's baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=os.environ.get("PROJECT_ID"), required=False)
    parser.add_argument("--region", default=os.environ.get("REGION", "europe-west1"))
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="snapshot_date the champion trained on; read from metadata.json when recorded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what the rebuilt baseline looks like without writing metadata.json.",
    )
    args = parser.parse_args()

    if not args.project_id:
        raise SystemExit("Set PROJECT_ID or pass --project-id.")

    rebuild(args.project_id, args.region, args.snapshot_date, args.dry_run)


if __name__ == "__main__":
    main()
