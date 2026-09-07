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
#   "ml-common",
#   "pandas>=2",
#   "db-dtypes>=1.7.1,<2",
# ]
#
# [tool.uv.sources]
# ml-common = { path = "../projects/ml_common", editable = true }
# ///

import argparse
import json
import os

import pandas as pd
from google.cloud import aiplatform, bigquery, storage
from ml_common.config import (
    CHURN_PROBABILITY_FIELD,
    MODEL_DISPLAY_NAME,
    SPLIT_ASSIGNMENTS_TABLE,
)
from ml_common.drift import compute_baseline_stats
from ml_common.preprocess import prepare_features


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
            FROM `{project_id}.{SPLIT_ASSIGNMENTS_TABLE}`
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


def resolve_snapshot_date(project_id: str, champion, metadata: dict) -> str:
    """Determine which split_assignments partition the champion actually trained on.

    Models trained after training_snapshot_date was added to metadata.json say so directly.
    For older ones it is inferred as the most recent partition at or before the model's
    registration date — every training run writes a partition, rejected challengers
    included, so the newest partition overall is often some later run's, not this one's.
    """
    recorded = metadata.get("training_snapshot_date")
    if recorded:
        return str(recorded)

    bq = bigquery.Client(project=project_id)
    row = next(
        bq.query(
            f"""
            SELECT MAX(snapshot_date) AS d
            FROM `{project_id}.{SPLIT_ASSIGNMENTS_TABLE}`
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


def read_metadata(artifact_uri: str) -> dict:
    """Download an artifact directory's metadata.json."""
    bucket_name, _, prefix = artifact_uri.removeprefix("gs://").partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/metadata.json")
    return json.loads(blob.download_as_text())


def write_metadata(artifact_uri: str, metadata: dict) -> None:
    """Overwrite an artifact directory's metadata.json."""
    bucket_name, _, prefix = artifact_uri.removeprefix("gs://").partition("/")
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/metadata.json")
    blob.upload_from_string(json.dumps(metadata), content_type="application/json")


def rebuild(project_id: str, region: str, snapshot_date: str | None, dry_run: bool) -> None:
    """Recompute baseline_stats from the frozen training split and rewrite metadata.json."""
    aiplatform.init(project=project_id, location=region)

    champion = fetch_champion(MODEL_DISPLAY_NAME)
    if champion is None:
        raise SystemExit(f"No model labelled role=champion found for {MODEL_DISPLAY_NAME}.")

    metadata = read_metadata(champion.uri)
    snapshot_date = snapshot_date or resolve_snapshot_date(project_id, champion, metadata)
    print(f"Champion:       {champion.resource_name}")
    print(f"Artifact URI:   {champion.uri}")
    print(f"Training split: {SPLIT_ASSIGNMENTS_TABLE} @ {snapshot_date}")

    df = read_training_split(project_id, snapshot_date)
    if df.empty:
        raise SystemExit(f"No 'train' rows at snapshot_date={snapshot_date}.")

    X, _ = prepare_features(df)
    # Reproduce the training-time column set exactly rather than trusting whatever columns
    # the split table happens to carry: a baseline keyed on different columns than the model
    # was trained on would not line up with what the drift monitor reindexes to at check time.
    X = X[metadata["feature_names"]]

    baseline_stats = compute_baseline_stats(X)

    # The churn_probability pseudo-feature is a distribution of the model's own training-time
    # scores. Rescoring here would need the model loaded and the serving container's exact
    # preprocessing, so the existing entry is carried across untouched — it is a plain numeric
    # distribution built by the same code path and is unaffected by the collapsed-bucket
    # defect only if it, too, has stored proportions; if it doesn't, it stays unmonitored
    # until the next retrain, which is the safe direction.
    previous_score_spec = metadata["baseline_stats"].get(CHURN_PROBABILITY_FIELD)
    if previous_score_spec is not None:
        baseline_stats[CHURN_PROBABILITY_FIELD] = previous_score_spec

    unmonitored = sorted(c for c, spec in baseline_stats.items() if not spec.get("monitored"))
    print(f"\nRebuilt {len(baseline_stats)} feature baselines from {len(X)} training rows.")
    for col, spec in sorted(baseline_stats.items()):
        buckets = (
            len(spec["frequencies"])
            if spec["type"] in ("categorical", "discrete")
            else len(spec["bin_edges"]) - 1
        )
        print(f"  {col:38s} {spec['type']:12s} buckets={buckets}")
    if unmonitored:
        print(f"\nNo usable drift signal (will be reported unmonitored): {', '.join(unmonitored)}")

    if dry_run:
        print("\n--dry-run: metadata.json not written.")
        return

    metadata["baseline_stats"] = baseline_stats
    write_metadata(champion.uri, metadata)
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
