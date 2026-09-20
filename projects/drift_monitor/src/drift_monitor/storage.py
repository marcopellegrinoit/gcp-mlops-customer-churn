"""GCS read/write helpers for the drift monitor."""

import json

from data_contracts import DriftDecision, to_json
from google.cloud import storage


def download_json(gcs_uri: str) -> dict:
    """Read a JSON file from GCS."""
    bucket_name, blob_name = parse_uri(gcs_uri)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())


def upload_decision(gcs_uri: str, decision: DriftDecision) -> None:
    """Write this run's decision to GCS, where the orchestrator workflow reads it back.

    Serialised through data_contracts.to_json, which omits unset fields — the workflow
    distinguishes an absent optional section (no score check ran, no data-quality report)
    from a present-and-false one via map.get defaults.
    """
    bucket_name, blob_name = parse_uri(gcs_uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        to_json(decision), content_type="application/json"
    )


def parse_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, blob_name)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob
