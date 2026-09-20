"""GCS read helpers for the batch-prediction FastAPI app."""

import json

from google.cloud import storage


def download_json(gcs_uri: str, project_id: str) -> dict:
    """Read a JSON file from GCS."""
    bucket_name, blob_name = _parse_uri(gcs_uri)
    blob = storage.Client(project=project_id).bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())


def download_text(gcs_uri: str, project_id: str) -> str:
    """Read a text file from GCS."""
    bucket_name, blob_name = _parse_uri(gcs_uri)
    return storage.Client(project=project_id).bucket(bucket_name).blob(blob_name).download_as_text()


def _parse_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, blob_name)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob
