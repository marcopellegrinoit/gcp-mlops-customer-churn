"""GCS read/write helpers for the drift monitor."""

import json

from google.cloud import storage


def download_json(gcs_uri: str) -> dict:
    """Read a JSON file from GCS."""
    bucket_name, blob_name = parse_uri(gcs_uri)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())


def upload_json(gcs_uri: str, obj: dict) -> None:
    """Write a dict as JSON to GCS."""
    bucket_name, blob_name = parse_uri(gcs_uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        json.dumps(obj), content_type="application/json"
    )


def parse_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, blob_name)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob
