"""GCS read helpers shared by evaluate and register."""

import json

from google.cloud import storage


def download_json(gcs_uri: str) -> dict:
    """Read a JSON file from GCS."""
    bucket_name, blob_name = parse_uri(gcs_uri)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())


def upload_text(gcs_uri: str, content: str) -> None:
    """Write a string to GCS."""
    bucket_name, blob_name = parse_uri(gcs_uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(content)


def list_blobs(gcs_uri_prefix: str):
    """List every blob under a gs:// URI prefix."""
    bucket_name, prefix = parse_uri(gcs_uri_prefix)
    return storage.Client().list_blobs(bucket_name, prefix=prefix)


def parse_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into (bucket_name, blob_or_prefix)."""
    path = gcs_uri.removeprefix("gs://")
    bucket, _, blob = path.partition("/")
    return bucket, blob
