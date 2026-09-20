"""Upload a compiled KFP pipeline template to an Artifact Registry KFP repository."""

import argparse

from kfp.registry import RegistryClient


def upload_pipeline(file_name: str, host: str) -> None:
    """Upload the compiled pipeline YAML to the given Artifact Registry KFP host, tagged latest."""
    client = RegistryClient(host=host)
    client.upload_pipeline(file_name=file_name, tags=["latest"])


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Upload a compiled pipeline template to Artifact Registry"
    )
    _parser.add_argument("--file", required=True, help="Path to the compiled pipeline.yaml")
    _parser.add_argument("--host", required=True, help="Artifact Registry KFP repository host URL")
    _args = _parser.parse_args()
    upload_pipeline(_args.file, _args.host)
