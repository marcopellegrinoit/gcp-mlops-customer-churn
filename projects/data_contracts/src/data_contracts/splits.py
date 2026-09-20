"""Reference to one frozen train/test split partition of ml.split_assignments."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SplitRef(BaseModel):
    """Which ml.split_assignments partition and side a training stage should read.

    Passed between pipeline stages as a KFP artifact file rather than as a GCS data URI:
    the split itself is a permanent BigQuery partition, and this is the pointer to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_date: str
    split: Literal["train", "test"]
