"""Runtime configuration for the synthetic CDC generator, read from the job's environment.

Set by Terraform on the Cloud Run job (iac/config/cloud_run_jobs.yaml, with BQ_PROJECT_ID
injected in iac/locals.tf). The BigQuery destination is required — there is no sensible
default for "which project's data warehouse to write to" — while the simulation's shape
(batch size, anomaly rate, population growth) defaults to the values the platform runs
with and can be retuned by editing the job's env_vars and applying.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """BigQuery destination and simulation parameters for one generation run."""

    model_config = SettingsConfigDict(env_prefix="BQ_", extra="ignore", frozen=True)

    project_id: str
    dataset_id: str
    table_id: str

    # Simulation knobs: not BQ_-prefixed, so they keep the names the job already sets.
    batch_size: int = Field(default=2000, gt=0, validation_alias="BATCH_SIZE")
    # Fraction of events given deliberately out-of-range values, to exercise the downstream
    # drift and data-quality checks. 0.0 in the deployed job; raised for a drill.
    anomaly_rate: float = Field(default=0.0, ge=0.0, le=1.0, validation_alias="ANOMALY_RATE")
    # Founding base, acquired on the pool epoch.
    customer_pool_size: int = Field(default=10000, gt=0, validation_alias="USER_POOL_SIZE")
    # New customers acquired per elapsed day. Without acquisition the base only ever shrinks
    # as customers churn, and its tenure distribution climbs without bound.
    daily_acquisitions: int = Field(default=25, ge=0, validation_alias="DAILY_ACQUISITIONS")

    @property
    def table_ref(self) -> str:
        """Return the fully-qualified BigQuery table reference."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the job's settings, read from the environment once."""
    return Settings()
