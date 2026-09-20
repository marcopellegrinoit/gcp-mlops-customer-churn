"""GCP infrastructure and ops configuration for the drift monitor.

Every value arrives as an environment variable set by Terraform on the Cloud Run job
(iac/config/cloud_run_jobs.yaml, plus BQ_PROJECT_ID injected in iac/locals.tf), so the
image is environment-agnostic and retuning the monitor is an apply rather than a rebuild.

The infrastructure fields are **required**. They used to be read with ``os.environ.get()``
and annotated ``str``, which meant a missing variable produced ``None`` and the job ran to
completion against ``gs://None-None/None`` — a silent no-op that looked like a healthy
night. A missing variable now fails the container at startup, where Cloud Run reports it.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DriftMonitorSettings(BaseSettings):
    """Everything the nightly drift check needs, read from the job's environment."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    # --- Infrastructure identity: required, no defaults ----------------------------------
    # BQ_PROJECT_ID is the name iac/locals.tf injects into every Cloud Run job; PROJECT_ID
    # is accepted too so a local run or a one-off `docker run` can use the obvious name.
    project_id: str = Field(validation_alias=AliasChoices("BQ_PROJECT_ID", "PROJECT_ID"))
    region: str
    bq_features_table: str
    model_display_name: str
    gcs_bucket: str
    decision_blob: str
    # Must match submit_batch_predict's displayName in workflows/orchestrator-workflow.yaml —
    # it disambiguates the daily production-scoring job from post-training's own
    # evaluation-time batch predictions against the same champion model.
    batch_predict_display_name: str

    # --- Detection policy ---------------------------------------------------------------
    # The effect size the business considers meaningful. A feature must clear this *and* the
    # bootstrapped sampling-noise floor for the snapshot's size before it counts as breached.
    psi_threshold: float = Field(default=0.2, gt=0.0)

    # A feature must breach in at least persistence_min_breaches of the last
    # persistence_window runs before it gates retraining. 2-of-3 is enough to reject a
    # one-off sampling artefact or a partial upstream load without delaying a real shift by
    # more than a couple of days.
    persistence_window: int = Field(default=3, ge=1)
    persistence_min_breaches: int = Field(default=2, ge=1)

    # How many preceding feature-table partitions the data-quality row-volume assertion
    # compares this snapshot against. A week smooths over any single odd day without
    # reaching so far back that a deliberate change in scale looks like a failed load.
    quality_history_partitions: int = Field(default=7, ge=0)

    @model_validator(mode="after")
    def _check_persistence_rule(self):
        """Reject a rule no run can ever satisfy.

        min_breaches above the window means a feature would have to breach more times than
        there are runs to breach in, so drift would never once trigger retraining — and the
        symptom is silence, which is indistinguishable from a healthy platform. Cheaper to
        refuse the deployment than to discover it months later.
        """
        if self.persistence_min_breaches > self.persistence_window:
            raise ValueError(
                f"persistence_min_breaches ({self.persistence_min_breaches}) exceeds "
                f"persistence_window ({self.persistence_window}): no run could ever trigger "
                "retraining"
            )
        return self

    @property
    def decision_gcs_uri(self) -> str:
        """Where the orchestrator reads this run's decision from."""
        return f"gs://{self.project_id}-{self.gcs_bucket}/{self.decision_blob}"

    @property
    def staging_bucket_uri(self) -> str:
        """Vertex AI staging bucket for this project's pipeline metadata."""
        return f"gs://{self.project_id}-pipeline-metadata"


@lru_cache(maxsize=1)
def get_settings() -> DriftMonitorSettings:
    """Return the job's settings, read from the environment once."""
    return DriftMonitorSettings()
