"""Runtime configuration for the churn dashboard, read from the Cloud Run environment.

Every value is set by Terraform on the Cloud Run service (iac/config/cloud_run_services.yaml,
plus BQ_PROJECT_ID injected in iac/locals.tf), so retuning thresholds or the cache window is
an apply rather than an image rebuild.

Infrastructure fields are required and have no defaults, following the same rule as the
drift monitor: a missing variable must fail the container rather than resolve to None and
render an empty dashboard that looks like a quiet night with no at-risk customers.

Unlike the Cloud Run *jobs*, that failure does not come for free here. Streamlit does not
execute app.py — where these settings are read — until the first request arrives, and its
health endpoint is served by the server itself, so a misconfigured container would otherwise
start, pass its startup probe, take traffic, and only then show every user an error page.
dashboard.serve parses these settings before binding the port precisely to close that gap;
keep that call if this module is ever refactored.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    """Everything the dashboard needs, read from the service's environment."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    # --- Infrastructure identity: required, no defaults ----------------------------------
    # BQ_PROJECT_ID is the name iac/locals.tf injects into every deployable; PROJECT_ID is
    # accepted too so a local `streamlit run` or one-off `docker run` can use the obvious name.
    project_id: str = Field(validation_alias=AliasChoices("BQ_PROJECT_ID", "PROJECT_ID"))
    bq_location: str = "EU"
    risk_table: str = "features.churn_risk_current"
    trend_table: str = "features.churn_risk_daily"

    # --- Cost guards ---------------------------------------------------------------------
    # Hard ceiling on a single query, enforced by BigQuery itself: a job projected to exceed
    # it is rejected before it runs rather than billed. This is the backstop for the caching
    # strategy, not a substitute for it — if a view is ever rewritten into something that
    # scans full history, this turns a surprise bill into a visible error.
    max_bytes_billed: int = Field(default=2 * 1024**3, gt=0)

    # How long a fetched snapshot stays in the process cache. The pipeline writes
    # predictions once a night, so anything under a few hours re-queries BigQuery for data
    # that provably has not changed. Filtering and sorting happen in pandas against this
    # cached frame, so no user interaction costs a query.
    cache_ttl_seconds: int = Field(default=1800, gt=0)

    # --- Risk banding --------------------------------------------------------------------
    # Probability cutoffs for the High / Medium / Low bands shown to business users. These
    # are a *presentation* choice, independent of the champion's registered decision
    # threshold (which produces churn_prediction): the model's operating point answers "act
    # or not", while the bands answer "in what order should a finite retention team work".
    high_risk_threshold: float = Field(default=0.7, gt=0.0, lt=1.0)
    medium_risk_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)

    # A driver feature is called out for a customer only when it is at least this many times
    # the cohort median (or, for "higher is better" features, this far below it). Without a
    # floor, nearly every customer shows a full list of marginal deviations and the
    # explanation panel stops distinguishing anything.
    indicator_ratio: float = Field(default=1.5, gt=1.0)

    @model_validator(mode="after")
    def _check_band_ordering(self):
        """Reject bands that would leave the Medium range empty or inverted."""
        if self.medium_risk_threshold >= self.high_risk_threshold:
            raise ValueError(
                "medium_risk_threshold must be below high_risk_threshold; got "
                f"medium={self.medium_risk_threshold} high={self.high_risk_threshold}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> DashboardSettings:
    """Return the process-wide settings, parsed once."""
    return DashboardSettings()
