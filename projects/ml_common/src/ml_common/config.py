"""Model policy and resource naming, overridable per deployment through the environment.

These are the knobs the data science team owns — what counts as an improvement worth
promoting, how aggressively the data-quality assertions fire, which tables the platform
reads and writes. Every one has a working default, so importing this module never requires
an environment: the training pipeline is *compiled* on a developer's laptop and on Cloud
Build, where none of these variables are set, and a required field here would break that.

Overriding is by environment variable with an ``ML_`` prefix — ``ML_TARGET_RECALL``,
``ML_PR_AUC_MIN_DELTA``, and so on — set in iac/config/cloud_run_jobs.yaml for the Cloud
Run jobs and threaded onto the Vertex AI pipeline's containers as pipeline parameters (see
training_pipeline.tasks). Changing one is a Terraform apply, not an image rebuild.

Infrastructure identity (project, region, bucket names) is deliberately *not* here. That
belongs to whichever service is deployed into that infrastructure, and is required rather
than defaulted, so a missing value fails the container at start instead of silently
pointing production at a default. See drift_monitor.config, serving.settings and
data_generator.config.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLSettings(BaseSettings):
    """Tunable model policy and BigQuery resource names, read from ``ML_``-prefixed env vars."""

    model_config = SettingsConfigDict(env_prefix="ML_", extra="ignore", frozen=True)

    # --- Promotion gate -----------------------------------------------------------------
    # Both metrics must improve together, so a challenger cannot be promoted by trading one
    # away for the other.
    pr_auc_min_delta: float = 0.02
    f1_min_delta: float = 0.01
    # Consecutive failures to promote before the run recommends a feature review: a
    # challenger that keeps losing points at the feature set, not at the training data.
    max_consecutive_rejections: int = Field(default=3, gt=0)

    # --- Training -----------------------------------------------------------------------
    target_recall: float = Field(default=0.80, gt=0.0, le=1.0)
    random_state: int = 42
    test_size: float = Field(default=0.20, gt=0.0, lt=1.0)

    # --- Data-quality assertions --------------------------------------------------------
    # A snapshot smaller than this fraction of the recent median means the upstream load is
    # incomplete — dbt writes the partition whether or not every source event arrived.
    min_row_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    # Percentage-point rise in a feature's null rate over its training baseline that
    # indicates a broken join rather than a population shift. Generous, because genuine
    # missingness does move: the failures this catches are the ones that jump to near-total.
    max_null_rate_increase: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Resource naming ----------------------------------------------------------------
    # Shared across the training pipeline, trainer, post-training and the drift monitor —
    # changed once here rather than retyped at each call site.
    bq_features_table: str = "features.customer_features"
    split_assignments_table: str = "ml.split_assignments"
    drift_metrics_table: str = "ml.drift_metrics"
    model_display_name: str = "churn-predictor"
    consecutive_rejection_key: str = "consecutive_rejections"


@lru_cache(maxsize=1)
def get_settings() -> MLSettings:
    """Return the process-wide ML settings, read from the environment once.

    Cached because these are process configuration, not per-call arguments: a container's
    environment does not change under it, and re-reading would let the same run disagree
    with itself. Tests that override the environment call ``get_settings.cache_clear()``,
    or pass an explicit ``MLSettings`` to the functions that accept one.
    """
    return MLSettings()
