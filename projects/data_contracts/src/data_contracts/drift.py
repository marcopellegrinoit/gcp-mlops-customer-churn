"""The drift monitor's outputs: the nightly decision file, and one run's PSI history rows.

The decision is not an internal value. It is written to GCS as drift/latest.json and read
back by the Cloud Workflows orchestrator, which branches the whole nightly DAG on
``drift_detected``, ``data_quality.data_quality_failed`` and ``score_drift_detected``. The
workflow reads those with ``map.get(...)`` defaults, so a *missing* key is a defined state
there — which is why the decision is serialised with ``exclude_none`` (data_contracts.serde)
and why the conditionally-computed sections below default to None rather than to an empty
value that would read as "computed, and clear".
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NullRateComparison(BaseModel):
    """A feature's missingness at training time against its missingness now.

    Reported next to the PSI because it is the part of a breach a human most often needs
    stated outright: "PSI 0.4" and "null rate went 3% -> 40%" call for opposite responses,
    and the second is usually a broken upstream join rather than a population shift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline: float
    current: float


class QualityFailure(BaseModel):
    """One failed data-quality assertion, named by check and explained in prose.

    ``detail`` is written for the operator who receives the alert email, not for a parser.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    detail: str


class DataQualityReport(BaseModel):
    """Verdict of every data-quality assertion run against one feature snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_quality_failed: bool
    failures: list[QualityFailure] = Field(default_factory=list)
    row_count: int


class DriftDecision(BaseModel):
    """One drift-monitor run's full verdict, as written to drift/latest.json.

    Assembled in stages — the PSI comparison first, then the score-drift check, the
    persistence rule, and the data-quality gate, each of which may override
    ``drift_detected`` — so this model is mutable, with ``validate_assignment`` on to keep
    every stage's writes checked rather than only the initial construction.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # The field the orchestrator branches on: the verdict after every gate has had its say.
    drift_detected: bool = False

    # Set instead of the feature block when there is no champion to compare against.
    reason: str | None = None

    champion_model: str | None = None
    snapshot_date: str | None = None

    # Per-feature PSI comparison (ml_common.drift.evaluate_drift).
    threshold: float | None = None
    feature_psi: dict[str, float] = Field(default_factory=dict)
    feature_thresholds: dict[str, float] = Field(default_factory=dict)
    feature_null_rates: dict[str, NullRateComparison] = Field(default_factory=dict)
    breached_features: dict[str, float] = Field(default_factory=dict)
    unmonitored_features: list[str] = Field(default_factory=list)
    current_sample_size: int = 0

    # Score-distribution drift: monitoring-only, and absent entirely when the champion's
    # baseline predates it or the check could not run.
    score_psi: float | None = None
    score_threshold: float | None = None
    score_drift_detected: bool | None = None

    # Persistence rule: this run's raw verdict, and how many of the last N runs each
    # breaching feature has now breached in.
    run_drift_detected: bool | None = None
    persistence_window: int | None = None
    persistence_min_breaches: int | None = None
    breach_counts: dict[str, int] = Field(default_factory=dict)
    persistent_breaches: dict[str, float] = Field(default_factory=dict)

    # Data-quality gate. retrain_suppressed_by_data_quality records that a retrain *would*
    # have been triggered had the snapshot been sound — otherwise a suppressed night is
    # indistinguishable from a quiet one.
    data_quality: DataQualityReport | None = None
    retrain_suppressed_by_data_quality: bool | None = None


class DriftMetricRow(BaseModel):
    """One row of ml.drift_metrics: one feature's PSI in one run.

    Mirrors the table's schema in iac/config/bigquery.yaml field for field; the persistence
    rule reads this history back to decide whether a breach has repeated often enough to
    gate retraining.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_ts: datetime
    snapshot_date: str
    champion_model: str
    feature: str
    psi: float
    threshold: float
    breached: bool
    monitored: bool
