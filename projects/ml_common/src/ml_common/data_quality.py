"""Data-quality assertions that gate retraining, kept separate from distribution drift.

Drift and defects look alike from a PSI score alone — a broken upstream join, a partial
load, or a column that silently became NULL all move a distribution just as a real
population shift does. The responses are opposite. A genuine shift means the world changed
and the model should be retrained on it; a defect means the *data* is wrong, and
retraining on it launders the defect into the model and ships it to production. Worse, the
retrained challenger can even pass the promotion gate, because it is evaluated against a
test split drawn from the same corrupted snapshot.

So these assertions run first and are absolute, not distributional: they compare the live
snapshot against the champion's frozen baseline and against the feature table's own recent
history, and a failure suppresses the retrain decision entirely and pages a human instead.

Deliberately assertions, not PSI: "this column is now 100% NULL" and "we loaded a fifth of
the usual rows" are facts about the pipeline, and they should fire on the first occurrence
rather than waiting for the N-of-M persistence rule that governs genuine drift.

The two tolerances these checks apply — ``min_row_ratio`` and ``max_null_rate_increase`` —
are deployment policy rather than constants, and live in ml_common.config where a Terraform
env var can move them without an image rebuild.
"""

from collections.abc import Mapping

import pandas as pd

from ml_common.config import MLSettings, get_settings
from ml_common.contracts import (
    CHURN_PROBABILITY_FIELD,
    BaselineSpec,
    DataQualityReport,
    QualityFailure,
    parse_baseline_stats,
)


def check_data_quality(
    current: pd.DataFrame,
    baseline_stats: Mapping[str, object],
    recent_row_counts: list[int],
    duplicate_key_count: int,
    settings: MLSettings | None = None,
) -> DataQualityReport:
    """Run every assertion against the live snapshot and return a pass/fail report.

    recent_row_counts are the row counts of the feature table's preceding partitions
    (excluding this one); duplicate_key_count is how many customer_ids appear more than
    once in this snapshot. settings defaults to the process environment; pass one
    explicitly to assert against a specific tolerance rather than the deployed one.
    """
    settings = settings or get_settings()
    baseline = parse_baseline_stats(baseline_stats)
    failures: list[QualityFailure] = []

    failures.extend(_check_row_volume(len(current), recent_row_counts, settings.min_row_ratio))
    failures.extend(_check_duplicate_keys(duplicate_key_count))
    failures.extend(_check_expected_columns(current, baseline))
    failures.extend(_check_null_rates(current, baseline, settings.max_null_rate_increase))
    failures.extend(_check_collapsed_columns(current, baseline))

    return DataQualityReport(
        data_quality_failed=bool(failures), failures=failures, row_count=len(current)
    )


def _check_row_volume(
    row_count: int, recent_row_counts: list[int], min_row_ratio: float
) -> list[QualityFailure]:
    """A snapshot far smaller than recent ones means an incomplete load, not a shift."""
    if not recent_row_counts:
        return []  # no history to compare against yet (first runs after deploy)

    median_recent = float(pd.Series(recent_row_counts).median())
    if median_recent <= 0:
        return []

    ratio = row_count / median_recent
    if ratio >= min_row_ratio:
        return []
    return [
        QualityFailure(
            check="row_volume",
            detail=(
                f"snapshot has {row_count} rows, {ratio:.0%} of the recent median "
                f"({median_recent:.0f}); expected at least {min_row_ratio:.0%}"
            ),
        )
    ]


def _check_duplicate_keys(duplicate_key_count: int) -> list[QualityFailure]:
    """customer_features is one row per customer per snapshot; more means dedup broke.

    stg_activity_cdc dedupes CDC events by event_id and customer_features picks one row per
    customer with QUALIFY ROW_NUMBER(). If either stops holding, features are silently
    computed over duplicated history and every downstream aggregate is wrong.
    """
    if duplicate_key_count == 0:
        return []
    return [
        QualityFailure(
            check="duplicate_keys",
            detail=f"{duplicate_key_count} customer_id(s) appear more than once in the snapshot",
        )
    ]


def _check_expected_columns(
    current: pd.DataFrame, baseline: dict[str, BaselineSpec]
) -> list[QualityFailure]:
    """Every feature the champion was trained on has to be present to score against it."""
    missing = sorted(_input_features(baseline) - set(current.columns))
    if not missing:
        return []
    return [
        QualityFailure(
            check="missing_columns",
            detail=f"features absent from the snapshot: {', '.join(missing)}",
        )
    ]


def _check_null_rates(
    current: pd.DataFrame, baseline: dict[str, BaselineSpec], max_null_rate_increase: float
) -> list[QualityFailure]:
    """A null rate that jumps well above its baseline is a broken join, not a shift."""
    failures = []
    for col, spec in baseline.items():
        if col not in current.columns or not spec.tracks_nulls:
            continue

        baseline_null = float(spec.null_rate)
        current_null = float(current[col].isna().mean()) if len(current) else 0.0
        if current_null - baseline_null > max_null_rate_increase:
            failures.append(
                QualityFailure(
                    check="null_rate",
                    detail=(
                        f"{col} is {current_null:.0%} null against a {baseline_null:.0%} "
                        f"training baseline"
                    ),
                )
            )
    return failures


def _check_collapsed_columns(
    current: pd.DataFrame, baseline: dict[str, BaselineSpec]
) -> list[QualityFailure]:
    """A feature that had variation at training and is now single-valued has stopped moving.

    This is what an upstream default being written into every row looks like — a constant
    is not a distribution, and PSI on it is unreliable in exactly the case where the
    underlying problem is most severe.

    Gated on the baseline's own *value* cardinality rather than on `monitored`: a column can
    be monitored purely because its null rate carries signal while its present values were
    already constant at training time (days_since_last_successful_payment is exactly this).
    Such a column has not collapsed — it never varied — and flagging it would fail this
    check every single night.
    """
    failures = []
    if len(current) == 0:
        return failures

    for col, spec in baseline.items():
        if col not in current.columns or spec.value_cardinality <= 1:
            continue
        if current[col].nunique(dropna=True) <= 1:
            failures.append(
                QualityFailure(
                    check="collapsed_column",
                    detail=f"{col} has a single distinct value but varied at training time",
                )
            )
    return failures


def _input_features(baseline: dict[str, BaselineSpec]) -> set[str]:
    """Baseline entries that are genuinely input columns of the feature snapshot.

    The baseline also carries the churn_probability pseudo-feature — the model's own
    training-time score distribution, folded in so score drift reuses the same machinery.
    It is an output, never a column of customer_features, so it is not "missing" from a
    snapshot and must not be asserted on.
    """
    return set(baseline) - {CHURN_PROBABILITY_FIELD}
