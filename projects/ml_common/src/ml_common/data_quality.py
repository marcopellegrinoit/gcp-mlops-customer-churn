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
"""

import pandas as pd

# A snapshot smaller than this fraction of the recent median means the upstream load is
# incomplete — dbt writes the partition whether or not every source event arrived.
MIN_ROW_RATIO: float = 0.5

# Percentage-point rise in a feature's null rate over its training baseline that indicates
# a broken join rather than a population shift. Generous, because genuine missingness does
# move: the failures this catches are the ones that jump to near-total.
MAX_NULL_RATE_INCREASE: float = 0.25


def check_data_quality(
    current: pd.DataFrame,
    baseline_stats: dict,
    recent_row_counts: list[int],
    duplicate_key_count: int,
) -> dict:
    """Run every assertion against the live snapshot and return a pass/fail report.

    recent_row_counts are the row counts of the feature table's preceding partitions
    (excluding this one); duplicate_key_count is how many customer_ids appear more than
    once in this snapshot.
    """
    failures: list[dict] = []

    failures.extend(_check_row_volume(len(current), recent_row_counts))
    failures.extend(_check_duplicate_keys(duplicate_key_count))
    failures.extend(_check_expected_columns(current, baseline_stats))
    failures.extend(_check_null_rates(current, baseline_stats))
    failures.extend(_check_collapsed_columns(current, baseline_stats))

    return {
        "data_quality_failed": bool(failures),
        "failures": failures,
        "row_count": len(current),
    }


def _check_row_volume(row_count: int, recent_row_counts: list[int]) -> list[dict]:
    """A snapshot far smaller than recent ones means an incomplete load, not a shift."""
    if not recent_row_counts:
        return []  # no history to compare against yet (first runs after deploy)

    median_recent = float(pd.Series(recent_row_counts).median())
    if median_recent <= 0:
        return []

    ratio = row_count / median_recent
    if ratio >= MIN_ROW_RATIO:
        return []
    return [
        {
            "check": "row_volume",
            "detail": (
                f"snapshot has {row_count} rows, {ratio:.0%} of the recent median "
                f"({median_recent:.0f}); expected at least {MIN_ROW_RATIO:.0%}"
            ),
        }
    ]


def _check_duplicate_keys(duplicate_key_count: int) -> list[dict]:
    """customer_features is one row per customer per snapshot; more means dedup broke.

    stg_activity_cdc dedupes CDC events by event_id and customer_features picks one row per
    customer with QUALIFY ROW_NUMBER(). If either stops holding, features are silently
    computed over duplicated history and every downstream aggregate is wrong.
    """
    if duplicate_key_count == 0:
        return []
    return [
        {
            "check": "duplicate_keys",
            "detail": f"{duplicate_key_count} customer_id(s) appear more than once in the snapshot",
        }
    ]


def _check_expected_columns(current: pd.DataFrame, baseline_stats: dict) -> list[dict]:
    """Every feature the champion was trained on has to be present to score against it."""
    missing = sorted(set(baseline_stats) - set(current.columns))
    if not missing:
        return []
    return [
        {
            "check": "missing_columns",
            "detail": f"features absent from the snapshot: {', '.join(missing)}",
        }
    ]


def _check_null_rates(current: pd.DataFrame, baseline_stats: dict) -> list[dict]:
    """A null rate that jumps well above its baseline is a broken join, not a shift."""
    failures = []
    for col, spec in baseline_stats.items():
        if col not in current.columns or "null_rate" not in spec:
            continue

        baseline_null = float(spec["null_rate"])
        current_null = float(current[col].isna().mean()) if len(current) else 0.0
        if current_null - baseline_null > MAX_NULL_RATE_INCREASE:
            failures.append(
                {
                    "check": "null_rate",
                    "detail": (
                        f"{col} is {current_null:.0%} null against a {baseline_null:.0%} "
                        f"training baseline"
                    ),
                }
            )
    return failures


def _check_collapsed_columns(current: pd.DataFrame, baseline_stats: dict) -> list[dict]:
    """A feature that had variation at training and is now single-valued has stopped moving.

    This is what an upstream default being written into every row looks like — a constant
    is not a distribution, and PSI on it is unreliable in exactly the case where the
    underlying problem is most severe.
    """
    failures = []
    for col, spec in baseline_stats.items():
        if col not in current.columns or not spec.get("monitored"):
            continue
        if len(current) == 0:
            continue
        if current[col].nunique(dropna=True) <= 1:
            failures.append(
                {
                    "check": "collapsed_column",
                    "detail": f"{col} has a single distinct value but varied at training time",
                }
            )
    return failures
