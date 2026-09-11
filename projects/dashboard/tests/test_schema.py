"""Unit tests for the dashboard's contract with the two dbt marts."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from dashboard.schema import (
    ChurnRiskDailyRow,
    ChurnRiskRow,
    MartContractError,
    require_columns,
    validate_trend,
)
from data_contracts import CHURN_PREDICTION_FIELD, CHURN_PROBABILITY_FIELD, ChurnPrediction

_MARTS = Path(__file__).parents[2] / "dbt_transform/models/marts"


def _selected_columns(model_sql: Path) -> set[str]:
    """Return the output column names a mart's SELECT produces.

    Reads the alias where one is given (`f.avg_transaction_30d AS monthly_value`) and the
    bare column otherwise, after stripping comments and everything from the top-level FROM
    onward — which is what keeps the table aliases in `FROM ... AS p` out of the result.
    Crude by design: it only has to be good enough to notice that a column the dashboard
    reads has stopped being emitted.
    """
    text = re.sub(r"/\*.*?\*/", "", model_sql.read_text(), flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", "", text)
    select_list = re.split(r"^FROM\s", text, flags=re.MULTILINE)[0]
    aliased = set(re.findall(r"\bAS\s+([a-z_][a-z0-9_]*)", select_list, flags=re.IGNORECASE))
    bare = set(
        re.findall(r"^\s*(?:[a-z]\.)?([a-z_][a-z0-9_]*),\s*$", select_list, flags=re.MULTILINE)
    )
    return aliased | bare


def _trend_frame(days: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime([f"2026-09-{d:02d}" for d in range(1, days + 1)]),
            "customers_scored": [1000] * days,
            "predicted_churners": [80] * days,
            "high_risk_customers": [40] * days,
            "mean_churn_probability": [0.21] * days,
            "predicted_churn_rate": [0.08] * days,
            "model_versions_used": [1] * days,
            "last_predicted_at": pd.to_datetime(["2026-09-01T02:00:00Z"] * days),
        }
    )


# ---------------------------------------------------------------------------
# the contract is honoured by the marts it describes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "mart"),
    [(ChurnRiskRow, "churn_risk_current.sql"), (ChurnRiskDailyRow, "churn_risk_daily.sql")],
)
def test_every_required_column_is_emitted_by_the_mart(model, mart):
    """The join between the dbt models and the app that reads them.

    Python and SQL cannot share a column list, so a rename on one side would otherwise only
    be discovered by a business user looking at a broken page. This is the same
    reconciliation-by-test approach risk.HIGH_RISK_FLOOR_IN_SQL uses for the band threshold.
    """
    emitted = _selected_columns(_MARTS / mart)
    missing = sorted(set(model.model_fields) - emitted)
    assert not missing, f"{mart} no longer emits: {missing}"


def test_the_prediction_columns_match_the_serving_contract():
    """The column this dashboard reads is the one the serving container writes.

    churn_probability is produced by the serving container as
    data_contracts.ChurnPrediction, synced into ml.predictions, and surfaced by the mart.
    All three points are pinned here: the shared model's own field name, the mart SQL that
    carries it, and this app's row contract. Renaming it anywhere fails this test rather
    than quietly emptying a column in the worklist.
    """
    emitted = _selected_columns(_MARTS / "churn_risk_current.sql")

    assert CHURN_PROBABILITY_FIELD in ChurnPrediction.model_fields
    assert CHURN_PROBABILITY_FIELD in emitted
    assert CHURN_PROBABILITY_FIELD in ChurnRiskRow.model_fields

    # churn_prediction is the thresholded label. The dashboard bands on the probability
    # instead (see docs/dashboard.md), so it does not read this column — but the mart still
    # carries it, and churn_risk_daily aggregates it into predicted_churn_rate.
    assert CHURN_PREDICTION_FIELD in ChurnPrediction.model_fields
    assert CHURN_PREDICTION_FIELD in emitted


# ---------------------------------------------------------------------------
# require_columns — the snapshot boundary
# ---------------------------------------------------------------------------


def test_a_conforming_snapshot_passes_through_unchanged(snapshot):
    assert require_columns(snapshot, ChurnRiskRow, "features.churn_risk_current") is snapshot


def test_a_dropped_column_names_itself_and_the_view(snapshot):
    # The failure this exists to convert: without it, a renamed column surfaces as a KeyError
    # inside an Altair encoding, or as a driver that silently stops being listed.
    broken = snapshot.drop(columns=["payment_failure_rate_30d"])
    with pytest.raises(MartContractError, match="payment_failure_rate_30d"):
        require_columns(broken, ChurnRiskRow, "features.churn_risk_current")


def test_extra_mart_columns_are_allowed(snapshot):
    # dbt must stay free to add columns without a dashboard release.
    widened = snapshot.assign(newly_added_by_dbt=1.0)
    require_columns(widened, ChurnRiskRow, "features.churn_risk_current")


def test_a_frame_with_no_schema_at_all_is_the_empty_state(snapshot):
    # "The pipeline has not scored anything yet", which app.main renders as an explicit
    # message. Turning that into an error page would be a regression.
    require_columns(pd.DataFrame(), ChurnRiskRow, "features.churn_risk_current")


def test_an_empty_result_still_has_its_schema_checked(snapshot):
    # An empty *result set* still carries the view's columns, so a quiet day is no excuse
    # to skip the check.
    empty_but_typed = snapshot.iloc[0:0].drop(columns=["churn_probability"])
    with pytest.raises(MartContractError, match="churn_probability"):
        require_columns(empty_but_typed, ChurnRiskRow, "features.churn_risk_current")


# ---------------------------------------------------------------------------
# validate_trend — the row-level boundary
# ---------------------------------------------------------------------------


def test_a_conforming_trend_passes_through_unchanged():
    frame = _trend_frame()
    assert validate_trend(frame, "features.churn_risk_daily") is frame


def test_a_null_rate_is_accepted():
    # SAFE_DIVIDE returns NULL on a day that scored nobody; the chart leaves a gap there.
    frame = _trend_frame()
    frame.loc[1, "predicted_churn_rate"] = None
    validate_trend(frame, "features.churn_risk_daily")


def test_a_rate_above_one_is_rejected():
    # A view rewrite that changes what the rate divides by produces a number that is wrong
    # rather than absent — and 140% plotted on a percentage axis still looks like data.
    frame = _trend_frame()
    frame.loc[1, "predicted_churn_rate"] = 1.4
    with pytest.raises(MartContractError, match="predicted_churn_rate"):
        validate_trend(frame, "features.churn_risk_daily")


def test_a_negative_count_is_rejected():
    frame = _trend_frame()
    frame.loc[0, "high_risk_customers"] = -1
    with pytest.raises(MartContractError, match="high_risk_customers"):
        validate_trend(frame, "features.churn_risk_daily")


def test_an_empty_trend_is_accepted():
    # Before the second scored day there is nothing to plot, and app.main says so.
    validate_trend(_trend_frame().iloc[0:0], "features.churn_risk_daily")


# ---------------------------------------------------------------------------
# pandas' several spellings of NULL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("null", [float("nan"), None, pd.NA])
def test_every_pandas_null_representation_reads_as_absent(null):
    # A BigQuery NULL does not survive to_dataframe() as None: it is NaN in a float column
    # and pd.NA in a nullable-extension one. NaN is the dangerous one — it passes an
    # `is not None` check and then fails every bound, so a legitimately NULL rate would be
    # reported as out of range rather than as missing.
    row = ChurnRiskDailyRow.model_validate(
        {
            "snapshot_date": "2026-09-01",
            "customers_scored": 0,
            "high_risk_customers": 0,
            "predicted_churn_rate": null,
            "model_versions_used": 0,
        }
    )
    assert row.predicted_churn_rate is None


def test_nullable_snapshot_columns_accept_nan():
    # days_since_last_successful_payment is NULL for a customer who has never paid
    # successfully, and avg_engagement_30d is NULL for offline-only customers.
    row = ChurnRiskRow.model_validate(
        {
            "customer_id": "c1",
            "churn_probability": 0.9,
            "model_version": "models/1",
            "snapshot_date": "2026-09-01",
            "days_since_last_successful_payment": float("nan"),
            "avg_engagement_30d": float("nan"),
        }
    )
    assert row.days_since_last_successful_payment is None
    assert row.avg_engagement_30d is None


def test_an_out_of_range_probability_is_still_rejected():
    # The NaN handling must not have widened the bound it sits in front of.
    with pytest.raises(Exception, match="churn_probability"):
        ChurnRiskRow.model_validate(
            {
                "customer_id": "c1",
                "churn_probability": 1.2,
                "model_version": "models/1",
                "snapshot_date": "2026-09-01",
            }
        )
