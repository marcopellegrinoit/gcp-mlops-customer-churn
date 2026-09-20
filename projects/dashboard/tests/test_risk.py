import math
from pathlib import Path

import pandas as pd
import pytest
from dashboard import charts, risk


def test_bands_split_at_the_configured_thresholds():
    probabilities = pd.Series([0.0, 0.39, 0.4, 0.69, 0.7, 1.0])
    bands = risk.assign_risk_band(probabilities, high_threshold=0.7, medium_threshold=0.4)
    assert list(bands) == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_band_boundaries_are_inclusive_at_the_lower_edge():
    # A probability exactly on a threshold belongs to the worse band. The KPI row counts
    # high risk as `>= high_threshold`, so a band that excluded the boundary would report a
    # different high-risk count than the chart beside it for that one customer.
    bands = risk.assign_risk_band(pd.Series([0.7]), high_threshold=0.7, medium_threshold=0.4)
    assert bands.iloc[0] == "High"


def test_sql_and_python_high_risk_floors_agree():
    """The trend view hardcodes its own high-risk floor; it must match the default here.

    models/marts/churn_risk_daily.sql cannot import a Python constant, so the two are
    duplicated. If they diverge, the trend line and the table beneath it describe different
    populations while both are labelled "high risk" — a disagreement nobody would spot by
    looking. This test is the join between them.
    """
    sql = Path(__file__).parents[2] / "dbt_transform/models/marts/churn_risk_daily.sql"
    text = sql.read_text()
    assert f"churn_probability >= {risk.HIGH_RISK_FLOOR_IN_SQL}" in text

    from dashboard.settings import DashboardSettings

    defaults = DashboardSettings(BQ_PROJECT_ID="test")
    assert defaults.high_risk_threshold == risk.HIGH_RISK_FLOOR_IN_SQL


def test_cohort_reference_falls_back_off_a_zero_median(snapshot):
    reference = risk.cohort_reference(snapshot)
    # 70% of the base has no payment failures, so the median is 0 and comparing against it
    # would flag every customer with a single failure as an outlier.
    assert snapshot["payment_failure_rate_30d"].median() == 0.0
    assert reference["payment_failure_rate_30d"] > 0.0


def test_indicators_rank_the_worst_deviation_first(snapshot):
    reference = risk.cohort_reference(snapshot)
    customer = snapshot.iloc[0].copy()
    customer["contact_requests_last_30d"] = reference["contact_requests_last_30d"] * 10
    customer["avg_engagement_30d"] = reference["avg_engagement_30d"] * 0.9  # within the floor

    indicators = risk.customer_indicators(customer, reference, ratio=1.5)

    assert indicators, "a 10x deviation must produce an indicator"
    assert indicators[0]["label"] == "Support contacts (30d)"
    severities = [i["severity"] for i in indicators]
    assert severities == sorted(severities, reverse=True)


def test_a_cohort_typical_customer_gets_no_indicators(snapshot):
    reference = risk.cohort_reference(snapshot)
    customer = snapshot.iloc[0].copy()
    for column, value in reference.items():
        customer[column] = value

    assert risk.customer_indicators(customer, reference, ratio=1.5) == []


def test_never_having_paid_is_surfaced_rather_than_dropped(snapshot):
    """A NULL days-since-last-payment is the strongest signal there is, not missing data."""
    reference = risk.cohort_reference(snapshot)
    customer = snapshot.iloc[0].copy()
    customer["days_since_last_successful_payment"] = None

    indicators = risk.customer_indicators(customer, reference, ratio=1.5)
    matching = [i for i in indicators if i["label"].startswith("Days since")]

    assert len(matching) == 1
    assert matching[0]["detail"] == "never recorded a successful payment"
    assert math.isinf(float(matching[0]["severity"]))
    # Infinite severity must sort it to the top, ahead of any finite deviation.
    assert indicators[0] is matching[0]


def test_other_missing_drivers_are_skipped_not_reported(snapshot):
    reference = risk.cohort_reference(snapshot)
    customer = snapshot.iloc[0].copy()
    customer["avg_engagement_30d"] = None

    labels = [i["label"] for i in risk.customer_indicators(customer, reference, ratio=1.5)]
    assert "Engagement score (30d)" not in labels


def test_portfolio_summary_counts_and_values_the_high_risk_group(snapshot):
    summary = risk.portfolio_summary(snapshot, high_threshold=0.7)
    high = snapshot[snapshot["churn_probability"] >= 0.7]

    assert summary["customers_scored"] == len(snapshot)
    assert summary["high_risk_customers"] == len(high)
    assert summary["high_risk_share"] == pytest.approx(len(high) / len(snapshot))
    assert summary["monthly_value_at_risk"] == pytest.approx(high["monthly_value"].sum())


def test_portfolio_summary_survives_an_empty_snapshot():
    empty = pd.DataFrame({"churn_probability": [], "monthly_value": []})
    summary = risk.portfolio_summary(empty, high_threshold=0.7)
    assert summary == {
        "customers_scored": 0,
        "high_risk_customers": 0,
        "high_risk_share": 0.0,
        "monthly_value_at_risk": 0.0,
        "mean_probability": 0.0,
    }


def test_band_display_order_is_worst_first():
    assert charts.BAND_ORDER == ["High", "Medium", "Low"]
    assert set(charts.BAND_ORDER) == set(risk.RISK_BANDS)
