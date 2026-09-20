"""Risk banding and cohort-comparison indicators — pure pandas, no GCP, no Streamlit.

Everything a business user reads as an *interpretation* of the model lives here, isolated
from both the query layer and the UI so it can be unit-tested without credentials or a
browser session.

A deliberate boundary: this module explains a score by contrasting a customer against the
cohort scored on the same day. It does not compute the model's attribution for that
customer. Real per-customer attribution would mean SHAP values produced at scoring time and
carried through ml.predictions; the serving container returns a probability and nothing
else. Calling a cohort contrast "why the model said this" would be a fabrication, so every
label this module produces says "indicator", and the UI states the distinction in text.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The band a customer falls into, worst first — this is the display order everywhere.
RISK_BANDS: tuple[str, ...] = ("High", "Medium", "Low")

# Mirrors the COUNTIF(churn_probability >= 0.7) in models/marts/churn_risk_daily.sql. SQL
# and Python cannot share a constant, so the trend line and the table below it would drift
# apart silently if only one were retuned. test_risk.py asserts the two stay equal.
HIGH_RISK_FLOOR_IN_SQL = 0.7


@dataclass(frozen=True)
class Driver:
    """One comparable feature, with the direction that counts as worse."""

    column: str
    label: str
    higher_is_worse: bool
    # How to render the value to a business reader.
    fmt: str


# The features the dashboard is willing to put in front of a business user as a reason to
# look at someone. Everything else in the mart is either identity (tier, region), stakes
# (monthly_value), or model output. Kept short on purpose: an explanation panel that lists
# fifteen features explains nothing.
DRIVERS: tuple[Driver, ...] = (
    Driver("payment_failure_rate_30d", "Payment failures (30d)", True, "percent"),
    Driver("days_since_last_successful_payment", "Days since last successful payment", True, "int"),
    Driver("contact_requests_last_30d", "Support contacts (30d)", True, "int"),
    Driver("avg_engagement_30d", "Engagement score (30d)", False, "float"),
    Driver("events_last_30d", "Activity events (30d)", False, "int"),
    Driver("campaign_participation_rate", "Campaign participation", False, "percent"),
)


def assign_risk_band(
    probability: pd.Series, high_threshold: float, medium_threshold: float
) -> pd.Series:
    """Bucket churn probabilities into the High / Medium / Low display bands."""
    return pd.Series(
        pd.cut(
            probability,
            bins=[-float("inf"), medium_threshold, high_threshold, float("inf")],
            labels=["Low", "Medium", "High"],
            right=False,
        ),
        index=probability.index,
    ).astype("string")


def with_risk_bands(df: pd.DataFrame, high_threshold: float, medium_threshold: float):
    """Return a copy of the snapshot with a `risk_band` column added."""
    out = df.copy()
    out["risk_band"] = assign_risk_band(out["churn_probability"], high_threshold, medium_threshold)
    return out


def cohort_reference(df: pd.DataFrame) -> dict[str, float]:
    """Return the comparison point for each driver across the whole scored snapshot.

    The median, except where the median is degenerate. Payment-failure rate in a healthy
    base has a median of exactly 0, and comparing against 0 makes every customer with a
    single failed charge look like an outlier — the indicator would fire for most of the
    cohort and rank nobody. Where the median sits at zero, a tail percentile is used
    instead, so the feature still separates the customers it is actually unusual for.
    """
    reference: dict[str, float] = {}
    for driver in DRIVERS:
        if driver.column not in df.columns:
            continue
        values = pd.to_numeric(df[driver.column], errors="coerce").dropna()
        if values.empty:
            continue
        median = float(values.median())
        if median > 0:
            reference[driver.column] = median
        else:
            fallback = values.quantile(0.9 if driver.higher_is_worse else 0.1)
            reference[driver.column] = float(fallback)
    return reference


def customer_indicators(
    customer: pd.Series, reference: dict[str, float], ratio: float
) -> list[dict[str, object]]:
    """Return the drivers on which this customer is materially worse than the cohort.

    "Materially" is the `ratio` multiple either side of the cohort reference — a floor that
    exists so the panel does not list six marginal deviations for every customer. The
    returned rows are ordered by how far outside the cohort the customer sits, so the first
    one is the strongest thing a retention owner can say about them.
    """
    indicators: list[dict[str, object]] = []

    for driver in DRIVERS:
        if driver.column not in customer.index:
            continue
        raw = customer[driver.column]

        # A missing days-since-last-successful-payment is not missing data. dbt documents it
        # as "no successful payment on record", which is the single strongest thing that can
        # be true of a paying customer's billing history — surfacing it as a blank cell in a
        # retention worklist would hide exactly the people the list exists to find.
        if pd.isna(raw):
            if driver.column == "days_since_last_successful_payment":
                indicators.append(
                    {
                        "label": driver.label,
                        "value": None,
                        "cohort": reference.get(driver.column),
                        "detail": "never recorded a successful payment",
                        "severity": float("inf"),
                        "fmt": driver.fmt,
                    }
                )
            continue

        value = float(raw)
        cohort = reference.get(driver.column)
        if cohort is None:
            continue

        if driver.higher_is_worse:
            triggered = value >= cohort * ratio if cohort > 0 else value > 0
            severity = value / cohort if cohort > 0 else float("inf")
        else:
            triggered = value <= cohort / ratio if cohort > 0 else False
            severity = cohort / value if value > 0 else float("inf")

        if not triggered:
            continue

        indicators.append(
            {
                "label": driver.label,
                "value": value,
                "cohort": cohort,
                "detail": _describe(value, cohort, driver),
                "severity": severity,
                "fmt": driver.fmt,
            }
        )

    return sorted(indicators, key=lambda i: i["severity"], reverse=True)


def _describe(value: float, cohort: float, driver: Driver) -> str:
    """Phrase one indicator as a comparison a non-technical reader can repeat."""
    if driver.higher_is_worse:
        if cohort <= 0:
            return "above a cohort where the typical customer has none"
        return f"{value / cohort:.1f}× the cohort median of {format_value(cohort, driver.fmt)}"
    if value <= 0:
        return "none at all, against a cohort median of " + format_value(cohort, driver.fmt)
    return f"{cohort / value:.1f}× below the cohort median of {format_value(cohort, driver.fmt)}"


def format_value(value: float | None, fmt: str) -> str:
    """Render a driver value for display."""
    if value is None or pd.isna(value):
        return "—"
    if fmt == "percent":
        return f"{value:.0%}"
    if fmt == "int":
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def portfolio_summary(df: pd.DataFrame, high_threshold: float) -> dict[str, float]:
    """Return the headline numbers for the top-of-page metric row."""
    total = len(df)
    high_risk = df[df["churn_probability"] >= high_threshold]
    return {
        "customers_scored": total,
        "high_risk_customers": len(high_risk),
        "high_risk_share": len(high_risk) / total if total else 0.0,
        # Monthly recurring value carried by the high-risk group, not a forecast of loss:
        # it is what is on the table if none of them are retained, which is the upper bound
        # the UI labels it as.
        "monthly_value_at_risk": float(
            pd.to_numeric(high_risk.get("monthly_value"), errors="coerce").fillna(0).sum()
        )
        if total
        else 0.0,
        "mean_probability": float(df["churn_probability"].mean()) if total else 0.0,
    }
