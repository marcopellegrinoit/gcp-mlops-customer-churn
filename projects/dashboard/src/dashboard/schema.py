"""What the dashboard requires of the two dbt marts it reads.

This is a **consumer-driven** contract, and the direction matters. These models declare the
columns *this app depends on*, not everything `churn_risk_current` and `churn_risk_daily`
emit — so dbt stays free to add columns without touching the dashboard, while dropping or
renaming one the UI reads fails at the query boundary with a message naming the column and
the view. Without that, the same change surfaces as a `KeyError` several frames deep inside
an Altair encoding, or worse, as an indicator panel that silently stops listing a driver
(`risk.cohort_reference` and `customer_indicators` both skip columns that are absent).

The model-output columns are not redeclared here: `CHURN_PROBABILITY_FIELD` and
`CHURN_PREDICTION_FIELD` come from `data_contracts`, the same pure-pydantic package the
serving container uses to *write* them, so the two cannot drift apart. The mart row shapes
themselves stay local — they describe views owned by dbt and read by nothing else, and
`test_schema.py` asserts them against the mart SQL directly, the same reconciliation-by-test
approach `risk.HIGH_RISK_FLOOR_IN_SQL` uses for the band threshold.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, TypeVar

import pandas as pd
from data_contracts import CHURN_PREDICTION_FIELD, CHURN_PROBABILITY_FIELD
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "CHURN_PREDICTION_FIELD",
    "CHURN_PROBABILITY_FIELD",
    "ChurnRiskDailyRow",
    "ChurnRiskRow",
    "MartContractError",
    "require_columns",
    "validate_trend",
]


class MartContractError(RuntimeError):
    """A mart no longer provides what the dashboard reads from it."""


def _missing_to_none(value: Any) -> Any:
    """Normalise pandas' several spellings of NULL to None before validation.

    A BigQuery NULL does not survive `to_dataframe()` as None. In a float column it becomes
    `float('nan')`, and in a nullable-extension column `pd.NA` — and neither is what an
    `X | None` field accepts. Worse, NaN passes an `is not None` check and then fails every
    bound silently-plausibly: `nan <= 1.0` is False, so a legitimately NULL
    `predicted_churn_rate` would be reported as a value out of range rather than as absent.

    Applied to every optional field below, so "the mart returned NULL" reads as None
    everywhere in this module regardless of which dtype pandas chose for the column.
    """
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    return value


_T = TypeVar("_T")
# An optional mart column: NULL in any of pandas' representations arrives as None.
type Nullable[_T] = Annotated[_T | None, BeforeValidator(_missing_to_none)]


class ChurnRiskRow(BaseModel):
    """One scored customer, as the dashboard reads them out of `features.churn_risk_current`.

    Nullable columns are nullable in the mart: `days_since_last_successful_payment` is NULL
    for a customer with no successful payment on record — which `risk.customer_indicators`
    treats as the strongest signal it has rather than as missing data — and the engagement
    columns are NULL for offline-only customers.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    # --- Identity and model output -------------------------------------------------------
    customer_id: str
    # The serving container writes this as data_contracts.ChurnPrediction.churn_probability;
    # a test pins the name below to that model's own field so the two cannot drift. Bounded
    # here for the same reason it is bounded there.
    churn_probability: float = Field(ge=0.0, le=1.0)
    model_version: str
    snapshot_date: date

    # --- Segmentation the worklist filters on --------------------------------------------
    membership_tier: Nullable[str] = None
    region: Nullable[str] = None
    preferred_channel: Nullable[str] = None
    member_since_days: Nullable[int] = None

    # --- Stakes ---------------------------------------------------------------------------
    monthly_value: Nullable[float] = None

    # --- Cohort-comparison drivers (risk.DRIVERS) ------------------------------------------
    payment_failure_rate_30d: Nullable[float] = None
    days_since_last_successful_payment: Nullable[float] = None
    contact_requests_last_30d: Nullable[int] = None
    avg_engagement_30d: Nullable[float] = None
    events_last_30d: Nullable[int] = None
    campaign_participation_rate: Nullable[float] = None


class ChurnRiskDailyRow(BaseModel):
    """One scored day of `features.churn_risk_daily`, behind the trend strip."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    snapshot_date: date
    customers_scored: int = Field(ge=0)
    high_risk_customers: int = Field(ge=0)
    # SAFE_DIVIDE returns NULL rather than erroring on a day that scored nobody, so this is
    # genuinely nullable — and the chart has to leave a gap there rather than plot a zero.
    predicted_churn_rate: Nullable[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    # More than one on a single day means a champion was promoted mid-day: the step it puts
    # in the trend line is a model change, not a customer change, and the tooltip says so.
    model_versions_used: int = Field(ge=0)


_DAILY_ROWS = TypeAdapter(list[ChurnRiskDailyRow])


def require_columns(frame: pd.DataFrame, model: type[BaseModel], source: str) -> pd.DataFrame:
    """Assert the frame carries every column `model` declares; return it unchanged.

    Used for the per-customer snapshot, where a full per-row validation would be a pass over
    the whole scored base on every cache refresh to re-check guarantees that already hold:
    a BigQuery view fixes each column's type for every row at once, and `churn_probability`
    was already bounded to [0, 1] by `ChurnPrediction` in the serving container before it
    reached `ml.predictions`. What can actually change between deploys is the *set* of
    columns, and that is what this checks — in time proportional to the columns, not the rows.

    A frame carrying no columns at all is the "nothing scored yet" shape, which the app
    already renders as an explicit empty state; an empty result set that still carries the
    view's schema is checked normally, so a dropped column is caught even on a quiet day.
    """
    if frame.columns.empty:
        return frame

    missing = sorted(set(model.model_fields) - set(frame.columns))
    if missing:
        raise MartContractError(
            f"{source} is missing column(s) the dashboard reads: {', '.join(missing)}. "
            "The mart and this app have diverged — check "
            "projects/dbt_transform/models/marts/ against dashboard.schema."
        )
    return frame


def validate_trend(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate every row of the trend frame; return it unchanged.

    Full per-row validation is affordable here in a way it is not for the snapshot: this
    frame is one row per scored day, bounded by `ml.predictions`' retention rather than by
    the size of the customer base. It is also worth doing, because unlike the snapshot these
    are SQL aggregates — a view rewrite that changes what `predicted_churn_rate` divides by
    produces a per-row value that is wrong rather than a column that is absent, and a rate
    above 1 plotted on a percentage axis reads as a plausible number.
    """
    require_columns(frame, ChurnRiskDailyRow, source)
    if frame.empty:
        return frame

    try:
        _DAILY_ROWS.validate_python(frame.to_dict("records"))
    except ValidationError as error:
        raise MartContractError(
            f"{source} returned rows the dashboard cannot read: {error}"
        ) from error
    return frame
