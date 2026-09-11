"""The synthetic CDC event schema — the shape every downstream stage is derived from.

This is the platform's source table: dbt reads raw.activity_cdc to build
features.customer_features, which trains the model, which produces the predictions the
drift monitor watches. A field renamed or retyped here changes all of it, so the row is
declared once, here, and mirrors raw.activity_cdc's schema in iac/config/bigquery.yaml
field for field.

The enums are the categorical vocabulary of the simulation. Declaring them as types rather
than as loose string lists is what makes ``membership_tier`` a closed set the generator
cannot accidentally widen — the model's categorical splits are frozen against exactly
these values at training time (ml_common.preprocess.categorical_categories), and a value
outside them silently becomes NaN at inference.

**Numeric ranges are deliberately unconstrained.** The generator injects out-of-range
values on purpose — negative transactions, engagement scores above 1, 200 payment attempts
— to exercise the drift and data-quality machinery downstream. Bounding them here would
reject the very rows this platform exists to detect. Validation covers shape and type;
plausibility is the data-quality gate's job, further down.
"""

import json
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class MembershipTier(StrEnum):
    """Membership tiers, ordered by the transaction size each implies."""

    SUPPORTER = "supporter"
    FRIEND = "friend"
    CHAMPION = "champion"
    GUARDIAN = "guardian"


class Region(StrEnum):
    """Geographic regions the membership base spans."""

    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    APAC = "apac"
    LATAM = "latam"


class Channel(StrEnum):
    """Acquisition and communication channels.

    DIRECT_MAIL and PHONE are the offline ones: customers on those channels emit no digital
    engagement signal at all, which is the platform's MAR missingness (see main._OFFLINE_CHANNELS).
    """

    EMAIL = "email"
    WEB = "web"
    MOBILE = "mobile"
    DIRECT_MAIL = "direct_mail"
    IN_PERSON = "in_person"
    PHONE = "phone"


class EventType(StrEnum):
    """What a customer did.

    MEMBERSHIP_CANCELLED is reserved for an actual churn — it is emitted only when the churn
    hazard fires, so it marks the end of a customer's history rather than appearing at random
    mid-life. customer_features derives both renewal_count and the label from these.
    """

    TRANSACTION = "transaction"
    MEMBERSHIP_RENEWAL = "membership_renewal"
    CAMPAIGN_ACTION = "campaign_action"
    EMAIL_ENGAGEMENT = "email_engagement"
    EVENT_ATTENDED = "event_attended"
    CONTACT_REQUEST = "contact_request"
    MEMBERSHIP_CANCELLED = "membership_cancelled"


class PaymentStatus(StrEnum):
    """Outcome of the customer's most recent payment attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class ActivityEvent(BaseModel):
    """One customer activity event, as the source system would emit it.

    Nullable fields are nullable for a reason the model depends on: engagement_score is
    absent for offline-only customers (MAR), member_since_days for the ~3% whose tenure was
    never recorded (MCAR), and the platform treats a shift in how often either is null as a
    real population change rather than noise to drop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    event_id: str
    event_timestamp: datetime
    customer_id: str
    event_type: EventType
    region: Region
    membership_tier: MembershipTier
    monthly_transaction: float
    payment_status: PaymentStatus
    payment_attempts_last_30d: int
    engagement_score: float | None
    member_since_days: int | None
    contact_requests_last_30d: int
    churned: bool
    channel: Channel

    def to_cdc_row(self, anomaly_injected: bool) -> "ActivityCdcRow":
        """Wrap this event as the CDC row that lands in raw.activity_cdc."""
        return ActivityCdcRow(
            **self.model_dump(),
            raw_payload=json.dumps(self.model_dump(mode="json")),
            anomaly_injected=anomaly_injected,
        )


class ActivityCdcRow(ActivityEvent):
    """One row of raw.activity_cdc: the event, plus what the CDC pipeline records about it.

    raw_payload is the event as it arrived, before parsing — retained because a real CDC
    landing table keeps the original document, so a field the schema does not yet model is
    still recoverable. anomaly_injected marks the rows this simulation corrupted on purpose,
    so a drift alarm can be traced back to a deliberate injection rather than investigated
    as a production incident.
    """

    raw_payload: str
    anomaly_injected: bool

    def to_bigquery_row(self) -> dict:
        """Render this row as the JSON-typed dict insert_rows_json streams to BigQuery."""
        return self.model_dump(mode="json")
