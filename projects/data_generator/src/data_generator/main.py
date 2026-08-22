"""Synthetic customer-event generator that streams batches to BigQuery."""

import dataclasses
import json
import logging
import os
import random
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from google.cloud import bigquery
from obs_common.logging import configure_logging

configure_logging()
log = logging.getLogger(__name__)

_TIERS = ["supporter", "friend", "champion", "guardian"]
_TIER_BASE_TRANSACTION: dict[str, float] = {
    "supporter": 10.0,
    "friend": 25.0,
    "champion": 50.0,
    "guardian": 100.0,
}
_REGIONS = ["us-east", "us-west", "eu-west", "eu-central", "apac", "latam"]
_EVENT_TYPES = [
    "transaction",
    "membership_renewal",
    "campaign_action",
    "email_engagement",
    "event_attended",
    "contact_request",
    "membership_cancelled",
]
_PAYMENT_STATUSES = ["success", "failed", "pending"]
_CHANNELS = ["email", "web", "mobile", "direct_mail", "in_person", "phone"]

# Customers on these channels never generate digital engagement signals (MAR).
_OFFLINE_CHANNELS = frozenset({"direct_mail", "phone"})

_ANOMALY_TRANSACTIONS = [-999.0, 0.0, 9999.99]
_ANOMALY_ENGAGEMENT_SCORES = [-1.0, 5.0, 99.9]
_ANOMALY_PAYMENT_ATTEMPTS_MIN = 50
_ANOMALY_PAYMENT_ATTEMPTS_MAX = 200

# Churn probability rises for customers with low engagement or repeated payment failures.
_CHURN_HIGH_ENGAGEMENT_THRESHOLD = 0.15
_CHURN_HIGH_PAYMENT_THRESHOLD = 5
_CHURN_PROB_HIGH = 0.25
_CHURN_PROB_LOW = 0.04


@dataclasses.dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    project_id: str
    dataset_id: str
    table_id: str
    batch_size: int = 2000
    anomaly_rate: float = 0.0
    customer_pool_size: int = 10000

    @classmethod
    def from_env(cls) -> "Config":
        """Build Config from environment variables; raises KeyError for missing required vars."""
        return cls(
            project_id=os.environ["BQ_PROJECT_ID"],
            dataset_id=os.environ["BQ_DATASET_ID"],
            table_id=os.environ["BQ_TABLE_ID"],
            batch_size=int(os.getenv("BATCH_SIZE", "2000")),
            anomaly_rate=float(os.getenv("ANOMALY_RATE", "0.0")),
            customer_pool_size=int(os.getenv("USER_POOL_SIZE", "10000")),
        )

    @property
    def table_ref(self) -> str:
        """Return the fully-qualified BigQuery table reference."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"


@dataclasses.dataclass(frozen=True)
class CustomerProfile:
    """Stable per-customer traits that persist across all their events."""

    customer_id: str
    preferred_channel: str
    is_offline_only: bool  # True → engagement_score always None  (MAR)
    always_fails_payments: bool  # True → payment_status never 'success' (MNAR)
    member_since_days: int | None  # None for ~3% of customers               (MCAR)


def _create_customer_pool(size: int, rng: np.random.Generator) -> dict[str, CustomerProfile]:
    """Create a pool of customers with stable traits assigned once at startup."""
    profiles: dict[str, CustomerProfile] = {}
    for _ in range(size):
        customer_id = str(uuid.uuid4())
        channel = str(rng.choice(_CHANNELS))
        profiles[customer_id] = CustomerProfile(
            customer_id=customer_id,
            preferred_channel=channel,
            is_offline_only=channel in _OFFLINE_CHANNELS,
            always_fails_payments=bool(rng.random() < 0.08),
            member_since_days=None if rng.random() < 0.03 else int(rng.integers(1, 3651)),
        )
    return profiles


def _generate_event(
    customer_pool: dict[str, CustomerProfile],
    rng: np.random.Generator,
    anomaly: bool = False,
) -> dict[str, Any]:
    """Build one synthetic customer event, optionally with injected out-of-range values."""
    profile = random.choice(list(customer_pool.values()))
    tier = random.choice(_TIERS)
    base_transaction = _TIER_BASE_TRANSACTION[tier]

    if anomaly:
        monthly_transaction = round(random.choice(_ANOMALY_TRANSACTIONS), 2)
        engagement_score = round(random.choice(_ANOMALY_ENGAGEMENT_SCORES), 4)
        payment_attempts = random.randint(
            _ANOMALY_PAYMENT_ATTEMPTS_MIN, _ANOMALY_PAYMENT_ATTEMPTS_MAX
        )
        payment_status = random.choice(_PAYMENT_STATUSES)
    else:
        monthly_transaction = round(
            float(np.clip(rng.normal(base_transaction, base_transaction * 0.1 + 0.5), 0.0, None)), 2
        )
        # Offline customers (direct_mail, phone) produce no digital engagement signal.
        engagement_score = (
            None if profile.is_offline_only else round(float(np.clip(rng.beta(2, 5), 0.0, 1.0)), 4)
        )
        payment_attempts = int(max(0, rng.poisson(1.5)))
        payment_statuses = (
            ["failed", "pending"] if profile.always_fails_payments else _PAYMENT_STATUSES
        )
        payment_status = random.choice(payment_statuses)

    # Anomaly churn uses only the injected numeric values (profile traits don't apply).
    if anomaly:
        high_churn = (
            engagement_score < _CHURN_HIGH_ENGAGEMENT_THRESHOLD
            or payment_attempts > _CHURN_HIGH_PAYMENT_THRESHOLD
        )
    else:
        high_churn = (
            engagement_score is None  # missing engagement is a risk signal
            or engagement_score < _CHURN_HIGH_ENGAGEMENT_THRESHOLD
            or payment_attempts > _CHURN_HIGH_PAYMENT_THRESHOLD
            # MNAR: null days_since_last_payment correlates with churn
            or profile.always_fails_payments
        )
    churned = random.random() < (_CHURN_PROB_HIGH if high_churn else _CHURN_PROB_LOW)

    core: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "event_timestamp": datetime.now(tz=UTC).isoformat(),
        "customer_id": profile.customer_id,
        "event_type": random.choice(_EVENT_TYPES),
        "region": random.choice(_REGIONS),
        "membership_tier": tier,
        "monthly_transaction": monthly_transaction,
        "payment_status": payment_status,
        "payment_attempts_last_30d": payment_attempts,
        "engagement_score": engagement_score,
        "member_since_days": profile.member_since_days,
        "contact_requests_last_30d": int(max(0, rng.poisson(0.3))),
        "churned": churned,
        "channel": profile.preferred_channel,
    }
    return {**core, "raw_payload": json.dumps(core), "anomaly_injected": anomaly}


def run(config: Config, client: bigquery.Client) -> None:
    """Generate a batch of events and stream-insert them into BigQuery."""
    rng = np.random.default_rng()
    customer_pool = _create_customer_pool(config.customer_pool_size, rng)

    events = [
        _generate_event(customer_pool, rng, anomaly=random.random() < config.anomaly_rate)
        for _ in range(config.batch_size)
    ]

    errors = client.insert_rows_json(config.table_ref, events)
    if errors:
        raise RuntimeError(f"BigQuery streaming insert errors: {errors}")

    anomaly_count = sum(1 for e in events if e["anomaly_injected"])
    log.info(
        "Inserted %d events into %s (%d with anomaly injected)",
        len(events),
        config.table_ref,
        anomaly_count,
    )


def main() -> None:
    """Entry point: load config from env, build BigQuery client, and run one generation batch."""
    config = Config.from_env()
    client = bigquery.Client(project=config.project_id)
    run(config, client)


if __name__ == "__main__":
    main()
