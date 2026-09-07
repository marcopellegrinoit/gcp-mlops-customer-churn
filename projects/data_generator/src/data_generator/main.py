"""Synthetic customer-event generator that streams batches to BigQuery.

Customer identities are **stable across runs**. Every profile is derived deterministically
from its index — ``uuid5`` for the id, a per-index seeded RNG for its traits — so the
customer that index 7 denotes today is the same customer it denoted yesterday, and its
events accumulate into one history.

This is load-bearing, not a stylistic choice. Every feature in ``customer_features`` is a
window aggregate over a customer's own history: ``avg_transaction_30d``,
``events_last_90d``, ``payment_failure_rate_30d``, ``days_since_last_successful_payment``,
``renewal_count``. An earlier version minted a fresh ``uuid4()`` pool on every run, so no
customer ever received a second event; measured in production, that produced 32,609
customers over 14 days at 1.1 events each with **zero** spanning more than one day. Every
window aggregate collapsed to a constant or a two-value discrete, and ``churned`` — read
from the customer's most recent event — was a single coin flip uncorrelated with any
behaviour, so the model was being fit to noise.

Two further properties follow from stable identities and are needed for the simulation to
behave like a real membership base:

* **Churn is absorbing.** A customer who cancels emits a final ``membership_cancelled``
  event and is never selected again. ``customer_features`` reads the label off the latest
  event, so the label persists and describes the customer rather than the moment.
* **The base grows.** New customers are acquired daily, so the population turns over
  instead of being one fixed cohort ageing in place, and tenure reaches a steady state
  rather than climbing without bound for everyone at once.
"""

import dataclasses
import json
import logging
import os
import random
import uuid
from datetime import UTC, date, datetime, timedelta
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
# Events an active customer can emit. 'membership_cancelled' is excluded because it is now
# reserved for an actual churn: it is emitted only when the churn hazard fires, so it marks
# the end of a customer's history rather than appearing at random mid-life.
_ACTIVE_EVENT_TYPES = [e for e in _EVENT_TYPES if e != "membership_cancelled"]
_PAYMENT_STATUSES = ["success", "failed", "pending"]
_CHANNELS = ["email", "web", "mobile", "direct_mail", "in_person", "phone"]

# Customers on these channels never generate digital engagement signals (MAR).
_OFFLINE_CHANNELS = frozenset({"direct_mail", "phone"})

_ANOMALY_TRANSACTIONS = [-999.0, 0.0, 9999.99]
_ANOMALY_ENGAGEMENT_SCORES = [-1.0, 5.0, 99.9]
_ANOMALY_PAYMENT_ATTEMPTS_MIN = 50
_ANOMALY_PAYMENT_ATTEMPTS_MAX = 200

# Churn probability rises for customers with low engagement or repeated payment failures.
# These are per-event hazards now that churn is absorbing, so they are far lower than the
# per-event draws they replace: a customer sees many events over their life, and a 25%
# chance on each would empty the base within days.
_CHURN_HIGH_ENGAGEMENT_THRESHOLD = 0.15
_CHURN_HIGH_PAYMENT_THRESHOLD = 5
_CHURN_PROB_HIGH = 0.02
_CHURN_PROB_LOW = 0.002

# Fixed namespace for uuid5 customer ids. Changing it re-identifies the entire customer
# base — every existing customer's history would be orphaned and a new population would
# appear overnight, which is exactly the failure this generator was fixed to avoid.
_CUSTOMER_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# The simulated membership base opens here. Elapsed days since this date drive both how
# many customers have been acquired and how long each has been a member, so a redeploy or
# a backfill reconstructs the same population rather than restarting the clock.
_POOL_EPOCH = date(2026, 1, 1)
_ZERO_DAYS = timedelta(0)


@dataclasses.dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    project_id: str
    dataset_id: str
    table_id: str
    batch_size: int = 2000
    anomaly_rate: float = 0.0
    customer_pool_size: int = 10000  # founding base, acquired on the pool epoch
    # New customers acquired per elapsed day. Without acquisition the base only ever shrinks
    # as customers churn, and its tenure distribution climbs without bound.
    daily_acquisitions: int = 25

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
            daily_acquisitions=int(os.getenv("DAILY_ACQUISITIONS", "25")),
        )

    @property
    def table_ref(self) -> str:
        """Return the fully-qualified BigQuery table reference."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"


@dataclasses.dataclass(frozen=True)
class CustomerProfile:
    """Stable per-customer traits that persist across all their events, in every run."""

    customer_id: str
    preferred_channel: str
    is_offline_only: bool  # True → engagement_score always None  (MAR)
    always_fails_payments: bool  # True → payment_status never 'success' (MNAR)
    tenure_days_at_join: int | None  # None for ~3% of customers               (MCAR)
    membership_tier: str
    region: str
    joined_on_day: int  # days after _POOL_EPOCH that this customer was acquired


def _create_customer_profile(index: int, joined_on_day: int) -> CustomerProfile:
    """Build customer `index`'s traits deterministically, identically on every run.

    Seeding the RNG with the index (not with run-level entropy) is what makes the profile
    reproducible: nothing is persisted between runs, yet index 7 always denotes the same
    customer with the same channel, payment behaviour and tier.

    Tier and region live here rather than being drawn per event. They are properties of the
    customer, and drawing them per event made a customer appear in a different tier and
    region on every event — with customer_features reading whichever the latest event
    happened to carry, both columns were noise.
    """
    rng = np.random.default_rng(index)
    channel = str(rng.choice(_CHANNELS))
    return CustomerProfile(
        customer_id=str(uuid.uuid5(_CUSTOMER_NAMESPACE, f"customer-{index}")),
        preferred_channel=channel,
        is_offline_only=channel in _OFFLINE_CHANNELS,
        always_fails_payments=bool(rng.random() < 0.08),
        tenure_days_at_join=None if rng.random() < 0.03 else int(rng.integers(1, 3651)),
        membership_tier=str(rng.choice(_TIERS)),
        region=str(rng.choice(_REGIONS)),
        joined_on_day=joined_on_day,
    )


def _create_customer_pool(
    initial_size: int, daily_acquisitions: int, days_elapsed: int
) -> dict[str, CustomerProfile]:
    """Return every customer acquired up to `days_elapsed` after the pool epoch.

    The first `initial_size` customers are the founding base; each subsequent day adds
    `daily_acquisitions` more. Because index determines identity, yesterday's pool is a
    strict prefix of today's — existing customers keep their ids and their history, and
    only the new arrivals are new.
    """
    total = initial_size + daily_acquisitions * max(days_elapsed, 0)
    profiles = {}
    for index in range(total):
        joined_on_day = (
            0 if index < initial_size else (index - initial_size) // max(daily_acquisitions, 1)
        )
        profile = _create_customer_profile(index, joined_on_day)
        profiles[profile.customer_id] = profile
    return profiles


def _days_since_epoch(today: date | None = None) -> int:
    """Days elapsed since the simulated membership base opened."""
    return max((today or datetime.now(tz=UTC).date()) - _POOL_EPOCH, _ZERO_DAYS).days


def _generate_event(
    customer_pool: dict[str, CustomerProfile],
    rng: np.random.Generator,
    anomaly: bool = False,
    profile: CustomerProfile | None = None,
    days_elapsed: int | None = None,
) -> dict[str, Any]:
    """Build one synthetic customer event, optionally with injected out-of-range values.

    Returns an event whose ``churned`` flag, when true, is this customer's cancellation —
    run() removes them from the pool afterwards so it is also their last event.
    """
    profile = profile or random.choice(list(customer_pool.values()))
    days_elapsed = _days_since_epoch() if days_elapsed is None else days_elapsed
    tier = profile.membership_tier
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
        # A cancellation is the event that ends the relationship, so the two have to agree:
        # customer_features derives renewal_count and the label from these, and a customer
        # whose final event is churned=true but typed 'transaction' is incoherent.
        "event_type": "membership_cancelled" if churned else random.choice(_ACTIVE_EVENT_TYPES),
        "region": profile.region,
        "membership_tier": tier,
        "monthly_transaction": monthly_transaction,
        "payment_status": payment_status,
        "payment_attempts_last_30d": payment_attempts,
        "engagement_score": engagement_score,
        "member_since_days": _member_since_days(profile, days_elapsed),
        "contact_requests_last_30d": int(max(0, rng.poisson(0.3))),
        "churned": churned,
        "channel": profile.preferred_channel,
    }
    return {**core, "raw_payload": json.dumps(core), "anomaly_injected": anomaly}


def _member_since_days(profile: CustomerProfile, days_elapsed: int) -> int | None:
    """Tenure at the moment of this event: what they joined with, plus time since.

    Tenure that never advances is not tenure. Because it grows for everyone at once, the
    population-level distribution would climb without bound if the base were closed — daily
    acquisition of new (low-tenure) customers is what keeps it at a steady state instead of
    turning member_since_days into a permanent source of drift.
    """
    if profile.tenure_days_at_join is None:
        return None  # MCAR: tenure unknown for ~3% of customers, and stays unknown
    return profile.tenure_days_at_join + max(days_elapsed - profile.joined_on_day, 0)


def run(config: Config, client: bigquery.Client) -> None:
    """Generate a batch of events for the active customer base and stream them to BigQuery."""
    rng = np.random.default_rng()
    days_elapsed = _days_since_epoch()
    customer_pool = _create_customer_pool(
        config.customer_pool_size, config.daily_acquisitions, days_elapsed
    )

    already_churned = _fetch_churned_customers(client, config.table_ref)
    active = {cid: p for cid, p in customer_pool.items() if cid not in already_churned}
    if not active:
        log.warning("Every customer in the pool has churned; no events generated.")
        return

    events = _generate_batch(active, rng, config, days_elapsed)

    errors = client.insert_rows_json(config.table_ref, events)
    if errors:
        raise RuntimeError(f"BigQuery streaming insert errors: {errors}")

    anomaly_count = sum(1 for e in events if e["anomaly_injected"])
    churn_count = sum(1 for e in events if e["churned"])
    log.info(
        "Inserted %d events into %s for %d active customers "
        "(%d churned this run, %d already churned, %d with anomaly injected)",
        len(events),
        config.table_ref,
        len(active),
        churn_count,
        len(already_churned),
        anomaly_count,
    )


def _generate_batch(
    active: dict[str, CustomerProfile],
    rng: np.random.Generator,
    config: Config,
    days_elapsed: int,
) -> list[dict[str, Any]]:
    """Generate one run's events, retiring each customer at the moment they churn.

    Selecting the customer here rather than inside _generate_event is what makes churn
    absorbing: once a customer's event comes back flagged churned, they are dropped from the
    selectable set, so the cancellation is genuinely their last event and no later event
    can contradict it.
    """
    selectable = dict(active)
    events: list[dict[str, Any]] = []

    for _ in range(config.batch_size):
        if not selectable:
            break
        profile = random.choice(list(selectable.values()))
        event = _generate_event(
            selectable,
            rng,
            anomaly=random.random() < config.anomaly_rate,
            profile=profile,
            days_elapsed=days_elapsed,
        )
        events.append(event)
        if event["churned"]:
            del selectable[profile.customer_id]

    return events


def _fetch_churned_customers(client: bigquery.Client, table_ref: str) -> set[str]:
    """Return every customer who has already cancelled, so they are never selected again.

    Churn is state, and this is the only place it is read back rather than derived — a
    customer's history is the record of whether they are still a member. A read failure
    degrades to "nobody has churned yet", which keeps generating events for customers who
    have actually left; that is a visible data problem rather than a silent one, and it is
    preferable to failing the run and producing no events at all.
    """
    try:
        rows = client.query(
            f"SELECT DISTINCT customer_id FROM `{table_ref}` WHERE churned"
        ).result()
        return {row["customer_id"] for row in rows}
    except Exception:
        log.exception("Could not read churned customers; treating the whole pool as active.")
        return set()


def main() -> None:
    """Entry point: load config from env, build BigQuery client, and run one generation batch."""
    config = Config.from_env()
    client = bigquery.Client(project=config.project_id)
    run(config, client)


if __name__ == "__main__":
    main()
