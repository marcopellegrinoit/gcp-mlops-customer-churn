"""Unit tests for _generate_event()."""

import json
import uuid
from datetime import datetime

import numpy as np
from data_generator.main import (
    _ANOMALY_ENGAGEMENT_SCORES,
    _ANOMALY_PAYMENT_ATTEMPTS_MAX,
    _ANOMALY_PAYMENT_ATTEMPTS_MIN,
    _ANOMALY_TRANSACTIONS,
    _CHURN_HIGH_ENGAGEMENT_THRESHOLD,
    _CHURN_HIGH_PAYMENT_THRESHOLD,
    _CHURN_PROB_HIGH,
    _CHURN_PROB_LOW,
    CustomerProfile,
    _create_customer_pool,
    _generate_event,
)
from data_generator.schema import (
    ActivityCdcRow,
    Channel,
    EventType,
    MembershipTier,
    PaymentStatus,
    Region,
)

_ALL_FIELDS = frozenset(
    {
        "event_id",
        "event_timestamp",
        "customer_id",
        "event_type",
        "region",
        "membership_tier",
        "monthly_transaction",
        "payment_status",
        "payment_attempts_last_30d",
        "engagement_score",
        "member_since_days",
        "contact_requests_last_30d",
        "churned",
        "channel",
        "raw_payload",
        "anomaly_injected",
    }
)
_CORE_FIELDS = _ALL_FIELDS - {"raw_payload", "anomaly_injected"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_row_declares_exactly_the_bigquery_columns(self):
        # raw.activity_cdc's schema in iac/config/bigquery.yaml, field for field. A column
        # added on one side and not the other fails the streaming insert at runtime.
        assert set(ActivityCdcRow.model_fields) == _ALL_FIELDS

    def test_normal_event_serializes_every_column(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=False)
        assert set(event.to_bigquery_row()) == _ALL_FIELDS

    def test_anomaly_event_serializes_every_column(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=True)
        assert set(event.to_bigquery_row()) == _ALL_FIELDS

    def test_event_id_is_valid_uuid(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        uuid.UUID(event.event_id)  # raises ValueError if malformed

    def test_event_timestamp_is_utc(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.event_timestamp.tzinfo is not None

    def test_serialized_timestamp_round_trips(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        parsed = datetime.fromisoformat(event.to_bigquery_row()["event_timestamp"])
        assert parsed.tzinfo is not None

    def test_churned_is_bool(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert isinstance(event.churned, bool)

    def test_anomaly_injected_is_bool(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert isinstance(event.anomaly_injected, bool)


# ---------------------------------------------------------------------------
# Enum constraints
# ---------------------------------------------------------------------------


class TestEnumConstraints:
    def test_event_type_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.event_type in set(EventType)

    def test_region_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.region in set(Region)

    def test_membership_tier_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.membership_tier in set(MembershipTier)

    def test_payment_status_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.payment_status in set(PaymentStatus)

    def test_channel_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.channel in set(Channel)

    def test_customer_id_from_pool(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event.customer_id in customer_pool


# ---------------------------------------------------------------------------
# Normal event value ranges
# ---------------------------------------------------------------------------


class TestNormalValues:
    def test_transaction_is_non_negative(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            assert event.monthly_transaction >= 0.0

    def test_engagement_score_in_unit_interval(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            score = event.engagement_score
            assert score is None or 0.0 <= score <= 1.0

    def test_payment_attempts_non_negative(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            assert event.payment_attempts_last_30d >= 0

    def test_contact_requests_non_negative(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            assert event.contact_requests_last_30d >= 0

    def test_member_since_days_in_range(self, customer_pool):
        # Tenure now advances with elapsed time, so the ceiling is the join-time tenure
        # plus however long the customer has been in the base.
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False, days_elapsed=0)
            days = event.member_since_days
            assert days is None or 1 <= days <= 3650

    def test_member_since_days_advances_with_elapsed_time(self, customer_pool, rng):
        profile = next(p for p in customer_pool.values() if p.tenure_days_at_join is not None)
        at_join = _generate_event(customer_pool, rng, profile=profile, days_elapsed=0)
        later = _generate_event(customer_pool, rng, profile=profile, days_elapsed=100)
        assert later.member_since_days == at_join.member_since_days + 100

    def test_anomaly_injected_false(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=False)
        assert event.anomaly_injected is False


# ---------------------------------------------------------------------------
# Anomaly event values
# ---------------------------------------------------------------------------


class TestAnomalyValues:
    def test_anomaly_injected_true(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=True)
        assert event.anomaly_injected is True

    def test_anomaly_transaction_from_known_set(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True).monthly_transaction
            for _ in range(100)
        }
        assert observed.issubset(set(_ANOMALY_TRANSACTIONS))

    def test_anomaly_engagement_from_known_set(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True).engagement_score for _ in range(100)
        }
        assert observed.issubset(set(_ANOMALY_ENGAGEMENT_SCORES))

    def test_anomaly_payment_attempts_in_range(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(100):
            event = _generate_event(customer_pool, rng, anomaly=True)
            assert (
                _ANOMALY_PAYMENT_ATTEMPTS_MIN
                <= event.payment_attempts_last_30d
                <= _ANOMALY_PAYMENT_ATTEMPTS_MAX
            )

    def test_anomaly_transactions_cover_all_values(self, customer_pool):
        """All three anomaly transaction values should appear across enough trials."""
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True).monthly_transaction
            for _ in range(300)
        }
        assert observed == set(_ANOMALY_TRANSACTIONS)

    def test_anomaly_engagement_covers_all_values(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True).engagement_score for _ in range(300)
        }
        assert observed == set(_ANOMALY_ENGAGEMENT_SCORES)


# ---------------------------------------------------------------------------
# raw_payload integrity
# ---------------------------------------------------------------------------


class TestRawPayload:
    def test_raw_payload_is_valid_json(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        parsed = json.loads(event.raw_payload)
        assert isinstance(parsed, dict)

    def test_raw_payload_contains_core_fields(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event.raw_payload)
        assert _CORE_FIELDS.issubset(payload.keys())

    def test_raw_payload_excludes_itself(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event.raw_payload)
        assert "raw_payload" not in payload

    def test_raw_payload_excludes_anomaly_injected(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event.raw_payload)
        assert "anomaly_injected" not in payload

    def test_raw_payload_values_consistent_with_event(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event.raw_payload)
        assert payload["event_id"] == event.event_id
        assert payload["customer_id"] == event.customer_id
        assert payload["monthly_transaction"] == event.monthly_transaction
        assert payload["engagement_score"] == event.engagement_score


# ---------------------------------------------------------------------------
# Churn logic
# ---------------------------------------------------------------------------


class TestChurnLogic:
    def test_churn_constants_high_greater_than_low(self):
        assert _CHURN_PROB_HIGH > _CHURN_PROB_LOW

    def test_churn_thresholds_defined(self):
        assert _CHURN_HIGH_ENGAGEMENT_THRESHOLD == 0.15
        assert _CHURN_HIGH_PAYMENT_THRESHOLD == 5

    def test_high_risk_anomaly_events_churn_at_elevated_rate(self, customer_pool):
        """Anomaly events exceed the payment threshold, so they take the high churn hazard.

        The hazard is now per-event on an absorbing state rather than a per-event coin flip
        that a customer could re-roll forever, so the rate is deliberately low — a customer
        sees many events over their life, and the old 25% would empty the base in days.
        """
        rng = np.random.default_rng(seed=7)
        trials = 4000
        churned = sum(
            _generate_event(customer_pool, rng, anomaly=True).churned for _ in range(trials)
        )
        churn_rate = churned / trials
        assert 0.5 * _CHURN_PROB_HIGH < churn_rate < 2.0 * _CHURN_PROB_HIGH, (
            f"Expected ~{_CHURN_PROB_HIGH:.1%} churn for high-risk events, got {churn_rate:.1%}"
        )

    def test_low_risk_events_churn_at_low_rate(self, customer_pool):
        """Normal events with safe engagement/payment values take the low churn hazard.

        Needs far more samples than the high-risk case: the low hazard is a few tenths of a
        percent per event, so a few hundred draws would frequently contain no churn at all
        and the test would be measuring nothing.
        """
        rng = np.random.default_rng(seed=3)
        safe_events = []
        for _ in range(60000):
            e = _generate_event(customer_pool, rng, anomaly=False)
            profile = customer_pool[e.customer_id]
            if (
                e.engagement_score is not None
                and e.engagement_score >= _CHURN_HIGH_ENGAGEMENT_THRESHOLD
                and e.payment_attempts_last_30d <= _CHURN_HIGH_PAYMENT_THRESHOLD
                and not profile.always_fails_payments
            ):
                safe_events.append(e)
            if len(safe_events) >= 8000:
                break

        churn_rate = sum(e.churned for e in safe_events) / len(safe_events)
        assert 0.0 < churn_rate < 3.0 * _CHURN_PROB_LOW, (
            f"Expected ~{_CHURN_PROB_LOW:.1%} churn for low-risk events, got {churn_rate:.2%}"
        )


# ---------------------------------------------------------------------------
# Missing value behaviour (MAR / MNAR / MCAR)
# ---------------------------------------------------------------------------


def _make_pool(*profiles: CustomerProfile) -> dict[str, CustomerProfile]:
    return {p.customer_id: p for p in profiles}


def _offline_profile() -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(uuid.uuid4()),
        preferred_channel="direct_mail",
        is_offline_only=True,
        always_fails_payments=False,
        tenure_days_at_join=365,
        membership_tier="friend",
        region="eu-west",
        joined_on_day=0,
    )


def _failing_profile() -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(uuid.uuid4()),
        preferred_channel="email",
        is_offline_only=False,
        always_fails_payments=True,
        tenure_days_at_join=365,
        membership_tier="friend",
        region="eu-west",
        joined_on_day=0,
    )


class TestMissingValues:
    def test_offline_customer_has_null_engagement(self):
        """Offline-only customers always produce engagement_score=None (MAR)."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(_offline_profile())
        for _ in range(30):
            event = _generate_event(pool, rng, anomaly=False)
            assert event.engagement_score is None

    def test_online_customer_always_has_engagement(self):
        """Online customers always produce a numeric engagement_score."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(
            CustomerProfile(
                customer_id=str(uuid.uuid4()),
                preferred_channel="email",
                is_offline_only=False,
                always_fails_payments=False,
                tenure_days_at_join=365,
                membership_tier="friend",
                region="eu-west",
                joined_on_day=0,
            )
        )
        for _ in range(30):
            event = _generate_event(pool, rng, anomaly=False)
            assert event.engagement_score is not None

    def test_always_fails_customer_never_has_successful_payment(self):
        """Customers flagged always_fails_payments never produce payment_status='success' (MNAR)."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(_failing_profile())
        for _ in range(50):
            event = _generate_event(pool, rng, anomaly=False)
            assert event.payment_status != PaymentStatus.SUCCESS

    def test_member_since_days_can_be_null(self):
        """Customers with no recorded start date produce member_since_days=None (MCAR)."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(
            CustomerProfile(
                customer_id=str(uuid.uuid4()),
                preferred_channel="web",
                is_offline_only=False,
                always_fails_payments=False,
                tenure_days_at_join=None,
                membership_tier="friend",
                region="eu-west",
                joined_on_day=0,
            )
        )
        for _ in range(10):
            event = _generate_event(pool, rng, anomaly=False)
            assert event.member_since_days is None

    def test_anomaly_overrides_offline_engagement(self):
        """Anomaly injection always sets a numeric engagement_score, even for offline customers."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(_offline_profile())
        for _ in range(30):
            event = _generate_event(pool, rng, anomaly=True)
            assert event.engagement_score in _ANOMALY_ENGAGEMENT_SCORES

    def test_offline_pool_produces_null_engagement_at_expected_rate(self):
        """With ~33% offline channels, roughly a third of events should have null engagement."""
        pool = _create_customer_pool(initial_size=300, daily_acquisitions=0, days_elapsed=0)
        offline_fraction = sum(1 for p in pool.values() if p.is_offline_only) / len(pool)

        rng2 = np.random.default_rng(seed=1)
        events = [_generate_event(pool, rng2, anomaly=False) for _ in range(500)]
        null_fraction = sum(1 for e in events if e.engagement_score is None) / len(events)

        # null rate should be close to the offline customer fraction
        assert abs(null_fraction - offline_fraction) < 0.10


# ---------------------------------------------------------------------------
# Stable identity — the property every window aggregate depends on
# ---------------------------------------------------------------------------


class TestStableIdentity:
    def test_pool_is_identical_across_calls(self):
        # The regression that made every rolling-window feature meaningless: a fresh uuid4()
        # pool per run meant no customer ever received a second event.
        first = _create_customer_pool(initial_size=50, daily_acquisitions=0, days_elapsed=0)
        second = _create_customer_pool(initial_size=50, daily_acquisitions=0, days_elapsed=0)
        assert set(first) == set(second)

    def test_profiles_are_identical_across_calls(self):
        first = _create_customer_pool(initial_size=50, daily_acquisitions=0, days_elapsed=0)
        second = _create_customer_pool(initial_size=50, daily_acquisitions=0, days_elapsed=0)
        assert first == second

    def test_yesterdays_pool_is_a_subset_of_todays(self):
        # Acquisition must add customers, never re-identify the existing base.
        yesterday = _create_customer_pool(initial_size=50, daily_acquisitions=10, days_elapsed=3)
        today = _create_customer_pool(initial_size=50, daily_acquisitions=10, days_elapsed=4)
        assert set(yesterday) < set(today)
        assert len(today) - len(yesterday) == 10

    def test_pool_grows_with_elapsed_days(self):
        pool = _create_customer_pool(initial_size=100, daily_acquisitions=5, days_elapsed=10)
        assert len(pool) == 150

    def test_tier_and_region_are_stable_per_customer(self, customer_pool, rng):
        # Both were drawn per event before, so a customer appeared in a different tier and
        # region on every event and customer_features read whichever came last.
        profile = next(iter(customer_pool.values()))
        events = [_generate_event(customer_pool, rng, profile=profile) for _ in range(50)]
        assert {e.membership_tier for e in events} == {profile.membership_tier}
        assert {e.region for e in events} == {profile.region}


# ---------------------------------------------------------------------------
# Churn is absorbing
# ---------------------------------------------------------------------------


class TestChurnSemantics:
    def test_churned_events_are_cancellations(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        events = [_generate_event(customer_pool, rng) for _ in range(2000)]
        churned = [e for e in events if e.churned]
        assert churned, "expected at least one churn in 2000 events"
        assert all(e.event_type == "membership_cancelled" for e in churned)

    def test_active_events_are_never_cancellations(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        events = [_generate_event(customer_pool, rng) for _ in range(2000)]
        assert all(e.event_type != "membership_cancelled" for e in events if not e.churned)
