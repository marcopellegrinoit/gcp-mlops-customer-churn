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
    _CHANNELS,
    _CHURN_HIGH_ENGAGEMENT_THRESHOLD,
    _CHURN_HIGH_PAYMENT_THRESHOLD,
    _CHURN_PROB_HIGH,
    _CHURN_PROB_LOW,
    _EVENT_TYPES,
    _PAYMENT_STATUSES,
    _REGIONS,
    _TIERS,
    CustomerProfile,
    _create_customer_pool,
    _generate_event,
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
    def test_normal_event_has_all_fields(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=False)
        assert _ALL_FIELDS.issubset(event.keys())

    def test_anomaly_event_has_all_fields(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=True)
        assert _ALL_FIELDS.issubset(event.keys())

    def test_no_extra_fields(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert set(event.keys()) == _ALL_FIELDS

    def test_event_id_is_valid_uuid(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        uuid.UUID(event["event_id"])  # raises ValueError if malformed

    def test_event_timestamp_is_utc_isoformat(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        parsed = datetime.fromisoformat(event["event_timestamp"])
        assert parsed.tzinfo is not None

    def test_churned_is_bool(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert isinstance(event["churned"], bool)

    def test_anomaly_injected_is_bool(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert isinstance(event["anomaly_injected"], bool)


# ---------------------------------------------------------------------------
# Enum constraints
# ---------------------------------------------------------------------------


class TestEnumConstraints:
    def test_event_type_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event["event_type"] in _EVENT_TYPES

    def test_region_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event["region"] in _REGIONS

    def test_membership_tier_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event["membership_tier"] in _TIERS

    def test_payment_status_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event["payment_status"] in _PAYMENT_STATUSES

    def test_channel_in_valid_set(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event["channel"] in _CHANNELS

    def test_customer_id_from_pool(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        assert event["customer_id"] in customer_pool


# ---------------------------------------------------------------------------
# Normal event value ranges
# ---------------------------------------------------------------------------


class TestNormalValues:
    def test_transaction_is_non_negative(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            assert event["monthly_transaction"] >= 0.0

    def test_engagement_score_in_unit_interval(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            score = event["engagement_score"]
            assert score is None or 0.0 <= score <= 1.0

    def test_payment_attempts_non_negative(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            assert event["payment_attempts_last_30d"] >= 0

    def test_contact_requests_non_negative(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            assert event["contact_requests_last_30d"] >= 0

    def test_member_since_days_in_range(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(300):
            event = _generate_event(customer_pool, rng, anomaly=False)
            days = event["member_since_days"]
            assert days is None or 1 <= days <= 3650

    def test_anomaly_injected_false(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=False)
        assert event["anomaly_injected"] is False


# ---------------------------------------------------------------------------
# Anomaly event values
# ---------------------------------------------------------------------------


class TestAnomalyValues:
    def test_anomaly_injected_true(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng, anomaly=True)
        assert event["anomaly_injected"] is True

    def test_anomaly_transaction_from_known_set(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True)["monthly_transaction"]
            for _ in range(100)
        }
        assert observed.issubset(set(_ANOMALY_TRANSACTIONS))

    def test_anomaly_engagement_from_known_set(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True)["engagement_score"]
            for _ in range(100)
        }
        assert observed.issubset(set(_ANOMALY_ENGAGEMENT_SCORES))

    def test_anomaly_payment_attempts_in_range(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        for _ in range(100):
            event = _generate_event(customer_pool, rng, anomaly=True)
            assert (
                _ANOMALY_PAYMENT_ATTEMPTS_MIN
                <= event["payment_attempts_last_30d"]
                <= _ANOMALY_PAYMENT_ATTEMPTS_MAX
            )

    def test_anomaly_transactions_cover_all_values(self, customer_pool):
        """All three anomaly transaction values should appear across enough trials."""
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True)["monthly_transaction"]
            for _ in range(300)
        }
        assert observed == set(_ANOMALY_TRANSACTIONS)

    def test_anomaly_engagement_covers_all_values(self, customer_pool):
        rng = np.random.default_rng(seed=0)
        observed = {
            _generate_event(customer_pool, rng, anomaly=True)["engagement_score"]
            for _ in range(300)
        }
        assert observed == set(_ANOMALY_ENGAGEMENT_SCORES)


# ---------------------------------------------------------------------------
# raw_payload integrity
# ---------------------------------------------------------------------------


class TestRawPayload:
    def test_raw_payload_is_valid_json(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        parsed = json.loads(event["raw_payload"])
        assert isinstance(parsed, dict)

    def test_raw_payload_contains_core_fields(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event["raw_payload"])
        assert _CORE_FIELDS.issubset(payload.keys())

    def test_raw_payload_excludes_itself(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event["raw_payload"])
        assert "raw_payload" not in payload

    def test_raw_payload_excludes_anomaly_injected(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event["raw_payload"])
        assert "anomaly_injected" not in payload

    def test_raw_payload_values_consistent_with_event(self, customer_pool, rng):
        event = _generate_event(customer_pool, rng)
        payload = json.loads(event["raw_payload"])
        assert payload["event_id"] == event["event_id"]
        assert payload["customer_id"] == event["customer_id"]
        assert payload["monthly_transaction"] == event["monthly_transaction"]
        assert payload["engagement_score"] == event["engagement_score"]


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
        """Anomaly events have payment_attempts >= 50 > threshold, so churn_prob = 0.25."""
        rng = np.random.default_rng(seed=7)
        churned = sum(
            _generate_event(customer_pool, rng, anomaly=True)["churned"] for _ in range(400)
        )
        churn_rate = churned / 400
        # Expected ~25%; bounds are wide enough to survive sampling variance
        assert 0.12 < churn_rate < 0.40, (
            f"Expected ~25% churn for high-risk events, got {churn_rate:.1%}"
        )

    def test_low_risk_events_churn_at_low_rate(self, customer_pool):
        """Normal events with safe engagement/payment values should churn at ~4%."""
        rng = np.random.default_rng(seed=3)
        safe_events = []
        for _ in range(5000):
            e = _generate_event(customer_pool, rng, anomaly=False)
            profile = customer_pool[e["customer_id"]]
            if (
                e["engagement_score"] is not None
                and e["engagement_score"] >= _CHURN_HIGH_ENGAGEMENT_THRESHOLD
                and e["payment_attempts_last_30d"] <= _CHURN_HIGH_PAYMENT_THRESHOLD
                and not profile.always_fails_payments
            ):
                safe_events.append(e)
            if len(safe_events) >= 400:
                break

        churn_rate = sum(e["churned"] for e in safe_events) / len(safe_events)
        assert 0.0 < churn_rate < 0.12, (
            f"Expected ~4% churn for low-risk events, got {churn_rate:.1%}"
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
        member_since_days=365,
    )


def _failing_profile() -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(uuid.uuid4()),
        preferred_channel="email",
        is_offline_only=False,
        always_fails_payments=True,
        member_since_days=365,
    )


class TestMissingValues:
    def test_offline_customer_has_null_engagement(self):
        """Offline-only customers always produce engagement_score=None (MAR)."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(_offline_profile())
        for _ in range(30):
            event = _generate_event(pool, rng, anomaly=False)
            assert event["engagement_score"] is None

    def test_online_customer_always_has_engagement(self):
        """Online customers always produce a numeric engagement_score."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(
            CustomerProfile(
                customer_id=str(uuid.uuid4()),
                preferred_channel="email",
                is_offline_only=False,
                always_fails_payments=False,
                member_since_days=365,
            )
        )
        for _ in range(30):
            event = _generate_event(pool, rng, anomaly=False)
            assert event["engagement_score"] is not None

    def test_always_fails_customer_never_has_successful_payment(self):
        """Customers flagged always_fails_payments never produce payment_status='success' (MNAR)."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(_failing_profile())
        for _ in range(50):
            event = _generate_event(pool, rng, anomaly=False)
            assert event["payment_status"] != "success"

    def test_member_since_days_can_be_null(self):
        """Customers with no recorded start date produce member_since_days=None (MCAR)."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(
            CustomerProfile(
                customer_id=str(uuid.uuid4()),
                preferred_channel="web",
                is_offline_only=False,
                always_fails_payments=False,
                member_since_days=None,
            )
        )
        for _ in range(10):
            event = _generate_event(pool, rng, anomaly=False)
            assert event["member_since_days"] is None

    def test_anomaly_overrides_offline_engagement(self):
        """Anomaly injection always sets a numeric engagement_score, even for offline customers."""
        rng = np.random.default_rng(seed=0)
        pool = _make_pool(_offline_profile())
        for _ in range(30):
            event = _generate_event(pool, rng, anomaly=True)
            assert event["engagement_score"] in _ANOMALY_ENGAGEMENT_SCORES

    def test_offline_pool_produces_null_engagement_at_expected_rate(self):
        """With ~33% offline channels, roughly a third of events should have null engagement."""
        rng = np.random.default_rng(seed=0)
        pool = _create_customer_pool(300, rng)
        offline_fraction = sum(1 for p in pool.values() if p.is_offline_only) / len(pool)

        rng2 = np.random.default_rng(seed=1)
        events = [_generate_event(pool, rng2, anomaly=False) for _ in range(500)]
        null_fraction = sum(1 for e in events if e["engagement_score"] is None) / len(events)

        # null rate should be close to the offline customer fraction
        assert abs(null_fraction - offline_fraction) < 0.10
