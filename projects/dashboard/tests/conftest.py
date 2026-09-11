import os

import numpy as np
import pandas as pd
import pytest

# app.py builds its settings at module scope, the way a Streamlit script does, so the
# environment has to be in place before anything imports it — including pytest's own
# collection-time import. Set here rather than in a fixture, which runs too late.
os.environ.setdefault("BQ_PROJECT_ID", "test-project")


@pytest.fixture
def snapshot() -> pd.DataFrame:
    """A scored snapshot with a deliberate spread: most customers healthy, a few not."""
    rng = np.random.default_rng(1234)
    size = 200
    return pd.DataFrame(
        {
            "customer_id": [f"cust-{i:04d}" for i in range(size)],
            "churn_probability": rng.uniform(0, 1, size),
            "churn_prediction": rng.uniform(0, 1, size) > 0.5,
            "model_version": "projects/p/locations/eu/models/churn-predictor",
            "snapshot_date": pd.Timestamp("2026-09-10").date(),
            "membership_tier": rng.choice(["supporter", "friend", "champion"], size),
            "region": rng.choice(["north", "south"], size),
            "preferred_channel": rng.choice(["email", "web"], size),
            "member_since_days": rng.integers(30, 2000, size),
            "monthly_value": rng.uniform(5, 200, size),
            # Median is a healthy 0.0 for most of the base, which is exactly the degenerate
            # case cohort_reference has to handle.
            "payment_failure_rate_30d": np.where(
                rng.uniform(size=size) < 0.7, 0.0, rng.uniform(0.1, 0.9, size)
            ),
            "days_since_last_successful_payment": rng.integers(1, 120, size).astype(float),
            "contact_requests_last_30d": rng.integers(0, 6, size),
            "avg_engagement_30d": rng.uniform(10, 90, size),
            "events_last_30d": rng.integers(1, 40, size),
            "campaign_participation_rate": rng.uniform(0, 1, size),
        }
    )
