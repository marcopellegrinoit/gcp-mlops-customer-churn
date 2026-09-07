import numpy as np
import pytest
from data_generator.main import CustomerProfile, _create_customer_pool


@pytest.fixture
def customer_pool() -> dict[str, CustomerProfile]:
    """A small fixed customer pool. Deterministic by construction — no seed needed."""
    return _create_customer_pool(initial_size=20, daily_acquisitions=0, days_elapsed=0)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG so tests are deterministic."""
    return np.random.default_rng(seed=42)
