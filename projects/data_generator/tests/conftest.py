import numpy as np
import pytest
from data_generator.main import CustomerProfile, _create_customer_pool


@pytest.fixture
def customer_pool() -> dict[str, CustomerProfile]:
    """A small fixed-size customer pool for tests."""
    rng = np.random.default_rng(seed=42)
    return _create_customer_pool(20, rng)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG so tests are deterministic."""
    return np.random.default_rng(seed=42)
