"""Environment configuration for the Vertex AI custom-container prediction server.

The AIP_* variables are set by Vertex AI itself when it starts this container, per its
custom-container contract. THRESHOLD and PROJECT_ID are set by post_training.register on
the registered Model's container spec, so every batch prediction job against a given model
version applies exactly the decision threshold that model was evaluated with.

All five are required. Vertex always supplies the AIP_* set, and register_or_reject always
supplies the other two, so a missing one means the container was started by something that
did not follow the contract — and the right response is to fail the container's startup
rather than serve a batch against a guessed threshold.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Vertex AI custom-container batch-prediction env vars, plus THRESHOLD and PROJECT_ID."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    aip_http_port: int = 8080
    aip_storage_uri: str
    aip_health_route: str
    aip_predict_route: str

    # The operating point chosen from out-of-fold CV at training time and frozen into this
    # model version's container spec. Bounded because it is compared against a probability:
    # a value outside [0, 1] silently labels every row identically, which a batch of
    # uniform predictions would not otherwise distinguish from a working model.
    threshold: float = Field(ge=0.0, le=1.0)

    # Passed explicitly rather than relying on google-cloud-storage's ADC project detection,
    # because the batch-predict service account is attached to this container via short-lived
    # credential exchange, which doesn't always carry project info.
    project_id: str

    @field_validator("aip_health_route", "aip_predict_route")
    @classmethod
    def _must_be_absolute_path(cls, value: str) -> str:
        """Reject a route Starlette would register at a path Vertex will never call."""
        if not value.startswith("/"):
            raise ValueError(f"route must start with '/', got {value!r}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the container's settings, read from the environment once.

    Read at import time by serving.app, which needs the route paths to register its
    endpoints — so a malformed environment fails the container at startup, before Vertex
    sends it a single instance.
    """
    return Settings()
