"""Environment configuration for the Vertex AI custom-container prediction server."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Vertex AI custom-container batch-prediction env vars, plus THRESHOLD and PROJECT_ID.

    PROJECT_ID is passed explicitly (rather than relying on google-cloud-storage's ADC
    project detection) because the batch-predict service account is attached to this
    container via short-lived credential exchange, which doesn't always carry project info.
    """

    aip_http_port: int = 8080
    aip_storage_uri: str
    aip_health_route: str
    aip_predict_route: str
    threshold: float
    project_id: str


settings = Settings()
