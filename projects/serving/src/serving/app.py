"""FastAPI app implementing Vertex AI's custom-container prediction contract.

Vertex AI BatchPredictionJob starts this container, sets AIP_STORAGE_URI to the
registered model's GCS artifact directory and AIP_HTTP_PORT/AIP_HEALTH_ROUTE/
AIP_PREDICT_ROUTE per its custom-container spec, then POSTs batches of instances
to the predict route and writes the responses to bigquery_destination.

THRESHOLD is passed as a container env var at BatchPredictionJob creation time
(read from the champion model's registry metadata, set by register_or_reject) —
the same decision threshold used during the evaluate stage, applied identically
at serving time.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ml_common.config import CHURN_PREDICTION_FIELD, CHURN_PROBABILITY_FIELD
from ml_common.preprocess import select_inference_features

from serving.predict import load_model
from serving.settings import settings
from serving.storage import download_json

_model = None
_feature_names: list[str] = []
_threshold: float = 0.5


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _model, _feature_names, _threshold
    artifact_uri = settings.aip_storage_uri.rstrip("/")
    _model = load_model(artifact_uri, project_id=settings.project_id)
    _feature_names = download_json(f"{artifact_uri}/metadata.json", project_id=settings.project_id)[
        "feature_names"
    ]
    _threshold = settings.threshold
    yield


app = FastAPI(lifespan=_lifespan)


@app.get(settings.aip_health_route)
def health() -> dict:
    """Report container readiness to Vertex AI's health probe."""
    return {"status": "ok"}


@app.post(settings.aip_predict_route)
async def predict(request: Request) -> JSONResponse:
    """Score a batch of instances and return churn probabilities with thresholded labels."""
    body = await request.json()
    df = pd.DataFrame(body["instances"])
    X = select_inference_features(df, _feature_names)
    proba = _model.predict_proba(X)[:, 1]
    predictions = [
        {CHURN_PROBABILITY_FIELD: float(p), CHURN_PREDICTION_FIELD: bool(p >= _threshold)}
        for p in proba
    ]
    return JSONResponse({"predictions": predictions})
