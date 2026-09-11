"""FastAPI app implementing Vertex AI's custom-container prediction contract.

Vertex AI BatchPredictionJob starts this container, sets AIP_STORAGE_URI to the
registered model's GCS artifact directory and AIP_HTTP_PORT/AIP_HEALTH_ROUTE/
AIP_PREDICT_ROUTE per its custom-container spec, then POSTs batches of instances
to the predict route and writes the responses to bigquery_destination.

THRESHOLD is passed as a container env var at BatchPredictionJob creation time
(read from the champion model's registry metadata, set by register_or_reject) —
the same decision threshold used during the evaluate stage, applied identically
at serving time.

The request and response bodies are ml_common.contracts models, so what this container
emits is the same declared shape post_training.evaluate parses back out of the Batch
Prediction output when it decides whether to promote a model.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from ml_common.contracts import ChurnPrediction, ModelMetadata, PredictRequest, PredictResponse
from ml_common.preprocess import select_inference_features

from serving.predict import load_model
from serving.settings import get_settings
from serving.storage import download_json

settings = get_settings()

_model = None
_metadata: ModelMetadata | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _model, _metadata
    artifact_uri = settings.aip_storage_uri.rstrip("/")
    _model = load_model(artifact_uri, project_id=settings.project_id)
    _metadata = ModelMetadata.model_validate(
        download_json(f"{artifact_uri}/metadata.json", project_id=settings.project_id)
    )
    yield


app = FastAPI(lifespan=_lifespan)


def _reject_unnamed_instances(df: pd.DataFrame) -> None:
    """Fail the batch when instances carry none of the training-time feature names.

    select_inference_features reindexes onto the frozen feature names, which turns an
    instance whose keys it does not recognise into an all-NaN row rather than an error —
    and XGBoost scores all-NaN happily, returning its all-missing constant. That
    combination is silent by construction: a BatchPredictionJob completes, reports every
    row successful, writes no errors table, and fills the output with one identical
    probability. It did, for 244,435 rows a night, until the score-drift check caught it.

    The cause was Vertex sending positional arrays instead of named objects (see
    instanceConfig in submit_batch_predict), which is now requested explicitly. This guard
    is the backstop: any future contract drift fails loudly as row errors instead of
    quietly producing confident, uniform, meaningless predictions.

    Deliberately keyed on column names rather than on all-NaN values — a batch of
    legitimately null-heavy rows is valid input, whereas instances carrying none of the
    expected names cannot be.
    """
    if df.empty:
        return
    if not set(df.columns) & set(_metadata.feature_names):
        raise HTTPException(
            status_code=400,
            detail=(
                "instances carry none of the model's feature names "
                f"(got {sorted(str(c) for c in df.columns)[:5]}...); "
                "expected named fields — check the batch prediction job's "
                "instanceConfig.instanceType is 'object'"
            ),
        )


@app.get(settings.aip_health_route)
def health() -> dict:
    """Report container readiness to Vertex AI's health probe."""
    return {"status": "ok"}


@app.post(settings.aip_predict_route, response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Score a batch of instances and return churn probabilities with thresholded labels."""
    df = pd.DataFrame(request.instances)
    _reject_unnamed_instances(df)
    X = select_inference_features(df, _metadata.feature_names, _metadata.categorical_categories)
    proba = _model.predict_proba(X)[:, 1]
    return PredictResponse(
        predictions=[
            ChurnPrediction(
                churn_probability=float(p), churn_prediction=bool(p >= settings.threshold)
            )
            for p in proba
        ]
    )
