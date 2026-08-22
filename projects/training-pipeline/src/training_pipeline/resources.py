"""Compute resource limits (CPU/memory) for pipeline stages.

Kept separate from pipeline.py so per-stage sizing can be tuned without touching
the pipeline's task wiring.
"""

# Sequential HPO trials/folds on tabular data don't need more than this to keep
# Vertex AI Pipelines compute costs low.
HPO_CPU_LIMIT = "2"
HPO_MEMORY_LIMIT = "8G"

TRAIN_CPU_LIMIT = "2"
TRAIN_MEMORY_LIMIT = "8G"

# Required by Vertex AI for BatchPredictionJob against custom-container models
# (registered or unmanaged) — unlike AutoML models, no default machine type is inferred.
BATCH_PREDICT_MACHINE_TYPE = "n1-standard-4"
