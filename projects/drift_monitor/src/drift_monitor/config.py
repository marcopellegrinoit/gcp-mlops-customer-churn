"""GCP infrastructure and ops configuration for the drift monitor."""

import os

PROJECT_ID: str = os.environ.get("BQ_PROJECT_ID")
REGION: str = os.environ.get("REGION")
BQ_FEATURES_TABLE: str = os.environ.get("BQ_FEATURES_TABLE")
MODEL_DISPLAY_NAME: str = os.environ.get("MODEL_DISPLAY_NAME")
GCS_BUCKET: str = os.environ.get("GCS_BUCKET")
DECISION_BLOB: str = os.environ.get("DECISION_BLOB")
PSI_THRESHOLD: float = float(os.environ.get("PSI_THRESHOLD", "0.2"))
BATCH_PREDICT_DISPLAY_NAME: str = os.environ.get("BATCH_PREDICT_DISPLAY_NAME")

# Persistence rule: a feature must breach in at least PERSISTENCE_MIN_BREACHES of the last
# PERSISTENCE_WINDOW runs before it gates retraining. Defaults to 2-of-3 — enough to reject
# a one-off sampling artefact or a partial upstream load without delaying a real shift by
# more than a couple of days.
PERSISTENCE_WINDOW: int = int(os.environ.get("PERSISTENCE_WINDOW", "3"))
PERSISTENCE_MIN_BREACHES: int = int(os.environ.get("PERSISTENCE_MIN_BREACHES", "2"))
