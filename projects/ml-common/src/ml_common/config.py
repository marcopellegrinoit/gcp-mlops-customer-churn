"""Shared inference-time configuration owned by the data science team."""

# champion/challenger promotion gate: both metrics must improve together to avoid gaming one
TARGET_RECALL: float = 0.80
PR_AUC_MIN_DELTA: float = 0.02
F1_MIN_DELTA: float = 0.01

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# Resource naming shared across the training pipeline, trainer, and post-training
# stages — change once here rather than re-typing the literal at each call site.
BQ_FEATURES_TABLE: str = "features.customer_features"
SPLIT_ASSIGNMENTS_TABLE: str = "ml.split_assignments"
MODEL_DISPLAY_NAME: str = "churn-predictor"
CONSECUTIVE_REJECTION_KEY: str = "consecutive_rejections"

# Prediction response field names — also the column names of the BigQuery daily-serving
# output table (iac/config/bigquery.yaml). serving.app writes these, post_training.evaluate
# reads them back from Batch Prediction output; sharing the literal keeps the two in sync.
CHURN_PROBABILITY_FIELD: str = "churn_probability"
CHURN_PREDICTION_FIELD: str = "churn_prediction"
