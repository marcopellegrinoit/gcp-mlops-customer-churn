"""HPO/training configuration owned by the data science team.

Inference-time constants (promotion gate, RANDOM_STATE, TEST_SIZE) live in
ml_common.config, shared with evaluate and serving — re-exported here so
existing `ml_config.RANDOM_STATE`-style call sites keep working unchanged.
"""

from ml_common.config import (  # noqa: F401
    BQ_FEATURES_TABLE,
    CONSECUTIVE_REJECTION_KEY,
    F1_MIN_DELTA,
    MODEL_DISPLAY_NAME,
    PR_AUC_MIN_DELTA,
    RANDOM_STATE,
    TARGET_RECALL,
    TEST_SIZE,
)

HPO_N_TRIALS: int = 100
HPO_N_FOLDS: int = 5

HPO_SEARCH_SPACE: dict = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "n_estimators": {"type": "int", "low": 50, "high": 500},
}

XGB_FIXED_PARAMS: dict = {
    "tree_method": "hist",
    "enable_categorical": True,
}
