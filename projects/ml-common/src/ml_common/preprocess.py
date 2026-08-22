"""Feature preprocessing shared across training, evaluation, and serving."""

import pandas as pd

CATEGORICAL_COLS: list[str] = ["membership_tier", "region", "preferred_channel"]
DROP_COLS: list[str] = ["customer_id", "snapshot_date"]
TARGET_COL: str = "churned"


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw feature DataFrame into X and y; cast categoricals to category dtype."""
    df = df.copy()
    y = df[TARGET_COL].astype(int)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    X = df.drop(columns=cols_to_drop)
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(
                "category"
            )  # enables XGBoost native categorical splits (no one-hot encoding)
    return X, y


def feature_names(df: pd.DataFrame) -> list[str]:
    """Return the ordered feature column names after preprocessing."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    return [c for c in df.columns if c not in cols_to_drop]


def select_inference_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reindex an inference-time DataFrame to the training-time feature set, in order.

    There is no independent feature-selection step at serving time: the feature set is
    fixed once at training time and frozen into the model artifact's metadata.json. This
    reproduces that exact column set/order/dtype rather than re-deriving it.

    Batch Prediction instances originate from a BigQuery JSON export (see
    post_training.batch_predict), and BigQuery's JSON export format always stringifies
    INTEGER/NUMERIC columns to avoid precision loss — so every non-categorical column must
    be coerced back to numeric here, or XGBoost rejects the DataFrame's dtypes at predict time.

    That same export also omits any key whose value is NULL for a given row, so a nullable
    feature column is absent from df entirely whenever every instance in a request batch
    happens to have it NULL — reindex (not plain __getitem__) so that case fills NaN instead
    of raising KeyError.
    """
    X = df.reindex(columns=feature_names)
    for col in X.columns:
        if col in CATEGORICAL_COLS:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col])
    return X
