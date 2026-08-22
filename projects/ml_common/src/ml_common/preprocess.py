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


def categorical_categories(X: pd.DataFrame) -> dict[str, list[str]]:
    """Snapshot each categorical column's training-time category values.

    XGBoost's native categorical splits (enable_categorical=True) operate on the integer
    codes pandas assigns from a category dtype's `categories` index. Freeze this mapping
    into metadata.json at training time and reuse it at inference (select_inference_features)
    so a prediction batch that doesn't span every category level gets the same code
    assignment training used, instead of each call independently re-deriving `categories`
    from whatever values happen to be present.
    """
    return {col: X[col].cat.categories.tolist() for col in CATEGORICAL_COLS if col in X.columns}


def select_inference_features(
    df: pd.DataFrame,
    feature_names: list[str],
    categorical_categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Reindex an inference-time DataFrame to the training-time feature set, in order.

    There is no independent feature-selection step at serving time: the feature set is
    fixed once at training time and frozen into the model artifact's metadata.json. This
    reproduces that exact column set/order/dtype rather than re-deriving it.

    categorical_categories, when given, is the training-time category list per column (see
    categorical_categories() above, threaded through from metadata.json) — a batch missing
    some category levels then still gets training's exact code mapping, and any category
    value unseen at training time becomes NaN (treated as missing) rather than a new code.
    Older artifacts predating this field pass None here, falling back to inferring categories
    from whatever's in the batch, same as before.

    Batch Prediction instances originate from a BigQuery JSON export (see
    post_training.batch_predict), and BigQuery's JSON export format always stringifies
    INTEGER/NUMERIC columns to avoid precision loss — so every non-categorical column must
    be coerced back to numeric here, or XGBoost rejects the DataFrame's dtypes at predict time.

    That same export also omits any key whose value is NULL for a given row, so a nullable
    feature column is absent from df entirely whenever every instance in a request batch
    happens to have it NULL — reindex (not plain __getitem__) so that case fills NaN instead
    of raising KeyError. A missing categorical column then reindexes in as float64 NaN, and
    casting float64 straight to "category" gives it a floating-point category index, which
    XGBoost rejects — so route categoricals through object dtype first to keep the index
    dtype string/empty instead.
    """
    categorical_categories = categorical_categories or {}
    X = df.reindex(columns=feature_names)
    for col in X.columns:
        if col in CATEGORICAL_COLS:
            X[col] = X[col].astype("object").astype("category")
            if col in categorical_categories:
                # set_categories (not astype(CategoricalDtype(...))) so a value unseen at
                # training time becomes NaN instead of pandas raising/deprecating on it.
                X[col] = X[col].cat.set_categories(categorical_categories[col])
        else:
            X[col] = pd.to_numeric(X[col])
    return X
