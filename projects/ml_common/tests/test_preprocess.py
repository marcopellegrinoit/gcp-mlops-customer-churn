"""Unit tests for feature preprocessing logic."""

import pandas as pd
import pytest
from ml_common.preprocess import (
    CATEGORICAL_COLS,
    TARGET_COL,
    categorical_categories,
    feature_names,
    prepare_features,
    select_inference_features,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["d1", "d2", "d3"],
            "membership_tier": ["friend", "champion", "guardian"],
            "region": ["north", "south", "north"],
            "preferred_channel": ["email", "web", "phone"],
            "member_since_days": [100, 200, 300],
            "avg_transaction_30d": [10.0, 20.0, 30.0],
            "churned": [0, 1, 0],
        }
    )


def test_prepare_features_drops_id_and_target(sample_df):
    X, _ = prepare_features(sample_df)
    assert "customer_id" not in X.columns
    assert TARGET_COL not in X.columns


def test_prepare_features_y_is_int(sample_df):
    _, y = prepare_features(sample_df)
    assert y.dtype == int
    assert list(y) == [0, 1, 0]


def test_prepare_features_categoricals_encoded(sample_df):
    X, _ = prepare_features(sample_df)
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            assert X[col].dtype.name == "category", f"{col} should be category dtype"


def test_prepare_features_numeric_unchanged(sample_df):
    X, _ = prepare_features(sample_df)
    assert list(X["member_since_days"]) == [100, 200, 300]
    assert list(X["avg_transaction_30d"]) == [10.0, 20.0, 30.0]


def test_feature_names_excludes_id_and_target(sample_df):
    names = feature_names(sample_df)
    assert "customer_id" not in names
    assert TARGET_COL not in names
    assert "member_since_days" in names


def test_prepare_features_missing_categorical_column(sample_df):
    """prepare_features should not crash when an optional categorical column is absent."""
    df = sample_df.drop(columns=["preferred_channel"])
    X, _ = prepare_features(df)
    assert "preferred_channel" not in X.columns


def test_select_inference_features_matches_training_order(sample_df):
    names = feature_names(sample_df)
    # incoming inference rows have no target column and arrive in a different column order
    inference_df = sample_df.drop(columns=[TARGET_COL])[list(reversed(names))]
    X = select_inference_features(inference_df, names)
    assert list(X.columns) == names


def test_select_inference_features_casts_categoricals(sample_df):
    names = feature_names(sample_df)
    inference_df = sample_df.drop(columns=[TARGET_COL])
    X = select_inference_features(inference_df, names)
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            assert X[col].dtype.name == "category"


def test_select_inference_features_fills_nan_when_nullable_column_entirely_absent(sample_df):
    # BigQuery's JSON export omits a NULL field per-row, so a batch where every instance
    # happens to have a nullable feature NULL arrives with that column missing altogether.
    names = feature_names(sample_df)
    inference_df = sample_df.drop(columns=[TARGET_COL, "avg_transaction_30d"])
    X = select_inference_features(inference_df, names)
    assert list(X.columns) == names
    assert X["avg_transaction_30d"].isna().all()


def test_select_inference_features_fills_nan_when_categorical_column_entirely_absent(sample_df):
    # Same missing-column case as above, but for a categorical: reindex fills float64 NaN,
    # which must not be cast straight to category dtype (XGBoost rejects a float category
    # index) — this is the bug behind the reported "floating point dtype" crash.
    names = feature_names(sample_df)
    inference_df = sample_df.drop(columns=[TARGET_COL, "region"])
    X = select_inference_features(inference_df, names)
    assert X["region"].dtype.name == "category"
    assert X["region"].cat.categories.dtype == object
    assert X["region"].isna().all()


def test_categorical_categories_snapshots_training_time_levels(sample_df):
    X, _ = prepare_features(sample_df)
    cats = categorical_categories(X)
    assert cats["membership_tier"] == sorted(["friend", "champion", "guardian"])
    assert set(cats.keys()) == set(CATEGORICAL_COLS)


def test_select_inference_features_reuses_frozen_categories(sample_df):
    # A prediction batch that only spans a subset of training's category levels must still
    # get training's exact code mapping, not a mapping re-derived from this smaller batch.
    X, _ = prepare_features(sample_df)
    names = feature_names(sample_df)
    frozen = categorical_categories(X)

    inference_df = sample_df.drop(columns=[TARGET_COL]).iloc[[0]]  # only "friend"/"north"/"email"
    result = select_inference_features(inference_df, names, frozen)

    assert list(result["membership_tier"].cat.categories) == frozen["membership_tier"]
    assert list(result["region"].cat.categories) == frozen["region"]


def test_select_inference_features_unseen_category_becomes_missing(sample_df):
    X, _ = prepare_features(sample_df)
    names = feature_names(sample_df)
    frozen = categorical_categories(X)

    inference_df = sample_df.drop(columns=[TARGET_COL]).copy()
    inference_df.loc[0, "membership_tier"] = "brand_new_tier"  # unseen at training time
    result = select_inference_features(inference_df, names, frozen)

    assert pd.isna(result["membership_tier"].iloc[0])
