"""Unit tests for run_hpo (no GCP dependencies)."""

import numpy as np
import pandas as pd
from modeling.config import FloatParam, IntParam
from modeling.hpo import run_hpo

FIXED_PARAMS = {"tree_method": "hist"}
SEARCH_SPACE = {
    "max_depth": IntParam(low=2, high=4),
    "learning_rate": FloatParam(low=0.05, high=0.3, log=True),
}


def _make_dataset(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] + X["f1"] > 1.0).astype(int))
    return X, y


def test_run_hpo_returns_params_within_search_space():
    X, y = _make_dataset()
    best_params = run_hpo(X, y, SEARCH_SPACE, FIXED_PARAMS, n_trials=3, n_folds=2)
    depth = SEARCH_SPACE["max_depth"]
    assert depth.low <= best_params["max_depth"] <= depth.high
    assert (
        SEARCH_SPACE["learning_rate"].low
        <= best_params["learning_rate"]
        <= SEARCH_SPACE["learning_rate"].high
    )


def test_run_hpo_excludes_fixed_params_from_best_params():
    X, y = _make_dataset()
    best_params = run_hpo(X, y, SEARCH_SPACE, FIXED_PARAMS, n_trials=3, n_folds=2)
    assert "tree_method" not in best_params


def test_run_hpo_is_deterministic_given_random_state():
    X, y = _make_dataset()
    best1 = run_hpo(X, y, SEARCH_SPACE, FIXED_PARAMS, n_trials=3, n_folds=2, random_state=5)
    best2 = run_hpo(X, y, SEARCH_SPACE, FIXED_PARAMS, n_trials=3, n_folds=2, random_state=5)
    assert best1 == best2


def test_run_hpo_invokes_trial_callback_per_trial():
    X, y = _make_dataset()
    calls = []

    def callback(trial_number, params, mean_pr_auc, std_pr_auc):
        calls.append((trial_number, params, mean_pr_auc, std_pr_auc))

    run_hpo(X, y, SEARCH_SPACE, FIXED_PARAMS, n_trials=3, n_folds=2, trial_callback=callback)
    assert len(calls) == 3
    for trial_number, params, mean_pr_auc, std_pr_auc in calls:
        assert isinstance(trial_number, int)
        assert "tree_method" not in params
        assert isinstance(mean_pr_auc, float)
        assert isinstance(std_pr_auc, float)
