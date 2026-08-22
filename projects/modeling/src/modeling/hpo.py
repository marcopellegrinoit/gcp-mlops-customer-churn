"""Bayesian hyperparameter search using Optuna TPE."""

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

# suppress per-trial INFO noise; results are captured via trial_callback
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    search_space: dict,
    fixed_params: dict,
    n_trials: int,
    n_folds: int,
    random_state: int = 42,
    trial_callback: Callable | None = None,
) -> dict:
    """Run Optuna TPE hyperparameter search and return the best parameter dict.

    trial_callback(trial_number, params, mean_pr_auc, std_pr_auc) is called after each
    trial when provided — inject Vertex AI Experiments logging from the platform layer.
    The objective maximises PR-AUC averaged across stratified K-folds.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            else:
                # log=True applies log-scale sampling (e.g. learning_rate spans orders of magnitude)
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
        params.update(fixed_params)
        params["random_state"] = random_state

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_scores: list[float] = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            # computed per-fold to avoid leaking the full-set class ratio into validation
            scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
            model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight)
            model.fit(X_train, y_train, verbose=False)
            proba = model.predict_proba(X_val)[:, 1]
            fold_scores.append(float(average_precision_score(y_val, proba)))

        mean_pr_auc = float(np.mean(fold_scores))
        std_pr_auc = float(np.std(fold_scores))

        if trial_callback is not None:
            # strip fixed/non-tunable params so only searched hyperparameters are reported
            log_params = {
                k: v for k, v in params.items() if k not in fixed_params and k != "random_state"
            }
            trial_callback(trial.number, log_params, mean_pr_auc, std_pr_auc)

        return mean_pr_auc

    # seed the sampler so repeated runs with the same config produce comparable trial sequences
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
