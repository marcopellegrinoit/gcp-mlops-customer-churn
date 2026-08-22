"""Champion/challenger evaluation gate: pure metrics, no GCP dependencies."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_full_metrics(
    y_true: np.ndarray,
    challenger_proba: np.ndarray,
    challenger_threshold: float,
    challenger_shap: dict,
    champion_proba: np.ndarray | None = None,
    champion_threshold: float | None = None,
    champion_shap: dict | None = None,
) -> dict:
    """Score the challenger (and champion, if any) against a held-out test set.

    challenger_threshold/champion_threshold are each pre-selected at training time from
    out-of-fold cross-validation predictions (see modeling.train.select_threshold_via_cv) —
    never from this test set, so the reported F1 isn't optimistically biased by also having
    picked the operating point on it.

    Returns a metrics dict: challenger_metrics, champion_metrics (None if no
    champion exists), threshold (the challenger's, baked into its serving container if
    promoted), and shap_rank_correlation. Carries no promotion verdict — see decide().
    """
    challenger_metrics = compute_metrics(y_true, challenger_proba, challenger_threshold)

    if champion_proba is None:
        return {
            "challenger_metrics": challenger_metrics,
            "champion_metrics": None,
            "threshold": challenger_metrics["threshold"],
            "shap_rank_correlation": None,
            "challenger_shap_importance": challenger_shap,
        }

    champion_metrics = compute_metrics(y_true, champion_proba, champion_threshold)

    return {
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "threshold": challenger_metrics["threshold"],
        "shap_rank_correlation": _spearman_rank_correlation(challenger_shap, champion_shap),
        "challenger_shap_importance": challenger_shap,
    }


def flatten_for_kfp_metrics(metrics: dict, decision: dict) -> dict:
    """Flatten compute_full_metrics()/decide() output into scalars for a KFP dsl.Metrics artifact.

    dsl.Metrics only displays flat numeric/boolean key-value pairs, so the nested
    challenger_metrics/champion_metrics dicts get prefixed and merged into one dict here.
    """
    flat = {f"challenger_{k}": v for k, v in metrics["challenger_metrics"].items()}
    if metrics["champion_metrics"] is not None:
        flat.update({f"champion_{k}": v for k, v in metrics["champion_metrics"].items()})
    if metrics["shap_rank_correlation"] is not None:
        flat["shap_rank_correlation"] = metrics["shap_rank_correlation"]
    flat["promote"] = decision["promote"]
    if decision["pr_auc_delta"] is not None:
        flat["pr_auc_delta"] = decision["pr_auc_delta"]
        flat["f1_delta"] = decision["f1_delta"]
    return flat


def decide(metrics: dict, pr_auc_min_delta: float, f1_min_delta: float) -> dict:
    """Apply the promotion gate to metrics already computed by compute_full_metrics().

    Auto-promotes when no champion exists yet (first-ever pipeline run). Otherwise
    both the PR-AUC and F1 deltas must clear their floor to prevent single-metric gaming.
    Returns a decision dict: promote (bool), pr_auc_delta, f1_delta (both None if no champion).
    """
    if metrics["champion_metrics"] is None:
        return {"promote": True, "pr_auc_delta": None, "f1_delta": None}

    pr_auc_delta = metrics["challenger_metrics"]["pr_auc"] - metrics["champion_metrics"]["pr_auc"]
    f1_delta = metrics["challenger_metrics"]["f1"] - metrics["champion_metrics"]["f1"]
    promote = pr_auc_delta >= pr_auc_min_delta and f1_delta >= f1_min_delta

    return {"promote": promote, "pr_auc_delta": pr_auc_delta, "f1_delta": f1_delta}


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Compute PR-AUC, ROC-AUC, and F1 at an already-selected threshold.

    threshold is never derived from (y_true, proba) here — see select_threshold_via_cv
    in modeling.train for where it comes from — so F1 isn't biased by having also picked
    the operating point on this same data.
    """
    pr_auc = float(average_precision_score(y_true, proba))
    roc_auc = float(roc_auc_score(y_true, proba))
    preds = (proba >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds))
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1, "threshold": threshold}


def select_threshold(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Select the lowest threshold achieving target_recall; break ties by highest precision.

    Falls back to the threshold that maximises recall when no candidate meets the target.
    Public so it can be unit-tested independently.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve appends a sentinel (p=1, r=0) without a paired threshold
    p, r, t = precisions[:-1], recalls[:-1], thresholds

    mask = r >= target_recall
    if not mask.any():
        return float(t[np.argmax(r)])

    return float(t[mask][np.argmax(p[mask])])


def _spearman_rank_correlation(current: dict, baseline: dict | None) -> float | None:
    """Compute Spearman rank correlation between two SHAP importance dicts."""
    if baseline is None or len(current) < 2:
        return None
    features = sorted(set(current) & set(baseline))
    if len(features) < 2:
        return None
    c = np.array([current[f] for f in features], dtype=float)
    b = np.array([baseline[f] for f in features], dtype=float)
    # Rank both vectors (average method for ties)
    cr = np.argsort(np.argsort(c)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    n = len(features)
    d2 = np.sum((cr - br) ** 2)
    return float(1.0 - 6.0 * d2 / (n * (n**2 - 1)))
