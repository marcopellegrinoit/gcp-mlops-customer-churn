"""Champion/challenger evaluation gate: pure metrics, no GCP dependencies."""

import numpy as np
from data_contracts import EvaluationMetrics, ModelMetrics, PromotionDecision
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from ml_common.config import MLSettings, get_settings


def compute_full_metrics(
    y_true: np.ndarray,
    challenger_proba: np.ndarray,
    challenger_threshold: float,
    challenger_shap: dict,
    champion_proba: np.ndarray | None = None,
    champion_threshold: float | None = None,
    champion_shap: dict | None = None,
) -> EvaluationMetrics:
    """Score the challenger (and champion, if any) against a held-out test set.

    challenger_threshold/champion_threshold are each pre-selected at training time from
    out-of-fold cross-validation predictions (see modeling.train.select_threshold_via_cv) —
    never from this test set, so the reported F1 isn't optimistically biased by also having
    picked the operating point on it.

    Carries no promotion verdict — see decide().
    """
    challenger_metrics = compute_metrics(y_true, challenger_proba, challenger_threshold)

    if champion_proba is None:
        return EvaluationMetrics(
            challenger_metrics=challenger_metrics,
            champion_metrics=None,
            threshold=challenger_metrics.threshold,
            shap_rank_correlation=None,
            challenger_shap_importance=challenger_shap,
        )

    champion_metrics = compute_metrics(y_true, champion_proba, champion_threshold)

    return EvaluationMetrics(
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
        threshold=challenger_metrics.threshold,
        shap_rank_correlation=_spearman_rank_correlation(challenger_shap, champion_shap),
        challenger_shap_importance=challenger_shap,
    )


def flatten_for_kfp_metrics(metrics: EvaluationMetrics, decision: PromotionDecision) -> dict:
    """Flatten compute_full_metrics()/decide() output into scalars for a KFP dsl.Metrics artifact.

    dsl.Metrics only displays flat numeric/boolean key-value pairs, so the nested
    challenger/champion metrics get prefixed and merged into one dict here.
    """
    flat = {f"challenger_{k}": v for k, v in metrics.challenger_metrics.model_dump().items()}
    if metrics.champion_metrics is not None:
        flat.update({f"champion_{k}": v for k, v in metrics.champion_metrics.model_dump().items()})
    if metrics.shap_rank_correlation is not None:
        flat["shap_rank_correlation"] = metrics.shap_rank_correlation
    flat["promote"] = decision.promote
    if decision.pr_auc_delta is not None:
        flat["pr_auc_delta"] = decision.pr_auc_delta
        flat["f1_delta"] = decision.f1_delta
    return flat


def decide(metrics: EvaluationMetrics, settings: MLSettings | None = None) -> PromotionDecision:
    """Apply the promotion gate to metrics already computed by compute_full_metrics().

    Auto-promotes when no champion exists yet (first-ever pipeline run). Otherwise both the
    PR-AUC and F1 deltas must clear their floor to prevent single-metric gaming. The floors
    come from ml_common.config, so a deployment can tighten or relax the gate through
    ML_PR_AUC_MIN_DELTA/ML_F1_MIN_DELTA; pass settings explicitly to gate against a specific
    pair rather than the deployed one.
    """
    settings = settings or get_settings()
    if metrics.champion_metrics is None:
        return PromotionDecision(promote=True, pr_auc_delta=None, f1_delta=None)

    pr_auc_delta = metrics.challenger_metrics.pr_auc - metrics.champion_metrics.pr_auc
    f1_delta = metrics.challenger_metrics.f1 - metrics.champion_metrics.f1
    promote = pr_auc_delta >= settings.pr_auc_min_delta and f1_delta >= settings.f1_min_delta

    return PromotionDecision(promote=promote, pr_auc_delta=pr_auc_delta, f1_delta=f1_delta)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> ModelMetrics:
    """Compute PR-AUC, ROC-AUC, and F1 at an already-selected threshold.

    threshold is never derived from (y_true, proba) here — see select_threshold_via_cv
    in modeling.train for where it comes from — so F1 isn't biased by having also picked
    the operating point on this same data.
    """
    pr_auc = float(average_precision_score(y_true, proba))
    roc_auc = float(roc_auc_score(y_true, proba))
    preds = (proba >= threshold).astype(int)
    f1 = float(f1_score(y_true, preds))
    return ModelMetrics(pr_auc=pr_auc, roc_auc=roc_auc, f1=f1, threshold=threshold)


def select_threshold(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Select the highest-precision threshold that still achieves target_recall.

    Recall falls as the threshold rises, so the candidates clearing target_recall are a
    prefix of the curve and the most precise of them is the *largest* threshold in it — the
    strictest operating point that still catches the required share of churners. (This
    docstring used to say "lowest threshold", which is the opposite of both the code and the
    intent: the lowest threshold clearing any recall target is the one that flags everybody.)

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
    """Compute Spearman rank correlation between two SHAP importance dicts.

    Ties are real here and have to be handled, not assumed away: XGBoost gives a feature it
    never split on a mean |SHAP| of exactly 0.0, so a model with unused features carries a
    block of exact ties — and the two models being compared rarely leave the *same* features
    unused. Hence average ranks (``_average_ranks``) rather than ordinal position, and
    Pearson over those ranks rather than the ``1 - 6*d²/(n(n²-1))`` shortcut, which is only
    equal to Spearman when no value is tied.

    None when either side is constant across the shared features: every rank is then the
    same, the correlation is 0/0, and reporting any number for it would be an invention.
    """
    if baseline is None or len(current) < 2:
        return None
    features = sorted(set(current) & set(baseline))
    if len(features) < 2:
        return None
    cr = _average_ranks(np.array([current[f] for f in features], dtype=float))
    br = _average_ranks(np.array([baseline[f] for f in features], dtype=float))
    if cr.std() == 0.0 or br.std() == 0.0:
        return None
    return float(np.corrcoef(cr, br)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` ascending from 1, giving every tied group its group's mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # Collapse each run of equal values onto the mean of the ordinal ranks it was handed.
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks
