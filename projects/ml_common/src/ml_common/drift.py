"""Population Stability Index (PSI) drift detection shared by training and monitoring.

The trainer freezes a baseline distribution per feature at training time (decile bin
edges for numeric columns, category frequencies for categorical columns) into the
model artifact's metadata.json. The drift monitor later rebuilds the same shape from a
live snapshot and compares it against that frozen baseline — both sides must use this
module so the binning is identical on each side of the comparison.
"""

import numpy as np
import pandas as pd

from ml_common.preprocess import CATEGORICAL_COLS

_EPSILON = 1e-4  # avoids div-by-zero / log(0) when a bucket is empty on either side
_N_BUCKETS = 10


def compute_baseline_stats(X: pd.DataFrame) -> dict:
    """Freeze a per-feature baseline distribution from training data for later PSI comparison."""
    stats: dict = {}
    for col in X.columns:
        if col in CATEGORICAL_COLS:
            stats[col] = {"type": "categorical", "frequencies": _category_frequencies(X[col])}
        elif pd.api.types.is_numeric_dtype(X[col]):
            stats[col] = {"type": "numeric", "bin_edges": _decile_edges(X[col])}
    return stats


def compute_psi(baseline_stats: dict, current: pd.DataFrame) -> dict[str, float]:
    """Return the PSI score for every baseline feature present in current."""
    scores = {}
    for col, spec in baseline_stats.items():
        if col not in current.columns:
            continue
        if spec["type"] == "categorical":
            scores[col] = _psi_categorical(spec["frequencies"], current[col])
        else:
            scores[col] = _psi_numeric(spec["bin_edges"], current[col])
    return scores


def evaluate_drift(baseline_stats: dict, current: pd.DataFrame, psi_threshold: float = 0.2) -> dict:
    """Flag drift if any feature's PSI against the frozen baseline breaches the threshold."""
    scores = compute_psi(baseline_stats, current)
    breached = {col: psi for col, psi in scores.items() if psi > psi_threshold}
    return {
        "drift_detected": bool(breached),
        "threshold": psi_threshold,
        "feature_psi": scores,
        "breached_features": breached,
    }


def _decile_edges(values: pd.Series) -> list[float]:
    """Equal-frequency bin edges so the baseline's own expected proportion per bucket is uniform."""
    quantiles = np.linspace(0, 1, _N_BUCKETS + 1)
    edges = np.unique(np.quantile(values.dropna().astype(float), quantiles))
    if len(edges) < 2:
        # A near-constant column collapses every quantile to the same value, leaving no
        # bucket boundary to speak of — fall back to a single (-inf, inf) bucket rather
        # than dividing by zero buckets downstream in _psi_numeric.
        return [-np.inf, np.inf]
    edges[0], edges[-1] = -np.inf, np.inf  # absorb out-of-range values at inference time
    return edges.tolist()


def _category_frequencies(values: pd.Series) -> dict[str, float]:
    return values.astype(str).value_counts(normalize=True).to_dict()


def _psi_numeric(bin_edges: list[float], current: pd.Series) -> float:
    if len(bin_edges) < 2:
        # A baseline frozen before the near-constant-column fix (or otherwise degenerate)
        # can still carry a single-element bin_edges array — fall back to the same
        # single (-inf, inf) bucket _decile_edges now produces for that case.
        bin_edges = [-np.inf, np.inf]
    n_buckets = len(bin_edges) - 1
    expected_pct = np.full(n_buckets, 1.0 / n_buckets)
    counts, _ = np.histogram(current.dropna().astype(float), bins=bin_edges)
    actual_pct = counts / max(counts.sum(), 1)
    return _psi(expected_pct, actual_pct)


def _psi_categorical(baseline_freqs: dict[str, float], current: pd.Series) -> float:
    categories = set(baseline_freqs) | set(current.astype(str).unique())
    current_freqs = current.astype(str).value_counts(normalize=True).to_dict()
    expected_pct = np.array([baseline_freqs.get(c, 0.0) for c in categories])
    actual_pct = np.array([current_freqs.get(c, 0.0) for c in categories])
    return _psi(expected_pct, actual_pct)


def _psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    expected_pct = np.clip(expected_pct, _EPSILON, None)
    actual_pct = np.clip(actual_pct, _EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
