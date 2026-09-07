"""Population Stability Index (PSI) drift detection shared by training and monitoring.

The trainer freezes a baseline distribution per feature at training time into the model
artifact's metadata.json. The drift monitor later rebuilds the same shape from a live
snapshot and compares against that frozen baseline — both sides must use this module so
the binning is identical on each side of the comparison.

Numeric columns take one of two baseline shapes, because a single binning strategy does
not fit both kinds of feature this platform produces:

* ``discrete`` — a frequency table over the observed values, used for low-cardinality
  columns (``renewal_count``, ``contact_requests_last_30d``, ``total_events``, and the
  rate columns that collapse to 0.0/1.0 for customers with a single event). Quantile
  binning is actively wrong for these: a mass point makes several deciles identical,
  ``np.unique`` collapses them, and the one or two surviving buckets hold nothing like
  an equal share of the baseline.
* ``numeric`` — decile bin edges *plus the baseline proportion actually measured in each
  bucket*. The proportions are stored rather than assumed uniform. Assuming uniformity
  is only valid when no edges collapse; when they do, the assumed-vs-actual gap becomes
  a permanent PSI floor and the feature reports drift against the very rows the baseline
  was built from. See ``test_no_drift_against_own_training_data``.

Every spec also carries ``monitored``. A column that yields a single bucket (constant, or
one distinct value) has no distributional signal at all; its PSI is structurally 0.0,
which reads as "healthy" when it really means "not being watched". Flagging it lets
callers report unmonitored coverage instead of silently passing it.

Missingness is part of the distribution, not something to drop before measuring it. Each
spec records ``null_rate`` and the comparison carries an extra bucket for it, so a feature
whose null rate moves is caught even when the values that *are* present look unchanged.
The missingness here is neither random nor ignorable — ``engagement_score`` is absent for
offline-only customers (MAR) and ``days_since_last_successful_payment`` is absent for
exactly those customers who never paid successfully (MNAR, and correlated with the target)
— so a shift in how often a column is null is a real population change, and often an
earlier signal than the values themselves.
"""

import logging

import numpy as np
import pandas as pd

from ml_common.preprocess import CATEGORICAL_COLS

log = logging.getLogger(__name__)

_EPSILON = 1e-4  # avoids div-by-zero / log(0) when a bucket is empty on either side
_N_BUCKETS = 10

# At or below this many distinct values, decile binning cannot represent the column and a
# frequency table is both exact and cheaper. Above it, a continuous column is assumed.
_MAX_DISCRETE_CARDINALITY = 20

# Sampling-noise calibration. A daily snapshot is a finite sample, so PSI is never exactly
# zero even against a perfectly stable population; the smaller the sample, the larger the
# noise. _BOOTSTRAP_ALPHA is the per-feature false-alarm rate we accept from noise alone.
# Multiplicity across features is deliberately NOT Bonferroni-corrected here — the drift
# monitor's N-of-M persistence rule is a far stronger control (two independent 1-in-100
# events on the same feature in three runs is ~3e-4) and does not need the deep bootstrap
# tail that a corrected alpha would demand.
_BOOTSTRAP_DRAWS = 1000
_BOOTSTRAP_ALPHA = 0.01
_BOOTSTRAP_SEED = 0  # fixed so a given (baseline, n) always yields the same threshold


def compute_baseline_stats(X: pd.DataFrame) -> dict:
    """Freeze a per-feature baseline distribution from training data for later PSI comparison."""
    stats: dict = {}
    for col in X.columns:
        if col in CATEGORICAL_COLS:
            stats[col] = _categorical_baseline(X[col])
        elif pd.api.types.is_numeric_dtype(X[col]):
            stats[col] = _numeric_baseline(X[col])
    return stats


def compute_psi(baseline_stats: dict, current: pd.DataFrame) -> dict[str, float]:
    """Return the PSI score for every baseline feature present in current."""
    scores = {}
    for col, spec in baseline_stats.items():
        if col not in current.columns:
            continue
        expected_pct, actual_pct = _distributions(spec, current[col])
        scores[col] = _psi(expected_pct, actual_pct)
    return scores


def psi_noise_floor(spec: dict, n_current: int) -> float:
    """Return the PSI this feature reaches from sampling noise alone at sample size n_current.

    Draws repeated multinomial samples of size n_current from the baseline's own bucket
    proportions and returns the (1 - alpha) quantile of the resulting PSI values. This
    needs only the frozen proportions, not the original training rows, so it can be
    computed at check time once the live sample size is known.
    """
    expected_pct = np.asarray(_expected_vector(spec), dtype=float)
    if expected_pct.size < 2 or n_current <= 0:
        return 0.0

    expected_pct = expected_pct / expected_pct.sum()
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    sampled_pct = rng.multinomial(n_current, expected_pct, size=_BOOTSTRAP_DRAWS) / n_current

    expected_clipped = np.clip(expected_pct, _EPSILON, None)
    sampled_clipped = np.clip(sampled_pct, _EPSILON, None)
    psis = np.sum(
        (sampled_clipped - expected_clipped) * np.log(sampled_clipped / expected_clipped), axis=1
    )
    return float(np.quantile(psis, 1 - _BOOTSTRAP_ALPHA))


def evaluate_drift(baseline_stats: dict, current: pd.DataFrame, psi_threshold: float = 0.2) -> dict:
    """Flag drift where a feature's PSI breaches both the configured floor and its noise floor.

    A feature alarms only when the shift is large enough to matter (psi_threshold, an
    effect size the business chose) *and* large enough to be distinguishable from the
    sampling noise of this particular snapshot size (psi_noise_floor). Taking the max of
    the two means small daily snapshots stop firing on noise without loosening the
    threshold for the features that have enough data to support it.
    """
    n_current = len(current)
    feature_psi: dict[str, float] = {}
    feature_thresholds: dict[str, float] = {}
    null_rates: dict[str, dict[str, float]] = {}
    breached: dict[str, float] = {}
    unmonitored: list[str] = []

    for col, spec in baseline_stats.items():
        if col not in current.columns:
            continue

        expected_pct, actual_pct = _distributions(spec, current[col])
        psi = _psi(expected_pct, actual_pct)
        threshold = max(psi_threshold, psi_noise_floor(spec, n_current))
        feature_psi[col] = psi
        feature_thresholds[col] = threshold

        # Reported alongside the PSI because missingness is the part of a breach that a
        # human most often needs to see explicitly: "PSI 0.4" and "null rate went 3% -> 40%"
        # call for very different responses, and the second is usually a broken upstream
        # join rather than a population shift.
        if _tracks_nulls(spec):
            null_rates[col] = {
                "baseline": float(spec["null_rate"]),
                "current": _null_rate(current[col]),
            }

        if not _is_monitored(spec):
            unmonitored.append(col)
        elif psi > threshold:
            breached[col] = psi

    if unmonitored:
        log.warning(
            "Features carrying no usable drift signal this run (not evaluated): %s",
            ", ".join(sorted(unmonitored)),
        )

    return {
        "drift_detected": bool(breached),
        "threshold": psi_threshold,
        "feature_psi": feature_psi,
        "feature_thresholds": feature_thresholds,
        "feature_null_rates": null_rates,
        "breached_features": breached,
        "unmonitored_features": sorted(unmonitored),
        "current_sample_size": n_current,
    }


def _categorical_baseline(values: pd.Series) -> dict:
    frequencies = _category_frequencies(values.dropna())
    null_rate = _null_rate(values)
    return {
        "type": "categorical",
        "frequencies": frequencies,
        "null_rate": null_rate,
        # A single-category column still carries signal if its null rate can move.
        "monitored": len(frequencies) > 1 or 0.0 < null_rate < 1.0,
    }


def _numeric_baseline(values: pd.Series) -> dict:
    clean = values.dropna().astype(float)
    null_rate = _null_rate(values)
    if clean.empty:
        # Entirely null at training time: there is no value distribution to compare against,
        # and a column in that state is a data-quality failure rather than a drift signal.
        return {"type": "discrete", "frequencies": {}, "null_rate": null_rate, "monitored": False}

    if clean.nunique() <= _MAX_DISCRETE_CARDINALITY:
        return {
            "type": "discrete",
            "frequencies": _value_frequencies(clean),
            "null_rate": null_rate,
            "monitored": clean.nunique() > 1 or 0.0 < null_rate < 1.0,
        }

    bin_edges = _decile_edges(clean)
    return {
        "type": "numeric",
        "bin_edges": bin_edges,
        "expected_pct": _bucket_proportions(clean, bin_edges),
        "null_rate": null_rate,
        "monitored": len(bin_edges) > 2 or 0.0 < null_rate < 1.0,  # >2 edges == >1 bucket
    }


def _null_rate(values: pd.Series) -> float:
    return float(values.isna().mean()) if len(values) else 0.0


def _is_monitored(spec: dict) -> bool:
    """Whether this feature's baseline carries enough information to evaluate drift against.

    Baselines frozen before expected_pct was stored carry bin_edges alone, and the true
    proportions behind those edges are unrecoverable from the artifact — evaluating them
    would mean re-assuming uniformity, the exact defect this module now avoids. Such
    numeric specs are reported as unmonitored until the baseline is rebuilt (see
    scripts/rebuild_champion_baseline.py, which recomputes it from ml.split_assignments).
    """
    if "monitored" in spec:
        return bool(spec["monitored"])
    return spec.get("type") != "numeric"


def _decile_edges(values: pd.Series) -> list[float]:
    """Decile bin edges; ties collapse, so the surviving buckets are not equal-frequency."""
    quantiles = np.linspace(0, 1, _N_BUCKETS + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        # A near-constant column collapses every quantile to the same value, leaving no
        # bucket boundary to speak of — fall back to a single (-inf, inf) bucket rather
        # than dividing by zero buckets downstream.
        return [-np.inf, np.inf]
    edges[0], edges[-1] = -np.inf, np.inf  # absorb out-of-range values at inference time
    return edges.tolist()


def _bucket_proportions(values: pd.Series, bin_edges: list[float]) -> list[float]:
    """Measure what share of the baseline actually falls in each bucket."""
    counts, _ = np.histogram(values, bins=bin_edges)
    return (counts / max(counts.sum(), 1)).tolist()


def _category_frequencies(values: pd.Series) -> dict[str, float]:
    return values.astype(str).value_counts(normalize=True).to_dict()


def _value_frequencies(values: pd.Series) -> dict[str, float]:
    """Frequency table for a discrete numeric column, keyed for JSON round-tripping.

    metadata.json is JSON, which has no numeric keys — json.dumps would stringify them on
    write and they would come back as strings on read. Keying on str(float(v)) up front
    makes both sides of the comparison agree without depending on that coercion.
    """
    return values.map(_numeric_key).value_counts(normalize=True).to_dict()


def _numeric_key(value: float) -> str:
    return str(float(value))


def _value_proportions(spec: dict) -> list[float]:
    """The baseline's proportion per bucket among present values — no null bucket."""
    if spec["type"] in ("categorical", "discrete"):
        return list(spec["frequencies"].values())

    expected_pct = spec.get("expected_pct")
    if expected_pct is None:
        # Legacy baseline: proportions were never stored. Report uniform so a PSI number
        # still appears in the logs, but _is_monitored keeps it out of the breach decision.
        n_buckets = max(len(spec.get("bin_edges", [])) - 1, 1)
        return [1.0 / n_buckets] * n_buckets
    return list(expected_pct)


def _expected_vector(spec: dict) -> list[float]:
    """The full baseline distribution the live column is compared against, nulls included.

    This is the vector the noise-floor bootstrap samples from, so it has to span exactly the
    buckets _distributions produces — value buckets rescaled to the non-null mass, plus the
    null bucket — or the simulated PSI would be computed over a different support than the
    real one and the calibrated threshold would not apply.
    """
    vector = _value_proportions(spec)
    if _tracks_nulls(spec):
        null_rate = float(spec["null_rate"])
        vector = [p * (1.0 - null_rate) for p in vector] + [null_rate]
    return vector


def _tracks_nulls(spec: dict) -> bool:
    """Whether this spec records missingness.

    Baselines frozen before null_rate was stored do not, and for those the old behaviour is
    preserved exactly: numeric columns drop nulls, and a categorical column's nulls fall into
    a "nan" pseudo-category via astype(str). Retrofitting a null bucket onto them would
    compare a live null rate against an expectation of zero and manufacture a breach.
    """
    return "null_rate" in spec


def _distributions(spec: dict, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align the baseline and the live column onto a shared set of buckets."""
    tracks_nulls = _tracks_nulls(spec)
    if spec["type"] == "categorical":
        keys = current.dropna().astype(str) if tracks_nulls else current.astype(str)
        expected_pct, actual_pct = _frequency_distributions(spec["frequencies"], keys)
    elif spec["type"] == "discrete":
        keys = current.dropna().astype(float).map(_numeric_key)
        expected_pct, actual_pct = _frequency_distributions(spec["frequencies"], keys)
    else:
        expected_pct, actual_pct = _binned_distributions(spec, current)

    if not tracks_nulls:
        return expected_pct, actual_pct
    return _append_null_bucket(spec, expected_pct, actual_pct, current)


def _append_null_bucket(
    spec: dict, expected_pct: np.ndarray, actual_pct: np.ndarray, current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale both value distributions to their non-null mass and add a bucket for nulls.

    Each side's buckets hold proportions *among present values*, so both are scaled by their
    own non-null share before the null bucket is appended. The result is a proper
    distribution over (values..., missing) on both sides, and a pure shift in missingness
    registers even when the present values are distributed identically.
    """
    baseline_null = float(spec["null_rate"])
    current_null = _null_rate(current)
    expected_pct = np.append(expected_pct * (1.0 - baseline_null), baseline_null)
    actual_pct = np.append(actual_pct * (1.0 - current_null), current_null)
    return expected_pct, actual_pct


def _frequency_distributions(
    baseline_freqs: dict[str, float], current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    categories = sorted(set(baseline_freqs) | set(current.unique()))
    current_freqs = current.value_counts(normalize=True).to_dict()
    expected_pct = np.array([baseline_freqs.get(c, 0.0) for c in categories])
    actual_pct = np.array([current_freqs.get(c, 0.0) for c in categories])
    return expected_pct, actual_pct


def _binned_distributions(spec: dict, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    bin_edges = spec.get("bin_edges") or []
    if len(bin_edges) < 2:
        # A degenerate baseline can still carry a single-element bin_edges array — fall
        # back to the same single (-inf, inf) bucket _decile_edges produces for that case.
        bin_edges = [-np.inf, np.inf]
    # Value buckets only — _distributions appends the null bucket to both sides afterwards.
    expected_pct = np.asarray(_value_proportions(spec), dtype=float)
    counts, _ = np.histogram(current.dropna().astype(float), bins=bin_edges)
    actual_pct = counts / max(counts.sum(), 1)
    return expected_pct, actual_pct


def _psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    expected_pct = np.clip(expected_pct, _EPSILON, None)
    actual_pct = np.clip(actual_pct, _EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
