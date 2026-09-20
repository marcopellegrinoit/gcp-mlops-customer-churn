"""Frozen per-feature baseline distributions carried in a model artifact's metadata.json.

The trainer freezes one spec per feature at training time; the drift monitor rebuilds the
same shape from a live snapshot months later and compares against it. The two sides are
separate containers, separate images, and often separate releases, so this is a genuine
wire format rather than an in-process data structure — and the legacy tolerances below are
load-bearing, not decoration.

Three shapes, discriminated on ``type`` (see ml_common.drift for why the numeric/discrete
split exists at all):

* ``categorical`` / ``discrete`` — a frequency table over observed values.
* ``numeric`` — decile bin edges plus the proportion measured in each bucket.

**Reading is deliberately tolerant.** A champion registered before a field existed is still
serving, and its artifact cannot be rewritten without either retraining it or running
scripts/rebuild_champion_baseline.py. Three fields are therefore optional, and *absent*
means something different from any value they could hold:

* ``monitored`` absent — the spec predates the unmonitored-coverage flag; fall back to
  "numeric specs are untrustworthy, everything else is watched" (see ``is_monitored``).
* ``null_rate`` absent — the spec predates null-bucket tracking; PSI must then be computed
  the old way, dropping nulls, or a live null rate would be compared against an implied
  zero and manufacture a breach (see ``tracks_nulls``).
* ``expected_pct`` absent on a numeric spec — per-bucket proportions were never stored and
  are unrecoverable from the artifact; uniformity must not be re-assumed, which is why such
  a spec reports ``is_monitored`` False.

``extra="allow"`` for the same reason in the other direction: a newer trainer adding a
field must not break an older reader that has not been redeployed yet.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

type BaselineSpec = Annotated[
    CategoricalBaseline | DiscreteBaseline | NumericBaseline, Field(discriminator="type")
]


class _BaselineBase(BaseModel):
    """Fields and legacy-tolerance rules shared by every baseline shape."""

    model_config = ConfigDict(frozen=True, extra="allow")

    null_rate: float | None = None
    monitored: bool | None = None

    @property
    def tracks_nulls(self) -> bool:
        """Whether this spec records missingness, and so carries a null bucket in its PSI."""
        return self.null_rate is not None

    @property
    def is_monitored(self) -> bool:
        """Whether this baseline carries enough information to gate retraining on."""
        if self.monitored is not None:
            return self.monitored
        return self.type != "numeric"

    @property
    def value_proportions(self) -> list[float]:
        """The baseline's proportion per bucket among present values — no null bucket."""
        raise NotImplementedError

    @property
    def value_cardinality(self) -> int:
        """How many distinct value buckets the baseline held, ignoring missingness."""
        raise NotImplementedError

    @property
    def expected_vector(self) -> list[float]:
        """The full distribution a live column is compared against, null bucket included.

        Spans exactly the buckets ml_common.drift builds at check time — value buckets
        rescaled to the non-null mass, plus the null bucket — so the noise-floor bootstrap
        samples from the same support the real PSI is computed over.
        """
        vector = self.value_proportions
        if not self.tracks_nulls:
            return vector
        null_rate = float(self.null_rate)
        return [p * (1.0 - null_rate) for p in vector] + [null_rate]


class _FrequencyBaseline(_BaselineBase):
    """A frequency table over observed values, shared by the categorical and discrete shapes."""

    frequencies: dict[str, float] = Field(default_factory=dict)

    @property
    def value_proportions(self) -> list[float]:
        """The baseline's proportion per observed value."""
        return list(self.frequencies.values())

    @property
    def value_cardinality(self) -> int:
        """How many distinct values the baseline observed."""
        return len(self.frequencies)


class CategoricalBaseline(_FrequencyBaseline):
    """Frequency table over a categorical column's training-time values."""

    type: Literal["categorical"] = "categorical"


class DiscreteBaseline(_FrequencyBaseline):
    """Frequency table over a low-cardinality numeric column's training-time values.

    Keys are ``str(float(value))`` rather than numbers: metadata.json is JSON, which has no
    numeric keys, so both sides of the comparison agree on the string form up front instead
    of depending on json.dumps' coercion.
    """

    type: Literal["discrete"] = "discrete"


class NumericBaseline(_BaselineBase):
    """Decile bin edges plus the proportion of the baseline actually measured in each bucket."""

    type: Literal["numeric"] = "numeric"
    bin_edges: list[float] = Field(default_factory=list)
    expected_pct: list[float] | None = None

    @property
    def value_proportions(self) -> list[float]:
        """The baseline's proportion per bucket, or uniform for a spec frozen without them.

        The uniform fallback exists only so a PSI number still appears in the logs for a
        legacy spec — ``is_monitored`` is False for exactly those specs, so the number it
        produces never reaches the breach decision.
        """
        if self.expected_pct is None:
            return [1.0 / self.n_buckets] * self.n_buckets
        return list(self.expected_pct)

    @property
    def value_cardinality(self) -> int:
        """How many buckets the decile edges describe."""
        return max(len(self.bin_edges) - 1, 0)

    @property
    def n_buckets(self) -> int:
        """Bucket count, floored at 1 so a degenerate edge list cannot divide by zero."""
        return max(len(self.bin_edges) - 1, 1)

    @property
    def is_monitored(self) -> bool:
        """False for a spec frozen without per-bucket proportions, whatever it claims.

        Those proportions are unrecoverable from the artifact, so evaluating the spec would
        mean re-assuming uniformity — the defect that made the monitor breach against its own
        training data. A legacy spec that predates ``monitored`` reaches the same answer via
        the base class; this covers one that carries ``monitored`` but still no proportions.
        """
        return self.expected_pct is not None and super().is_monitored


_BASELINE_STATS_ADAPTER = TypeAdapter(dict[str, BaselineSpec])


def parse_baseline_stats(raw: Mapping[str, object]) -> dict[str, BaselineSpec]:
    """Validate a raw ``baseline_stats`` mapping into typed specs.

    Already-typed input passes straight through.

    Callers that hold a validated ModelMetadata already have typed specs. This exists for
    the ones that do not — a test fixture, a notebook, a hand-built dict — so the drift
    functions can take either without every call site remembering which it holds.
    """
    return _BASELINE_STATS_ADAPTER.validate_python(raw)
