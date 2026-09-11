"""Hyperparameter search space and training configuration, owned by the data science team.

Overridable per deployment through ``MODELING_``-prefixed environment variables. The scalar
knobs take the obvious form (``MODELING_HPO_N_TRIALS=50``); the two structured ones take
JSON, which is how pydantic-settings reads any complex field from the environment:

    MODELING_HPO_SEARCH_SPACE='{"max_depth": {"type": "int", "low": 3, "high": 12}}'

so a search space can be narrowed for a cheap smoke run, or widened for a deliberate
re-tune, without touching the trainer image. Every field has a default, so importing this
module never requires an environment — the pipeline is compiled in CI, where none of these
are set.

Inference-time policy (the promotion gate, RANDOM_STATE, TEST_SIZE) lives in
ml_common.config, shared with evaluate and serving.
"""

from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type SearchSpaceParam = Annotated[IntParam | FloatParam, Field(discriminator="type")]


class _ParamBase(BaseModel):
    """Bounds shared by every searchable hyperparameter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float

    @model_validator(mode="after")
    def _check_bounds(self):
        """Reject an inverted range, which Optuna reports only as an opaque sampler error."""
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must not exceed high ({self.high})")
        return self


class IntParam(_ParamBase):
    """An integer hyperparameter searched uniformly over [low, high]."""

    type: Literal["int"] = "int"
    low: int
    high: int


class FloatParam(_ParamBase):
    """A float hyperparameter.

    ``log`` samples on a log scale, for ranges spanning orders of magnitude.
    """

    type: Literal["float"] = "float"
    log: bool = False


class XGBFixedParams(BaseModel):
    """XGBoost arguments held constant across every trial and the final fit.

    ``extra="allow"`` so a deployment can add booster arguments through
    MODELING_XGB_FIXED_PARAMS without a code change.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    tree_method: str = "hist"
    # Not a tunable. XGBoost's native categorical splits are what let the feature matrix keep
    # category dtypes instead of being one-hot encoded, and serving.predict.load_model
    # re-passes enable_categorical=True when it deserialises the booster (the sklearn wrapper
    # does not restore it from the saved model). Training with it off would produce a model
    # whose own serving container rejects its input dtypes at predict time, which is not a
    # failure worth making reachable through an environment variable.
    enable_categorical: Literal[True] = True


_DEFAULT_SEARCH_SPACE: dict[str, SearchSpaceParam] = {
    "max_depth": IntParam(low=3, high=10),
    "learning_rate": FloatParam(low=0.01, high=0.3, log=True),
    "subsample": FloatParam(low=0.5, high=1.0),
    "colsample_bytree": FloatParam(low=0.5, high=1.0),
    "n_estimators": IntParam(low=50, high=500),
}


class ModelingSettings(BaseSettings):
    """HPO and training configuration, read from ``MODELING_``-prefixed env vars."""

    model_config = SettingsConfigDict(env_prefix="MODELING_", extra="ignore", frozen=True)

    hpo_n_trials: int = Field(default=100, gt=0)
    hpo_n_folds: int = Field(default=5, gt=1)
    hpo_search_space: dict[str, SearchSpaceParam] = Field(
        default_factory=lambda: dict(_DEFAULT_SEARCH_SPACE)
    )
    xgb_fixed_params: XGBFixedParams = XGBFixedParams()

    @property
    def fixed_params(self) -> dict[str, Any]:
        """xgb_fixed_params as the plain kwargs dict XGBClassifier takes."""
        return self.xgb_fixed_params.model_dump()


@lru_cache(maxsize=1)
def get_settings() -> ModelingSettings:
    """Return the process-wide modeling settings, read from the environment once."""
    return ModelingSettings()
