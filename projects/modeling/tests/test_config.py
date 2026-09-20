"""Unit tests for the modeling settings and the typed HPO search space."""

import pytest
from modeling.config import ModelingSettings, XGBFixedParams, get_settings
from pydantic import ValidationError


def test_defaults_need_no_environment():
    # The pipeline is compiled in CI, where none of these are set — a required field here
    # would break the build rather than the run.
    settings = ModelingSettings()
    assert settings.hpo_n_trials == 100
    assert set(settings.hpo_search_space) >= {"max_depth", "learning_rate"}


def test_search_space_is_overridden_as_json(monkeypatch):
    monkeypatch.setenv(
        "MODELING_HPO_SEARCH_SPACE", '{"max_depth": {"type": "int", "low": 3, "high": 12}}'
    )
    get_settings.cache_clear()
    try:
        space = get_settings().hpo_search_space
        assert list(space) == ["max_depth"]
        assert space["max_depth"].high == 12
    finally:
        get_settings.cache_clear()


def test_inverted_range_is_rejected():
    # Optuna reports this only as an opaque sampler failure partway into a paid HPO run.
    with pytest.raises(ValidationError):
        ModelingSettings(hpo_search_space={"max_depth": {"type": "int", "low": 10, "high": 3}})


def test_categorical_support_cannot_be_switched_off():
    # Training without it produces a model whose own serving container rejects its input
    # dtypes at predict time — see serving.predict.load_model.
    with pytest.raises(ValidationError):
        XGBFixedParams(enable_categorical=False)


def test_extra_booster_arguments_pass_through():
    assert XGBFixedParams(max_bin=128).model_dump()["max_bin"] == 128
