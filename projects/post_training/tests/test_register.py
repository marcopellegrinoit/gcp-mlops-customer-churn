"""Unit tests for the register_or_reject stage (no GCP credentials required)."""

import post_training.register as register_module
import pytest
from data_contracts import EvaluationMetrics, ModelMetrics, PromotionDecision
from google.cloud import exceptions as gcs_exceptions
from post_training.register import register_or_reject


class _FakeModel:
    """Stands in for aiplatform.Model, whose `name` is the bare model ID, unprefixed."""

    def __init__(self, name, labels=None, resource_name=None, version_id="1"):
        self.name = name
        self.resource_name = resource_name or f"projects/p/locations/europe-west1/models/{name}"
        self.labels = labels or {}
        self.version_id = version_id
        self.updated_labels = None

    def update(self, labels):
        """The unversioned write path, as Model.list() objects expose it.

        Merges new keys but leaves existing values alone — so a demotion that calls
        model.update() directly, as the original code did, fails the tests below instead of
        silently appearing to work.
        """
        merged = dict(labels)
        merged.update(self.labels)
        self.labels = merged
        self.updated_labels = dict(labels)


class _ModelRef:
    """A handle on a model, reached either by its versioned or unversioned resource name.

    Reproduces the Vertex AI behaviour that hid the dual-champion bug: a labels update sent
    to an *unversioned* name merges new keys in but leaves an existing key's value alone,
    while reporting success. Only the version-qualified name actually changes a value. A fake
    that let either path write would make the regression test below pass against the very
    code it is meant to catch.
    """

    def __init__(self, model, versioned):
        self._model = model
        self._versioned = versioned

    @property
    def labels(self):
        return self._model.labels

    def update(self, labels):
        if self._versioned:
            self._model.labels = dict(labels)
        else:
            merged = dict(labels)
            merged.update(self._model.labels)  # existing values win, as the live API does
            self._model.labels = merged
        self._model.updated_labels = dict(labels)


class _FakeModelApi:
    """Stands in for the aiplatform.Model class: constructor lookup plus list/upload."""

    def __init__(self, models, uploaded=None):
        self._models = list(models)
        self._uploaded = uploaded

    def __call__(self, model_name):
        for m in self._models:
            if model_name == f"{m.resource_name}@{m.version_id}":
                return _ModelRef(m, versioned=True)
            if model_name == m.resource_name:
                return _ModelRef(m, versioned=False)
        raise LookupError(model_name)

    def list(self, **kwargs):
        return list(self._models)

    def upload(self, **kwargs):
        return self._uploaded


_METRICS = EvaluationMetrics(
    threshold=0.42,
    challenger_metrics=ModelMetrics(pr_auc=0.8, roc_auc=0.85, f1=0.7, threshold=0.42),
    champion_metrics=ModelMetrics(pr_auc=0.75, roc_auc=0.82, f1=0.68, threshold=0.4),
    shap_rank_correlation=0.9,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        metrics=_METRICS,
        decision=PromotionDecision(promote=True),
        challenger_uri="gs://bucket/challenger",
        serving_image_uri="gcr.io/p/serving:latest",
        project_id="proj",
        region="europe-west1",
        model_display_name="churn-predictor",
        experiment_name="churn-experiment",
        consecutive_rejection_key="consecutive_rejections",
    )
    kwargs.update(overrides)
    return kwargs


def _patch_experiment_logging(monkeypatch):
    monkeypatch.setattr(register_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(register_module.aiplatform, "start_run", lambda name: None)
    monkeypatch.setattr(register_module.aiplatform, "log_params", lambda params: None)
    monkeypatch.setattr(register_module.aiplatform, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(register_module.aiplatform, "end_run", lambda: None)


def _patch_state(monkeypatch, current_count, uploads=None):
    monkeypatch.setattr(
        register_module, "download_json", lambda uri: {"consecutive_rejections": current_count}
    )
    monkeypatch.setattr(
        register_module,
        "upload_text",
        lambda uri, content: uploads.append((uri, content)) if uploads is not None else None,
    )


# ---------------------------------------------------------------------------
# promotion
# ---------------------------------------------------------------------------


def test_promoted_challenger_is_uploaded_with_correct_container_spec(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    upload_calls = {}

    def fake_upload(**kwargs):
        upload_calls.update(kwargs)
        return _FakeModel(name="456", resource_name="projects/p/locations/europe-west1/models/456")

    monkeypatch.setattr(register_module.aiplatform.Model, "upload", staticmethod(fake_upload))
    monkeypatch.setattr(register_module.aiplatform.Model, "list", staticmethod(lambda **kwargs: []))
    uploads = []
    _patch_state(monkeypatch, current_count=2, uploads=uploads)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=True)))

    assert result.promoted is True
    assert result.model_version == "projects/p/locations/europe-west1/models/456"
    assert upload_calls["display_name"] == "churn-predictor"
    assert upload_calls["artifact_uri"] == "gs://bucket/challenger"
    assert upload_calls["serving_container_image_uri"] == "gcr.io/p/serving:latest"
    assert upload_calls["serving_container_predict_route"] == "/predict"
    assert upload_calls["serving_container_health_route"] == "/health"
    assert upload_calls["serving_container_ports"] == [8080]
    assert upload_calls["serving_container_environment_variables"] == {
        "THRESHOLD": "0.42",
        "PROJECT_ID": "proj",
    }
    assert upload_calls["labels"] == {"role": "champion"}


def test_promotion_resets_rejection_count_and_clears_alert(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    monkeypatch.setattr(
        register_module.aiplatform.Model,
        "upload",
        staticmethod(lambda **kwargs: _FakeModel(name="new")),
    )
    monkeypatch.setattr(register_module.aiplatform.Model, "list", staticmethod(lambda **kwargs: []))
    uploads = []
    _patch_state(monkeypatch, current_count=2, uploads=uploads)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=True)))

    assert result.consecutive_rejections == 0
    assert result.feature_review_alert is False
    assert len(uploads) == 1
    _, content = uploads[0]
    assert content == '{"consecutive_rejections": 0}'


def test_promotion_demotes_previous_champion_but_leaves_other_roles_alone(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    new_model = _FakeModel(
        name="new",
        labels={"role": "champion"},
        resource_name="projects/p/locations/europe-west1/models/456",
    )
    old_champion = _FakeModel(
        name="old",
        labels={"role": "champion"},
        resource_name="projects/p/locations/europe-west1/models/123",
    )
    already_retired = _FakeModel(
        name="retired",
        labels={"role": "retired"},
        resource_name="projects/p/locations/europe-west1/models/111",
    )
    rejected = _FakeModel(
        name="rejected",
        labels={"role": "rejected"},
        resource_name="projects/p/locations/europe-west1/models/999",
    )

    monkeypatch.setattr(
        register_module.aiplatform,
        "Model",
        _FakeModelApi([old_champion, already_retired, rejected, new_model], uploaded=new_model),
    )
    _patch_state(monkeypatch, current_count=0)

    register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=True)))

    assert old_champion.labels["role"] == "retired"
    assert already_retired.updated_labels is None
    assert rejected.updated_labels is None
    assert new_model.labels["role"] == "champion"  # excluded from demotion by resource_name


def test_demotion_targets_the_version_qualified_resource(monkeypatch):
    # The dual-champion bug: Model.list() yields unversioned names, and a labels update sent
    # to an unversioned name cannot change an existing key's value even though it reports
    # success. Demoting via models/<id> alone leaves two models labelled role=champion.
    _patch_experiment_logging(monkeypatch)
    new_model = _FakeModel(
        name="new",
        labels={"role": "champion"},
        resource_name="projects/p/locations/europe-west1/models/456",
    )
    old_champion = _FakeModel(
        name="old",
        labels={"role": "champion"},
        resource_name="projects/p/locations/europe-west1/models/123",
        version_id="3",
    )
    monkeypatch.setattr(
        register_module.aiplatform,
        "Model",
        _FakeModelApi([old_champion, new_model], uploaded=new_model),
    )
    _patch_state(monkeypatch, current_count=0)

    register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=True)))

    # Exactly one champion remains, which is the whole point of the invariant.
    assert old_champion.labels["role"] == "retired"
    assert new_model.labels["role"] == "champion"


def test_demotion_that_silently_no_ops_raises(monkeypatch):
    # The original failure was silent: the update succeeded, updateTime moved, and the label
    # did not change. Nothing downstream noticed because every champion lookup sorts by
    # create_time desc and takes the first match, so the newest model still won.
    _patch_experiment_logging(monkeypatch)
    new_model = _FakeModel(
        name="new",
        labels={"role": "champion"},
        resource_name="projects/p/locations/europe-west1/models/456",
    )
    stubborn = _FakeModel(
        name="old",
        labels={"role": "champion"},
        resource_name="projects/p/locations/europe-west1/models/123",
    )

    class _NeverWrites(_FakeModelApi):
        """A registry whose writes never take effect, versioned or not."""

        def __call__(self, model_name):
            ref = super().__call__(model_name)
            ref._versioned = False
            return ref

    monkeypatch.setattr(
        register_module.aiplatform,
        "Model",
        _NeverWrites([stubborn, new_model], uploaded=new_model),
    )
    _patch_state(monkeypatch, current_count=0)

    with pytest.raises(RuntimeError, match="Failed to demote previous champion"):
        register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=True)))


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


def test_rejected_challenger_is_never_uploaded(monkeypatch):
    _patch_experiment_logging(monkeypatch)

    def fail_upload(**kwargs):
        raise AssertionError("a rejected challenger must never be registered")

    monkeypatch.setattr(register_module.aiplatform.Model, "upload", staticmethod(fail_upload))
    _patch_state(monkeypatch, current_count=1)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=False)))

    assert result.promoted is False
    assert result.model_version is None


def test_rejection_increments_consecutive_rejection_count(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    uploads = []
    _patch_state(monkeypatch, current_count=1, uploads=uploads)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=False)))

    assert result.consecutive_rejections == 2
    assert uploads[0][1] == '{"consecutive_rejections": 2}'


def test_rejection_count_starts_at_zero_when_no_state_file_exists(monkeypatch):
    _patch_experiment_logging(monkeypatch)

    def raise_not_found(uri):
        raise gcs_exceptions.NotFound("no state file yet")

    monkeypatch.setattr(register_module, "download_json", raise_not_found)
    monkeypatch.setattr(register_module, "upload_text", lambda uri, content: None)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=False)))

    assert result.consecutive_rejections == 1


def test_feature_review_alert_fires_at_max_consecutive_rejections(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    _patch_state(monkeypatch, current_count=2)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=False)))

    assert result.consecutive_rejections == 3
    assert result.feature_review_alert is True


def test_feature_review_alert_does_not_fire_below_threshold(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    _patch_state(monkeypatch, current_count=1)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=False)))

    assert result.consecutive_rejections == 2
    assert result.feature_review_alert is False


# ---------------------------------------------------------------------------
# result payload / experiment logging
# ---------------------------------------------------------------------------


def test_result_carries_through_evaluation_metrics(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    _patch_state(monkeypatch, current_count=0)

    result = register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=False)))

    assert result.challenger_metrics == _METRICS.challenger_metrics
    assert result.champion_metrics == _METRICS.champion_metrics
    assert result.threshold == 0.42
    assert result.shap_rank_correlation == 0.9


def test_log_run_records_promotion_outcome_to_experiments(monkeypatch):
    calls = {}
    monkeypatch.setattr(register_module.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        register_module.aiplatform, "start_run", lambda name: calls.setdefault("run_name", name)
    )
    monkeypatch.setattr(
        register_module.aiplatform, "log_params", lambda params: calls.setdefault("params", params)
    )
    monkeypatch.setattr(
        register_module.aiplatform,
        "log_metrics",
        lambda metrics: calls.setdefault("metrics", metrics),
    )
    monkeypatch.setattr(
        register_module.aiplatform, "end_run", lambda: calls.setdefault("ended", True)
    )
    monkeypatch.setattr(
        register_module.aiplatform.Model,
        "upload",
        staticmethod(
            lambda **kwargs: _FakeModel(
                name="new", resource_name="projects/p/locations/europe-west1/models/456"
            )
        ),
    )
    monkeypatch.setattr(register_module.aiplatform.Model, "list", staticmethod(lambda **kwargs: []))
    _patch_state(monkeypatch, current_count=0)

    register_or_reject(**_base_kwargs(decision=PromotionDecision(promote=True)))

    assert calls["params"]["promoted"] == 1
    assert calls["params"]["consecutive_rejections"] == 0
    assert calls["params"]["model_version"] == "projects/p/locations/europe-west1/models/456"
    assert calls["metrics"] == {"threshold": 0.42}
    assert calls["ended"] is True
