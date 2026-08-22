"""Unit tests for the register_or_reject stage (no GCP credentials required)."""

import post_training.register as register_module
from google.cloud import exceptions as gcs_exceptions
from post_training.register import register_or_reject


class _FakeModel:
    """Stands in for aiplatform.Model, whose `name` is the bare model ID, unprefixed."""

    def __init__(self, name, labels=None, resource_name=None):
        self.name = name
        self.resource_name = resource_name or f"projects/p/locations/europe-west1/models/{name}"
        self.labels = labels or {}
        self.updated_labels = None

    def update(self, labels):
        self.labels = labels
        self.updated_labels = labels


def _base_kwargs(**overrides):
    kwargs = dict(
        metrics={
            "threshold": 0.42,
            "challenger_metrics": {"pr_auc": 0.8},
            "champion_metrics": {"pr_auc": 0.75},
            "shap_rank_correlation": 0.9,
        },
        decision={"promote": True},
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

    result = register_or_reject(**_base_kwargs(decision={"promote": True}))

    assert result["promoted"] is True
    assert result["model_version"] == "projects/p/locations/europe-west1/models/456"
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

    result = register_or_reject(**_base_kwargs(decision={"promote": True}))

    assert result["consecutive_rejections"] == 0
    assert result["feature_review_alert"] is False
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
        register_module.aiplatform.Model, "upload", staticmethod(lambda **kwargs: new_model)
    )
    monkeypatch.setattr(
        register_module.aiplatform.Model,
        "list",
        staticmethod(lambda **kwargs: [old_champion, already_retired, rejected, new_model]),
    )
    _patch_state(monkeypatch, current_count=0)

    register_or_reject(**_base_kwargs(decision={"promote": True}))

    assert old_champion.labels["role"] == "retired"
    assert already_retired.updated_labels is None
    assert rejected.updated_labels is None
    assert new_model.labels["role"] == "champion"  # excluded from demotion by resource_name


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------


def test_rejected_challenger_is_never_uploaded(monkeypatch):
    _patch_experiment_logging(monkeypatch)

    def fail_upload(**kwargs):
        raise AssertionError("a rejected challenger must never be registered")

    monkeypatch.setattr(register_module.aiplatform.Model, "upload", staticmethod(fail_upload))
    _patch_state(monkeypatch, current_count=1)

    result = register_or_reject(**_base_kwargs(decision={"promote": False}))

    assert result["promoted"] is False
    assert result["model_version"] is None


def test_rejection_increments_consecutive_rejection_count(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    uploads = []
    _patch_state(monkeypatch, current_count=1, uploads=uploads)

    result = register_or_reject(**_base_kwargs(decision={"promote": False}))

    assert result["consecutive_rejections"] == 2
    assert uploads[0][1] == '{"consecutive_rejections": 2}'


def test_rejection_count_starts_at_zero_when_no_state_file_exists(monkeypatch):
    _patch_experiment_logging(monkeypatch)

    def raise_not_found(uri):
        raise gcs_exceptions.NotFound("no state file yet")

    monkeypatch.setattr(register_module, "download_json", raise_not_found)
    monkeypatch.setattr(register_module, "upload_text", lambda uri, content: None)

    result = register_or_reject(**_base_kwargs(decision={"promote": False}))

    assert result["consecutive_rejections"] == 1


def test_feature_review_alert_fires_at_max_consecutive_rejections(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    _patch_state(monkeypatch, current_count=2)

    result = register_or_reject(**_base_kwargs(decision={"promote": False}))

    assert result["consecutive_rejections"] == 3
    assert result["feature_review_alert"] is True


def test_feature_review_alert_does_not_fire_below_threshold(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    _patch_state(monkeypatch, current_count=1)

    result = register_or_reject(**_base_kwargs(decision={"promote": False}))

    assert result["consecutive_rejections"] == 2
    assert result["feature_review_alert"] is False


# ---------------------------------------------------------------------------
# result payload / experiment logging
# ---------------------------------------------------------------------------


def test_result_carries_through_evaluation_metrics(monkeypatch):
    _patch_experiment_logging(monkeypatch)
    _patch_state(monkeypatch, current_count=0)

    result = register_or_reject(**_base_kwargs(decision={"promote": False}))

    assert result["challenger_metrics"] == {"pr_auc": 0.8}
    assert result["champion_metrics"] == {"pr_auc": 0.75}
    assert result["threshold"] == 0.42
    assert result["shap_rank_correlation"] == 0.9


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

    register_or_reject(**_base_kwargs(decision={"promote": True}))

    assert calls["params"]["promoted"] == 1
    assert calls["params"]["consecutive_rejections"] == 0
    assert calls["params"]["model_version"] == "projects/p/locations/europe-west1/models/456"
    assert calls["metrics"] == {"threshold": 0.42}
    assert calls["ended"] is True
