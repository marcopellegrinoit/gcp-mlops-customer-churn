"""Unit tests for the generator's environment configuration."""

import pytest
from data_generator.config import Settings, get_settings
from pydantic import ValidationError


@pytest.fixture()
def destination(monkeypatch):
    """Only the required BigQuery destination set; every simulation knob left to default."""
    monkeypatch.setenv("BQ_PROJECT_ID", "my-project")
    monkeypatch.setenv("BQ_DATASET_ID", "my-dataset")
    monkeypatch.setenv("BQ_TABLE_ID", "my-table")
    for name in ("BATCH_SIZE", "ANOMALY_RATE", "USER_POOL_SIZE", "DAILY_ACQUISITIONS"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


class TestFromEnvironment:
    def test_loads_the_bigquery_destination(self, destination):
        settings = get_settings()
        assert settings.project_id == "my-project"
        assert settings.table_ref == "my-project.my-dataset.my-table"

    def test_simulation_knobs_default(self, destination):
        settings = get_settings()
        assert settings.batch_size == 2000
        assert settings.anomaly_rate == 0.0
        assert settings.customer_pool_size == 10000
        assert settings.daily_acquisitions == 25

    def test_simulation_knobs_are_read_from_the_environment(self, destination):
        destination.setenv("BATCH_SIZE", "500")
        destination.setenv("ANOMALY_RATE", "0.1")
        destination.setenv("USER_POOL_SIZE", "1000")
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.batch_size == 500
        assert settings.anomaly_rate == 0.1
        assert settings.customer_pool_size == 1000

    @pytest.mark.parametrize("missing", ["BQ_PROJECT_ID", "BQ_DATASET_ID", "BQ_TABLE_ID"])
    def test_missing_destination_variable_fails_at_startup(self, destination, missing):
        destination.delenv(missing)
        get_settings.cache_clear()
        with pytest.raises(ValidationError, match=missing.removeprefix("BQ_").lower()):
            get_settings()


class TestValidation:
    def test_anomaly_rate_above_one_is_rejected(self):
        # It is a probability per event; anything above 1 silently means "every event".
        with pytest.raises(ValidationError):
            Settings(project_id="p", dataset_id="d", table_id="t", anomaly_rate=1.5)

    def test_zero_batch_size_is_rejected(self):
        # A run that emits nothing looks identical to a healthy quiet night downstream.
        with pytest.raises(ValidationError):
            Settings(project_id="p", dataset_id="d", table_id="t", batch_size=0)


class TestTableRef:
    def test_table_ref_uses_all_three_parts(self):
        assert Settings(project_id="a", dataset_id="b", table_id="c").table_ref == "a.b.c"

    def test_settings_are_frozen(self):
        settings = Settings(project_id="p", dataset_id="d", table_id="t")
        with pytest.raises(ValidationError):
            settings.project_id = "other"

    def test_settings_with_same_values_are_equal(self):
        a = Settings(project_id="p", dataset_id="d", table_id="t")
        b = Settings(project_id="p", dataset_id="d", table_id="t")
        assert a == b
