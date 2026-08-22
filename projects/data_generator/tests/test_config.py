"""Unit tests for the Config dataclass."""

import pytest
from data_generator.main import Config


class TestConfigFromEnv:
    def test_loads_required_vars(self, monkeypatch):
        monkeypatch.setenv("BQ_PROJECT_ID", "my-project")
        monkeypatch.setenv("BQ_DATASET_ID", "my-dataset")
        monkeypatch.setenv("BQ_TABLE_ID", "my-table")
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        monkeypatch.delenv("ANOMALY_RATE", raising=False)
        monkeypatch.delenv("USER_POOL_SIZE", raising=False)

        config = Config.from_env()

        assert config.project_id == "my-project"
        assert config.dataset_id == "my-dataset"
        assert config.table_id == "my-table"

    def test_defaults_when_optional_vars_absent(self, monkeypatch):
        monkeypatch.setenv("BQ_PROJECT_ID", "p")
        monkeypatch.setenv("BQ_DATASET_ID", "d")
        monkeypatch.setenv("BQ_TABLE_ID", "t")
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        monkeypatch.delenv("ANOMALY_RATE", raising=False)
        monkeypatch.delenv("USER_POOL_SIZE", raising=False)

        config = Config.from_env()

        assert config.batch_size == 2000
        assert config.anomaly_rate == 0.0
        assert config.customer_pool_size == 10000

    def test_reads_optional_vars(self, monkeypatch):
        monkeypatch.setenv("BQ_PROJECT_ID", "p")
        monkeypatch.setenv("BQ_DATASET_ID", "d")
        monkeypatch.setenv("BQ_TABLE_ID", "t")
        monkeypatch.setenv("BATCH_SIZE", "500")
        monkeypatch.setenv("ANOMALY_RATE", "0.1")
        monkeypatch.setenv("USER_POOL_SIZE", "1000")

        config = Config.from_env()

        assert config.batch_size == 500
        assert config.anomaly_rate == 0.1
        assert config.customer_pool_size == 1000

    def test_missing_project_id_raises(self, monkeypatch):
        monkeypatch.delenv("BQ_PROJECT_ID", raising=False)
        monkeypatch.setenv("BQ_DATASET_ID", "d")
        monkeypatch.setenv("BQ_TABLE_ID", "t")

        with pytest.raises(KeyError, match="BQ_PROJECT_ID"):
            Config.from_env()

    def test_missing_dataset_id_raises(self, monkeypatch):
        monkeypatch.setenv("BQ_PROJECT_ID", "p")
        monkeypatch.delenv("BQ_DATASET_ID", raising=False)
        monkeypatch.setenv("BQ_TABLE_ID", "t")

        with pytest.raises(KeyError, match="BQ_DATASET_ID"):
            Config.from_env()

    def test_missing_table_id_raises(self, monkeypatch):
        monkeypatch.setenv("BQ_PROJECT_ID", "p")
        monkeypatch.setenv("BQ_DATASET_ID", "d")
        monkeypatch.delenv("BQ_TABLE_ID", raising=False)

        with pytest.raises(KeyError, match="BQ_TABLE_ID"):
            Config.from_env()


class TestConfigTableRef:
    def test_table_ref_format(self):
        config = Config(project_id="proj", dataset_id="ds", table_id="tbl")
        assert config.table_ref == "proj.ds.tbl"

    def test_table_ref_uses_all_three_parts(self):
        config = Config(project_id="a", dataset_id="b", table_id="c")
        ref = config.table_ref
        assert ref.startswith("a.")
        assert ".b." in ref
        assert ref.endswith(".c")


class TestConfigImmutability:
    def test_config_is_frozen(self):
        config = Config(project_id="p", dataset_id="d", table_id="t")
        with pytest.raises((AttributeError, TypeError)):
            config.project_id = "other"  # type: ignore[misc]

    def test_configs_with_same_values_are_equal(self):
        a = Config(project_id="p", dataset_id="d", table_id="t")
        b = Config(project_id="p", dataset_id="d", table_id="t")
        assert a == b

    def test_configs_with_different_values_are_not_equal(self):
        a = Config(project_id="p", dataset_id="d", table_id="t1")
        b = Config(project_id="p", dataset_id="d", table_id="t2")
        assert a != b
