from pathlib import Path

import pytest

from src.managers.model_data_managers import CrcDataManager


@pytest.fixture
def mock_model_config() -> Path:
    return Path("tests/models/mock-model-config.yaml")


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


@pytest.fixture(scope="function")
def mock_crc_dm(monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
    monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
    dm = CrcDataManager(debug=True)
    return dm
