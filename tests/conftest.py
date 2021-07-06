import os
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import Verbosity, settings

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


#### ---- Hypothesis profiles ---- ####

settings.register_profile("ci", deadline=None, max_examples=1000)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=25)

IS_CI = os.getenv("CI") is not None
settings.register_profile(
    "slow-adaptive",
    max_examples=100 if IS_CI else 5,
    deadline=None if IS_CI else timedelta(minutes=0.5),
)

settings.load_profile("dev")
