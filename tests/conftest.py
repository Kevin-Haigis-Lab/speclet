import os
from datetime import timedelta
from pathlib import Path

import arviz as az
import pandas as pd
import pytest
import seaborn as sns
from hypothesis import HealthCheck, Verbosity, settings

from src.managers.model_data_managers import CrcDataManager


@pytest.fixture
def mock_model_config() -> Path:
    return Path("tests/models/mock-model-config.yaml")


@pytest.fixture
def depmap_test_data() -> Path:
    return Path("tests", "depmap_test_data.csv")


def monkey_get_data_path(*args, **kwargs) -> Path:
    return Path("tests", "depmap_test_data.csv")


@pytest.fixture(scope="function")
def mock_crc_dm(monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
    monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
    dm = CrcDataManager(debug=True)
    return dm


@pytest.fixture(scope="function")
def mock_crc_dm_multiscreen(monkeypatch: pytest.MonkeyPatch) -> CrcDataManager:
    monkeypatch.setattr(CrcDataManager, "get_data_path", monkey_get_data_path)
    dm = CrcDataManager(debug=True, broad_only=False)
    return dm


#### ---- Data frames ---- ####


@pytest.fixture
def iris() -> pd.DataFrame:
    return sns.load_dataset("iris")


#### ---- PyMC3 fixtures ---- ####


@pytest.fixture
def centered_eight() -> az.InferenceData:
    x = az.load_arviz_data("centered_eight")
    assert isinstance(x, az.InferenceData)
    return x


@pytest.fixture
def centered_eight_post(centered_eight: az.InferenceData) -> pd.DataFrame:
    x = az.summary(centered_eight)
    assert isinstance(x, pd.DataFrame)
    return x


#### ---- Hypothesis profiles ---- ####

settings.register_profile(
    "ci", deadline=None, max_examples=1000, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=5)

IS_CI = os.getenv("CI") is not None
settings.register_profile(
    "slow-adaptive",
    parent=settings.get_profile("ci") if IS_CI else settings.get_profile("default"),
    max_examples=100 if IS_CI else 5,
    deadline=None if IS_CI else timedelta(minutes=0.5),
)

settings.load_profile("dev")
