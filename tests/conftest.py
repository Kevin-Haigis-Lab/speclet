import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Final

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest
import seaborn as sns
from hypothesis import HealthCheck, Verbosity, settings

TEST_DATA: Final[Path] = Path("tests", "depmap_test_data.csv")


# ---- Callable fixtures ----
# These will tend to be function factories for use as monkeypatching  methods.


@pytest.fixture
def return_true() -> Callable:
    """Get a function that can take any args and always returns `True`"""

    def f(*args: Any, **kwargs: Any) -> bool:
        return True

    return f


# ---- Standard fixtures ----


@pytest.fixture
def mock_model_config() -> Path:
    return Path("tests", "models", "mock-model-config.yaml")


@pytest.fixture
def depmap_test_data() -> Path:
    return TEST_DATA


def monkey_get_data_path(*args: Any, **kwargs: Any) -> Path:
    return TEST_DATA


# ---- Data frames ----


@pytest.fixture
def iris() -> pd.DataFrame:
    iris_df = sns.load_dataset("iris")
    assert isinstance(iris_df, pd.DataFrame)
    return iris_df


# ---- PyMC3 fixtures ----


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


@pytest.fixture(scope="module")
def simple_model() -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=[1, 2, 3])  # noqa: F841
    return model


@pytest.fixture(scope="module")
def hierarchical_model() -> pm.Model:
    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", 0, 1)
        sigma_alpha = pm.HalfCauchy("sigma_alpha", 1)
        alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, shape=2)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal(  # noqa: F841
            "y", alpha[np.array([0, 0, 1, 1])], sigma, observed=[1, 2, 3, 4]
        )
    return model


# ---- Hypothesis profiles ----

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
