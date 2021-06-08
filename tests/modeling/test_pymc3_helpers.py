import pandas as pd
import pymc3 as pm
import pytest
from seaborn import load_dataset

from src.modeling import pymc3_helpers as pmhelp


@pytest.fixture(scope="module")
def iris_df() -> pd.DataFrame:
    return load_dataset("iris").head(25)


@pytest.fixture(scope="module")
def mock_model(iris_df: pd.DataFrame) -> pm.Model:
    with pm.Model() as m:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        mu = pm.Deterministic("mu", a + b * iris_df.sepal_length.values)
        sigma = pm.HalfNormal("sigma", 5)
        y = pm.Normal("y", mu, sigma, observed=iris_df.sepal_width.values)  # noqa: F841
    return m


def test_get_random_variables(mock_model: pm.Model):
    rvs = pmhelp.get_random_variable_names(mock_model)
    expected_rvs = set(["a", "b", "sigma_log__"])
    assert expected_rvs == set(rvs)


def test_get_random_variables_without_log(mock_model: pm.Model):
    rvs = pmhelp.get_random_variable_names(mock_model, rm_log=True)
    expected_rvs = set(["a", "b", "sigma"])
    assert expected_rvs == set(rvs)


def test_get_deterministic_variables(mock_model: pm.Model):
    rvs = pmhelp.get_deterministic_variable_names(mock_model)
    expected_rvs = set(["mu"])
    assert expected_rvs == set(rvs)


def test_get_all_variables(mock_model: pm.Model):
    rvs = pmhelp.get_variable_names(mock_model)
    expected_rvs = set(["a", "b", "sigma_log__", "mu", "sigma", "y"])
    assert expected_rvs == set(rvs)


def test_get_all_variables_rm_log(mock_model: pm.Model):
    rvs = pmhelp.get_variable_names(mock_model, rm_log=True)
    expected_rvs = set(["a", "b", "mu", "sigma", "y"])
    assert expected_rvs == set(rvs)
