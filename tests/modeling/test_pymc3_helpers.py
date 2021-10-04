import arviz as az
import pandas as pd
import pymc3 as pm
import pytest
from seaborn import load_dataset
from theano import tensor as tt

import src.exceptions
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


def test_get_random_variables(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_random_variable_names(mock_model)
    expected_rvs = {"a", "b", "sigma_log__"}
    assert expected_rvs == set(rvs)


def test_get_random_variables_without_log(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_random_variable_names(mock_model, rm_log=True)
    expected_rvs = {"a", "b", "sigma"}
    assert expected_rvs == set(rvs)


def test_get_deterministic_variables(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_deterministic_variable_names(mock_model)
    expected_rvs = {"mu"}
    assert expected_rvs == set(rvs)


def test_get_all_variables(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_variable_names(mock_model)
    expected_rvs = {"a", "b", "sigma_log__", "mu", "sigma", "y"}
    assert expected_rvs == set(rvs)


def test_get_all_variables_rm_log(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_variable_names(mock_model, rm_log=True)
    expected_rvs = {"a", "b", "mu", "sigma", "y"}
    assert expected_rvs == set(rvs)


def test_get_posterior_names(centered_eight: az.InferenceData) -> None:
    got_names = pmhelp.get_posterior_names(centered_eight)
    expected_names = {"mu", "theta", "tau"}
    assert set(got_names) == expected_names


@pytest.mark.parametrize(
    "var_name, thinned_shape",
    [("mu", (4, 100)), ("tau", (4, 100)), ("theta", (4, 100, 8))],
)
def test_thin_posterior(
    centered_eight: az.InferenceData, var_name: str, thinned_shape: tuple[int, ...]
) -> None:
    post = centered_eight["posterior"][var_name]
    thinned_post = pmhelp.thin_posterior(post, thin_to=100)
    assert thinned_post.shape == thinned_shape
    thinned_post = pmhelp.thin_posterior(post, step_size=5)
    assert thinned_post.shape == thinned_shape
    with pytest.raises(src.exceptions.RequiredArgumentError):
        pmhelp.thin_posterior(post)


@pytest.mark.parametrize("chain_num", list(range(4)))
@pytest.mark.parametrize(
    "var_name, new_shape",
    [("mu", (500,)), ("tau", (500,)), ("theta", (500, 8))],
)
def test_get_one_chain(
    centered_eight: az.InferenceData,
    var_name: str,
    new_shape: tuple[int, ...],
    chain_num: int,
) -> None:
    post = centered_eight["posterior"][var_name]
    new_post = pmhelp.get_one_chain(post, chain_num=chain_num)
    assert new_post.shape == new_shape


@pytest.mark.parametrize("centered", (True, False))
def test_hierarchical_normal(centered: bool) -> None:
    with pm.Model() as m:
        a = pmhelp.hierarchical_normal("var-name", shape=(2, 5), centered=centered)

    assert a.name == "var-name"
    assert a.ndim == 2

    if centered:
        assert a.dshape == (2, 5)
        assert isinstance(a, pm.model.FreeRV)
        with pytest.raises(KeyError):
            _ = m["Δ_var-name"]
    else:
        assert m["Δ_var-name"].dshape == (2, 5)
        assert isinstance(a, pm.model.DeterministicWrapper)


@pytest.mark.DEV
@pytest.mark.parametrize("centered", (True, False))
def test_hierarchical_normal_with_avg(centered: bool) -> None:

    avgs = tt.constant([1, 2, 3, 4, 5])
    shape = (5,)

    with pm.Model() as m:
        a = pmhelp.hierarchical_normal_with_avg(
            "var-name", avg_map={"other-var": avgs}, shape=shape, centered=centered
        )

    assert a.name == "var-name"
    assert a.ndim == 1

    gamma = m["γ_var-name_other-var_bar"]
    assert isinstance(gamma, pm.model.FreeRV)

    if centered:
        assert a.dshape == shape
        assert isinstance(a, pm.model.FreeRV)
        with pytest.raises(KeyError):
            _ = m["Δ_var-name"]
    else:
        assert m["Δ_var-name"].dshape == shape
        assert isinstance(a, pm.model.DeterministicWrapper)
