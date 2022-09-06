from math import ceil

import arviz as az
import pandas as pd
import pymc as pm
import pytest
from seaborn import load_dataset

from speclet.modeling import pymc_helpers as pmhelp


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
    expected_rvs = {"a", "b", "sigma", "y"}
    assert expected_rvs == set(rvs)


def test_get_deterministic_variables(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_deterministic_variable_names(mock_model)
    expected_rvs = {"mu"}
    assert expected_rvs == set(rvs)


def test_get_all_variables(mock_model: pm.Model) -> None:
    rvs = pmhelp.get_variable_names(mock_model)
    expected_rvs = {"a", "b", "mu", "sigma", "y"}
    assert expected_rvs == set(rvs)


def test_get_posterior_names(centered_eight_idata: az.InferenceData) -> None:
    got_names = pmhelp.get_posterior_names(centered_eight_idata)
    expected_names = {"mu", "theta", "tau"}
    assert set(got_names) == expected_names


@pytest.mark.DEV
@pytest.mark.parametrize("by", [1, 2, 4, 23, 61, 100000])
def test_thin(
    centered_eight_idata: az.InferenceData,
    by: int,
) -> None:
    n_draws_post = centered_eight_idata.posterior.dims["draw"]
    thinned_size = ceil(n_draws_post / by)
    thinned = pmhelp.thin(centered_eight_idata, by=by)
    assert thinned.posterior is not None
    assert thinned.posterior.dims["draw"] == thinned_size
    assert thinned.posterior_predictive.dims["draw"] == thinned_size


@pytest.mark.parametrize("by", [1, 2, 4, 23, 61, 100000])
def test_thin_posterior(
    centered_eight_idata: az.InferenceData,
    by: int,
) -> None:
    n_draws_post = centered_eight_idata.posterior.dims["draw"]
    n_draws_post_pred = centered_eight_idata.posterior_predictive.dims["draw"]
    thinned_size = ceil(n_draws_post / by)
    thinned = pmhelp.thin_posterior(centered_eight_idata, by=by)
    assert thinned is centered_eight_idata
    assert thinned.posterior is not None
    assert thinned.posterior.dims["draw"] == thinned_size
    assert thinned.posterior_predictive.dims["draw"] == n_draws_post_pred


@pytest.mark.parametrize("by", [1, 2, 4, 23, 61, 100000])
def test_thin_posterior_predictive(
    centered_eight_idata: az.InferenceData,
    by: int,
) -> None:
    n_draws_post = centered_eight_idata.posterior.dims["draw"]
    n_draws_post_pred = centered_eight_idata.posterior_predictive.dims["draw"]
    thinned_size = ceil(n_draws_post_pred / by)
    thinned = pmhelp.thin_posterior_predictive(centered_eight_idata, by=by)
    assert thinned is centered_eight_idata
    assert thinned.posterior is not None
    assert thinned.posterior.dims["draw"] == n_draws_post
    assert thinned.posterior_predictive.dims["draw"] == thinned_size


@pytest.mark.parametrize("chain_num", list(range(4)))
@pytest.mark.parametrize(
    "var_name, new_shape",
    [("mu", (500,)), ("tau", (500,)), ("theta", (500, 8))],
)
def test_get_one_chain(
    centered_eight_idata: az.InferenceData,
    var_name: str,
    new_shape: tuple[int, ...],
    chain_num: int,
) -> None:
    post = centered_eight_idata["posterior"][var_name]
    new_post = pmhelp.get_one_chain(post, chain_num=chain_num)
    assert new_post.shape == new_shape


# @pytest.mark.parametrize("centered", (True, False))
# def test_hierarchical_normal(centered: bool) -> None:
#     with pm.Model() as m:
#         a = pmhelp.hierarchical_normal("var-name", shape=(2, 5), centered=centered)

#     assert a.name == "var-name"
#     assert a.ndim == 2

#     if centered:
#         assert a.dshape == (2, 5)
#         assert isinstance(a, RandomVariable)
#         with pytest.raises(KeyError):
#             _ = m["Δ_var-name"]
#     else:
#         assert m["Δ_var-name"].dshape == (2, 5)
#         assert isinstance(a, Deterministic)


# @pytest.mark.parametrize("centered", (True, False))
# def test_hierarchical_normal_with_avg(centered: bool) -> None:

#     avgs = at.constant([1, 2, 3, 4, 5])
#     shape = (5,)

#     with pm.Model() as m:
#         a = pmhelp.hierarchical_normal_with_avg(
#             "var-name", avg_map={"other-var": avgs}, shape=shape, centered=centered
#         )

#     assert a.name == "var-name"
#     assert a.ndim == 1

#     gamma = m["γ_var-name_other-var_bar"]
#     assert isinstance(gamma, RandomVariable)

#     if centered:
#         assert a.dshape == shape
#         assert isinstance(a, RandomVariable)
#         with pytest.raises(KeyError):
#             _ = m["Δ_var-name"]
#     else:
#         assert m["Δ_var-name"].dshape == shape
#         assert isinstance(a, Deterministic)
