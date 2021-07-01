from pathlib import Path
from typing import List

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest
import seaborn as sns

from src.exceptions import IncorrectNumberOfFilesFoundError
from src.pipelines import collate_sbc as csbc

#### ---- Fixtures and helpers ---- ####


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


def return_iris(*args, **kwargs) -> pd.DataFrame:
    return sns.load_dataset("iris")


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


#### ---- Tests ---- ####


def test_is_true_value_within_hdi_lower_limit():
    n = 100
    low = pd.Series(list(range(0, n)))
    high = pd.Series([200] * n)
    vals = pd.Series([50] * n)
    is_within = csbc._is_true_value_within_hdi(low, vals, high)
    assert np.all(is_within[:50])
    assert not np.any(is_within[50:])


def test_is_true_value_within_hdi_upper_limit():
    n = 100
    low = pd.Series([0] * n)
    high = pd.Series(list(range(100)))
    vals = pd.Series([50] * n)
    is_within = csbc._is_true_value_within_hdi(low, vals, high)
    assert not np.any(is_within[:51])
    assert np.all(is_within[51:])


def test_get_prior_value_using_index_list_mismatch_index_size():
    a = np.array([4, 3, 2, 1])
    idx: List[int] = []
    with pytest.raises(AssertionError):
        _ = csbc._get_prior_value_using_index_list(a, idx)


def test_get_prior_value_using_index_list_empty_idx():
    a = np.array(4)
    idx: List[int] = []
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 4.0


def test_get_prior_value_using_index_list_empty_idx_but_not_flat_array():
    a = np.array([4])
    idx: List[int] = []
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 4.0


def test_get_prior_value_using_index_list_1d():
    a = np.array([4, 3, 2, 1])
    idx = [0]
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 4
    idx = [1]
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == 3


def test_get_prior_value_using_index_list_2d():
    a = np.arange(9).reshape((3, 3))
    idx = [1, 2]
    b = csbc._get_prior_value_using_index_list(a, idx)
    assert b == a[1, 2]


@pytest.mark.parametrize(
    "p, res",
    [
        ("a", ["a"]),
        ("abc", ["abc"]),
        ("abc[0]", ["abc", "0"]),
        ("abc[0,2,5]", ["abc", "0", "2", "5"]),
        ("abc[ x, y, z]", ["abc", " x", " y", " z"]),
        ("abc[x,y,z]", ["abc", "x", "y", "z"]),
    ],
)
def test_split_parameter(p: str, res: str):
    assert res == csbc._split_parameter(p)


def test_error_when_incorrect_number_of_results_found(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(csbc, "get_posterior_summary_for_file_manager", return_iris)

    n_paths = 20
    fake_paths = [Path(f"fake-path-{i}") for i in range(n_paths)]

    with pytest.raises(IncorrectNumberOfFilesFoundError) as err1:
        csbc.collate_sbc_posteriors(fake_paths, num_permutations=n_paths - 1)

    assert err1.value.expected == n_paths - 1
    assert err1.value.found == n_paths

    with pytest.raises(IncorrectNumberOfFilesFoundError) as err2:
        csbc.collate_sbc_posteriors(fake_paths, num_permutations=n_paths + 1)

    assert err2.value.expected == n_paths + 1
    assert err2.value.found == n_paths


@pytest.mark.slow
def test_make_priors_dataframe_simple(simple_model: pm.Model):
    with simple_model:
        priors = pm.sample_prior_predictive(samples=1)

    parameters = ["mu", "sigma"]
    prior_df = csbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


@pytest.mark.slow
def test_make_priors_dataframe_hierarchical(hierarchical_model: pm.Model):
    with hierarchical_model:
        priors = pm.sample_prior_predictive(samples=1)

    parameters = ["mu_alpha", "sigma_alpha", "alpha[0]", "alpha[1]", "sigma"]
    prior_df = csbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


@pytest.mark.slow
def test_make_priors_dataframe_hierarchical_with_post(hierarchical_model: pm.Model):
    with hierarchical_model:
        priors = pm.sample_prior_predictive(samples=1)
        trace = pm.sample(10, tune=10, cores=2, chains=2, return_inferencedata=True)

    parameters: List[str] = az.summary(trace).index.tolist()
    prior_df = csbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


def test_failure_if_data_does_not_exist(tmp_path: Path):
    with pytest.raises(csbc.SBCResultsNotFoundError):
        csbc.get_posterior_summary_for_file_manager(tmp_path)
