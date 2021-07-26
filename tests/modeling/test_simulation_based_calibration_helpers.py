from pathlib import Path
from string import ascii_letters
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest
import seaborn as sns

from src.exceptions import IncorrectNumberOfFilesFoundError
from src.modeling import simulation_based_calibration_helpers as sbc

chars = list(ascii_letters) + [str(i) for i in (range(10))]


class TestSBCFileManager:
    def test_init(self, tmp_path: Path):
        fm = sbc.SBCFileManager(dir=tmp_path)
        assert not fm.all_data_exists()

    @pytest.fixture()
    def priors(self) -> dict[str, Any]:
        return {
            "alpha": np.random.uniform(0, 100, size=3),
            "beta_log": np.random.uniform(0, 100, size=(10, 15)),
        }

    @pytest.fixture
    def posterior_summary(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [5, 6, 7], "y": ["a", "b", "c"]})

    @pytest.fixture
    def iris(self) -> pd.DataFrame:
        return sns.load_dataset("iris")

    def test_saving(
        self, tmp_path: Path, priors: dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)
        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()

    def test_reading(
        self, tmp_path: Path, priors: dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)

        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()
        read_results = fm.get_sbc_results()
        assert isinstance(read_results, sbc.SBCResults)
        assert isinstance(read_results.inference_obj, az.InferenceData)
        for k in read_results.priors:
            np.testing.assert_array_equal(read_results.priors[k], priors[k])

        for c in read_results.posterior_summary.columns:
            np.testing.assert_array_equal(
                read_results.posterior_summary[c].values, posterior_summary[c].values
            )

    def test_saving_simulation_dataframe(self, tmp_path: Path, iris: pd.DataFrame):
        fm = sbc.SBCFileManager(tmp_path)
        fm.save_sbc_data(iris)
        assert fm.get_sbc_data() is iris
        fm.sbc_data = None
        assert fm.get_sbc_data() is not iris
        assert fm.get_sbc_data().shape == iris.shape

    def test_clearing_saved_simulation_dataframe(
        self, tmp_path: Path, iris: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(tmp_path)
        fm.save_sbc_data(iris)
        assert fm.sbc_data_path.exists()
        fm.clear_results()
        assert fm.sbc_data_path.exists()
        fm.clear_saved_data()
        assert not fm.sbc_data_path.exists()


#### ---- Test SBC collation ---- ####


# Fixtures and helpers


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


# Tests


def test_is_true_value_within_hdi_lower_limit():
    n = 100
    low = pd.Series(list(range(0, n)))
    high = pd.Series([200] * n)
    vals = pd.Series([50] * n)
    is_within = sbc._is_true_value_within_hdi(low, vals, high)
    assert np.all(is_within[:50])
    assert not np.any(is_within[50:])


def test_is_true_value_within_hdi_upper_limit():
    n = 100
    low = pd.Series([0] * n)
    high = pd.Series(list(range(100)))
    vals = pd.Series([50] * n)
    is_within = sbc._is_true_value_within_hdi(low, vals, high)
    assert not np.any(is_within[:51])
    assert np.all(is_within[51:])


def test_get_prior_value_using_index_list_mismatch_index_size():
    a = np.array([4, 3, 2, 1])
    idx: list[int] = []
    with pytest.raises(AssertionError):
        _ = sbc._get_prior_value_using_index_list(a, idx)


def test_get_prior_value_using_index_list_empty_idx():
    a = np.array(4)
    idx: list[int] = []
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 4.0


def test_get_prior_value_using_index_list_empty_idx_but_not_flat_array():
    a = np.array([4])
    idx: list[int] = []
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 4.0


def test_get_prior_value_using_index_list_1d():
    a = np.array([4, 3, 2, 1])
    idx = [0]
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 4
    idx = [1]
    b = sbc._get_prior_value_using_index_list(a, idx)
    assert b == 3


def test_get_prior_value_using_index_list_2d():
    a = np.arange(9).reshape((3, 3))
    idx = [1, 2]
    b = sbc._get_prior_value_using_index_list(a, idx)
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
    assert res == sbc._split_parameter(p)


def test_error_when_incorrect_number_of_results_found(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sbc, "get_posterior_summary_for_file_manager", return_iris)

    n_paths = 20
    fake_paths = [Path(f"fake-path-{i}") for i in range(n_paths)]

    with pytest.raises(IncorrectNumberOfFilesFoundError) as err1:
        sbc.collate_sbc_posteriors(fake_paths, num_permutations=n_paths - 1)

    assert err1.value.expected == n_paths - 1
    assert err1.value.found == n_paths

    with pytest.raises(IncorrectNumberOfFilesFoundError) as err2:
        sbc.collate_sbc_posteriors(fake_paths, num_permutations=n_paths + 1)

    assert err2.value.expected == n_paths + 1
    assert err2.value.found == n_paths


@pytest.mark.slow
def test_make_priors_dataframe_simple(simple_model: pm.Model):
    with simple_model:
        priors = pm.sample_prior_predictive(samples=1)

    parameters = ["mu", "sigma"]
    prior_df = sbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


@pytest.mark.slow
def test_make_priors_dataframe_hierarchical(hierarchical_model: pm.Model):
    with hierarchical_model:
        priors = pm.sample_prior_predictive(samples=1)

    parameters = ["mu_alpha", "sigma_alpha", "alpha[0]", "alpha[1]", "sigma"]
    prior_df = sbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


@pytest.mark.slow
def test_make_priors_dataframe_hierarchical_with_post(hierarchical_model: pm.Model):
    with hierarchical_model:
        priors = pm.sample_prior_predictive(samples=1)
        trace = pm.sample(10, tune=10, cores=1, chains=2, return_inferencedata=True)

    parameters: list[str] = az.summary(trace).index.tolist()
    prior_df = sbc._make_priors_dataframe(priors, parameters=parameters)
    assert isinstance(prior_df, pd.DataFrame)
    assert set(parameters) == set(prior_df.index.tolist())


def test_failure_if_data_does_not_exist(tmp_path: Path):
    with pytest.raises(sbc.SBCResultsNotFoundError):
        sbc.get_posterior_summary_for_file_manager(tmp_path)
