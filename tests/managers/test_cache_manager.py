from itertools import product
from pathlib import Path

import arviz as az
import pytest

from speclet.managers import cache_manager as cm
from speclet.managers.cache_manager import PosteriorManager, get_posterior_cache_name
from speclet.project_enums import ModelFitMethod

# ---- PosteriorManager ----


@pytest.fixture
def posterior(centered_eight_idata: az.InferenceData) -> az.InferenceData:
    return centered_eight_idata.copy()


@pytest.fixture
def post_pred(centered_eight_idata: az.InferenceData) -> az.InferenceData:
    return centered_eight_idata.copy()


def test_init_posterior_manager(tmp_path: Path) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    assert not pm.posterior_cache_exists
    assert not pm.posterior_predictive_cache_exists
    assert pm.get_posterior() is None
    assert pm.get_posterior_predictive() is None


def test_caching_posterior(tmp_path: Path, posterior: az.InferenceData) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    assert not pm.posterior_cache_exists
    assert not pm.posterior_predictive_cache_exists
    pm.put_posterior(posterior)
    assert pm.posterior_cache_exists
    assert not pm.posterior_predictive_cache_exists
    assert pm.get_posterior() is posterior
    assert pm.get_posterior_predictive() is None


def test_caching_posterior_predictive(
    tmp_path: Path, post_pred: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    assert not pm.posterior_cache_exists
    assert not pm.posterior_predictive_cache_exists
    pm.put_posterior_predictive(post_pred)
    assert not pm.posterior_cache_exists
    assert pm.posterior_predictive_cache_exists
    assert pm.get_posterior() is None
    assert pm.get_posterior_predictive() is post_pred


def test_clearing_cache_file(
    tmp_path: Path, posterior: az.InferenceData, post_pred: az.InferenceData
) -> None:
    assert posterior is not post_pred
    pm = PosteriorManager("test-post", tmp_path)
    pm.put_posterior(posterior)
    assert pm.posterior_cache_exists
    pm.put_posterior_predictive(post_pred)
    assert pm.posterior_predictive_cache_exists
    pm.clear_cache()
    assert not pm.posterior_cache_exists
    assert not pm.posterior_predictive_cache_exists
    assert pm.get_posterior() is posterior
    assert pm.get_posterior_predictive() is post_pred


def test_clearing_posterior(
    tmp_path: Path, posterior: az.InferenceData, post_pred: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    pm.put_posterior_predictive(post_pred)
    assert pm.posterior_predictive_cache_exists
    pm.put_posterior(posterior)
    assert pm.posterior_cache_exists
    pm.clear()
    assert not pm.posterior_cache_exists
    assert pm.get_posterior() is None
    assert not pm.posterior_predictive_cache_exists
    assert pm.get_posterior_predictive() is None


def test_intentional_write_to_file(
    tmp_path: Path, posterior: az.InferenceData, post_pred: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    pm.put_posterior(posterior)
    assert pm.posterior_cache_exists
    pm.put_posterior_predictive(post_pred)
    assert pm.posterior_predictive_cache_exists
    pm.clear_cache()
    assert not pm.posterior_cache_exists
    assert not pm.posterior_predictive_cache_exists
    assert pm.get_posterior() is posterior
    assert pm.get_posterior_predictive() is post_pred
    pm.write_to_file()
    assert pm.posterior_cache_exists
    assert pm.posterior_predictive_cache_exists


# ---- Separate functions ----


def test_get_posterior_cache_name() -> None:
    possible_names = ["model-name", "cool_model", "such-bayes"]
    posterior_ids: list[str] = []
    for name, fit_method in product(possible_names, ModelFitMethod):
        posterior_ids.append(cm.get_posterior_cache_name(name, fit_method))
    assert len(posterior_ids) == len(set(posterior_ids))


@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_cache_posterior(
    tmp_path: Path, centered_eight_idata: az.InferenceData, fit_method: ModelFitMethod
) -> None:
    dest = cm.cache_posterior(
        centered_eight_idata,
        id=get_posterior_cache_name("test-model", fit_method),
        cache_dir=tmp_path,
    )
    assert dest.exists()
