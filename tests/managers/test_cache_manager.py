from itertools import product
from pathlib import Path

import arviz as az
import pytest

from speclet.managers import cache_manager as cm
from speclet.managers.cache_manager import PosteriorManager
from speclet.project_enums import ModelFitMethod

# ---- PosteriorManager ----


def test_init_posterior_manager(tmp_path: Path) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    assert not pm.cache_exists
    assert pm.get() is None


def test_caching_posterior(
    tmp_path: Path, centered_eight_idata: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    assert not pm.cache_exists
    pm.put(centered_eight_idata)
    assert pm.cache_exists
    assert pm.get() is centered_eight_idata


def test_clearing_cache_file(
    tmp_path: Path, centered_eight_idata: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    pm.put(centered_eight_idata)
    assert pm.cache_exists
    pm.clear_cache()
    assert not pm.cache_exists
    assert pm.get() is centered_eight_idata


def test_clearing_posterior(
    tmp_path: Path, centered_eight_idata: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    pm.put(centered_eight_idata)
    assert pm.cache_exists
    pm.clear()
    assert not pm.cache_exists
    assert pm.get() is None


def test_intentional_write_to_file(
    tmp_path: Path, centered_eight_idata: az.InferenceData
) -> None:
    pm = PosteriorManager("test-post", tmp_path)
    pm.put(centered_eight_idata)
    assert pm.cache_exists
    pm.clear_cache()
    assert not pm.cache_exists
    assert pm.get() is centered_eight_idata
    pm.write_to_file()
    assert pm.cache_exists


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
        name="test-posterior",
        fit_method=fit_method,
        cache_dir=tmp_path,
    )
    assert dest.exists()
