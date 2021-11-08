from pathlib import Path

import numpy as np
import pytest

from src.models.speclet_nine import (
    SpecletNine,
    SpecletNineConfiguration,
    SpecletNinePriors,
)
from src.project_enums import ModelFitMethod, ModelParameterization, assert_never


@pytest.fixture(scope="function")
def sp9(tmp_path: Path) -> SpecletNine:
    return SpecletNine("test-model", root_cache_dir=tmp_path)


def test_init(tmp_path: Path) -> None:
    sp9 = SpecletNine("test-model", root_cache_dir=tmp_path)
    assert sp9.model is None
    assert sp9.data_manager is not None


def test_get_data(sp9: SpecletNine) -> None:
    data = sp9.data_manager.get_data()
    for c in ["counts_initial_adj", "counts_final"]:
        assert c in data.columns


def test_build_model(sp9: SpecletNine) -> None:
    assert sp9.model is None
    sp9.build_model()
    assert sp9.model is not None
    assert sp9.observed_var_name is not None


@pytest.mark.parametrize(
    "config",
    [
        SpecletNineConfiguration(),
        SpecletNineConfiguration(
            beta_parameterization=ModelParameterization.NONCENTERED
        ),
        SpecletNineConfiguration(
            priors=SpecletNinePriors(**{"mu_beta": {"mu": 10, "sigma": 1.2}})
        ),
    ],
)
def test_set_config(sp9: SpecletNine, config: SpecletNineConfiguration) -> None:
    sp9.set_config(config.dict())
    assert sp9._config.dict() == config.dict()


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        SpecletNineConfiguration(),
        SpecletNineConfiguration(
            beta_parameterization=ModelParameterization.NONCENTERED
        ),
    ],
)
@pytest.mark.parametrize("method", ModelFitMethod)
def test_sample_model(
    sp9: SpecletNine, config: SpecletNineConfiguration, method: ModelFitMethod
) -> None:
    sp9.set_config(config.dict())
    data = sp9.data_manager.get_data()
    ct_i = np.abs(np.random.normal(loc=10, scale=2, size=data.shape[0]))
    ct_f = np.abs(ct_i + np.random.normal(loc=0, scale=5, size=data.shape[0]))
    data["counts_initial_adj"] = ct_i.astype(np.int32)
    data["counts_final"] = ct_f.astype(np.int32)
    sp9.data_manager.set_data(data, apply_transformations=False)

    sp9.build_model()
    assert sp9.mcmc_results is None
    assert sp9.advi_results is None

    if method is ModelFitMethod.MCMC:
        _ = sp9.mcmc_sample_model(
            prior_pred_samples=100, sample_kwargs={"draws": 10, "tune": 10, "chains": 1}
        )
        assert sp9.mcmc_results is not None
        assert sp9.advi_results is None
    elif method is ModelFitMethod.ADVI:
        _ = sp9.advi_sample_model(prior_pred_samples=100, n_iterations=100, draws=23)
        assert sp9.mcmc_results is None
        assert sp9.advi_results is not None
    else:
        assert_never(method)
