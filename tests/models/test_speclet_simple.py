from pathlib import Path
from typing import Optional

import arviz as az
import faker
import pandas as pd
import pymc3 as pm
import pytest

from src.managers.data_managers import CrisprScreenDataManager
from src.modeling.simulation_based_calibration_helpers import SBCFileManager
from src.models.speclet_model import SpecletModelDataManager
from src.models.speclet_simple import SpecletSimple
from src.project_enums import MockDataSize, ModelFitMethod, assert_never

fake = faker.Faker()


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize(
    "data_manager",
    [None, CrisprScreenDataManager(Path("tests", "depmap_test_data.csv"))],
)
def test_init(
    tmp_path: Path, debug: bool, data_manager: Optional[SpecletModelDataManager]
) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug, data_manager)
    assert isinstance(sps, SpecletSimple)
    assert sps.model is None
    assert sps.data_manager is not None
    assert sps.debug == debug
    assert sps.cache_manager is not None
    assert sps.mcmc_results is None and sps.advi_results is None


@pytest.mark.parametrize("debug", [True, False])
def test_build_model(tmp_path: Path, debug: bool) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug)
    assert isinstance(sps, SpecletSimple)
    assert sps.model is None
    sps.build_model()
    assert sps.model is not None
    assert isinstance(sps.model, pm.Model)


@pytest.mark.slow
@pytest.mark.parametrize("debug", [True, False])
def test_mcmc_sample_model(tmp_path: Path, debug: bool) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug)
    sps.build_model()
    assert sps.mcmc_results is None and sps.advi_results is None
    sps.mcmc_sample_model(
        draws=11, tune=10, chains=2, cores=1, target_accept=0.8, prior_pred_samples=101
    )
    assert sps.mcmc_results is not None
    assert isinstance(sps.mcmc_results, az.InferenceData)
    for x in (
        "posterior",
        "posterior_predictive",
        "sample_stats",
        "prior",
        "prior_predictive",
    ):
        assert hasattr(sps.mcmc_results, x)

    n_datapoints = sps.data_manager.get_data().shape[0]
    assert sps.mcmc_results.get("posterior")["a"].values.shape == (2, 11)
    assert sps.mcmc_results.get("prior_predictive")["lfc"].values.shape == (
        1,
        101,
        n_datapoints,
    )
    assert sps.mcmc_results.get("posterior_predictive")["lfc"].values.shape == (
        2,
        11,
        n_datapoints,
    )


@pytest.mark.slow
@pytest.mark.parametrize("debug", [True, False])
def test_advi_sample_model(tmp_path: Path, debug: bool) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug)
    sps.build_model()
    assert sps.mcmc_results is None and sps.advi_results is None
    sps.advi_sample_model(n_iterations=21, draws=11, prior_pred_samples=101)
    assert sps.advi_results is not None
    assert isinstance(sps.advi_results[0], az.InferenceData)
    assert isinstance(sps.advi_results[1], pm.Approximation)
    assert len(sps.advi_results[1].hist) == 21
    for x in (
        "posterior",
        "posterior_predictive",
        "prior",
        "prior_predictive",
    ):
        assert hasattr(sps.advi_results[0], x)

    n_datapoints = sps.data_manager.get_data().shape[0]
    assert sps.advi_results[0].get("posterior")["a"].values.shape == (1, 11)
    assert sps.advi_results[0].get("prior_predictive")["lfc"].values.shape == (
        1,
        101,
        n_datapoints,
    )
    assert sps.advi_results[0].get("posterior_predictive")["lfc"].values.shape == (
        1,
        11,
        n_datapoints,
    )


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("data_size", MockDataSize)
def test_generate_mock_data(
    tmp_path: Path, data_size: MockDataSize, debug: bool
) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug)
    mock_data = sps.generate_mock_data(data_size)
    assert "lfc" in mock_data.columns
    assert mock_data.shape[0] > 10


@pytest.mark.slow
@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("data_size", MockDataSize)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_run_simulation_based_calibration(
    tmp_path: Path, data_size: MockDataSize, fit_method: ModelFitMethod, debug: bool
) -> None:
    root_dir = tmp_path / "model-cache"
    root_dir.mkdir()
    sbc_dir = tmp_path / "sbc-cache"
    sbc_dir.mkdir()
    sps = SpecletSimple(fake.name(), root_dir, debug=debug)

    fit_kwargs: dict[str, float] = {"draws": 11, "prior_pred_samples": 101}
    if fit_method is ModelFitMethod.ADVI:
        fit_kwargs["n_iterations"] = 21
    elif fit_method is ModelFitMethod.MCMC:
        fit_kwargs["tune"] = 10
        fit_kwargs["chains"] = 2
        fit_kwargs["cores"] = 1
        fit_kwargs["target_accept"] = 0.8
    else:
        assert_never(fit_method)

    sps.run_simulation_based_calibration(
        results_path=sbc_dir,
        fit_method=fit_method,
        size=data_size,
        fit_kwargs=fit_kwargs,
    )


def make_sbc_subdirs(tmp_path: Path, n: int = 2) -> list[Path]:
    dirs: list[Path] = []
    for i in range(n):
        d = tmp_path / f"dir{i}"
        d.mkdir()
        dirs.append(d)
    return dirs


@pytest.mark.parametrize("debug", (True, False))
def test_get_sbc(
    tmp_path: Path,
    centered_eight: az.InferenceData,
    debug: bool,
) -> None:
    model_dir, sbc_dir = make_sbc_subdirs(tmp_path, n=2)
    sbc_fm = SBCFileManager(sbc_dir)
    sps = SpecletSimple(fake.name(), model_dir, debug=debug)

    sbc_fm.save_sbc_data(sps.generate_mock_data(MockDataSize.SMALL))
    sbc_fm.save_sbc_results(
        priors={}, inference_obj=centered_eight, posterior_summary=pd.DataFrame()
    )

    data, sbc_results, new_sbc_fm = sps.get_sbc(sbc_dir)
    assert data.shape == sps.data_manager.get_data().shape
    assert isinstance(sbc_results.priors, dict)
    assert isinstance(sbc_results.inference_obj, az.InferenceData)
    assert isinstance(sbc_results.posterior_summary, pd.DataFrame)
    assert new_sbc_fm.dir == sbc_dir
    assert sps.model is not None and isinstance(sps.model, pm.Model)


@pytest.mark.parametrize("debug", (True, False))
def test_mcmc_caching(
    tmp_path: Path,
    centered_eight: az.InferenceData,
    debug: bool,
) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug)
    assert not sps.cache_manager.mcmc_cache_exists()
    sps.mcmc_results = centered_eight
    sps.write_mcmc_cache()
    assert sps.cache_manager.mcmc_cache_exists()


@pytest.mark.slow
@pytest.mark.parametrize("debug", (True, False))
def test_advi_caching(
    tmp_path: Path,
    centered_eight: az.InferenceData,
    debug: bool,
) -> None:
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug)
    assert not sps.cache_manager.advi_cache_exists()
    sps.build_model()
    sps.advi_sample_model(n_iterations=21, draws=11, prior_pred_samples=101)
    sps.write_advi_cache()
    assert sps.cache_manager.advi_cache_exists()
