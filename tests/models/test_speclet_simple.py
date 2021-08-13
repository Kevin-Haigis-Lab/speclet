from pathlib import Path
from typing import Optional

import arviz as az
import faker
import pymc3 as pm
import pytest

from src.managers import model_data_managers as dms
from src.models.speclet_simple import SpecletSimple

fake = faker.Faker()


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize(
    "data_manager", [None, dms.CrcDataManager(), dms.MockDataManager()]
)
def test_init(tmp_path: Path, debug: bool, data_manager: Optional[dms.DataManager]):
    sps = SpecletSimple(fake.name(), tmp_path, debug, data_manager)
    assert isinstance(sps, SpecletSimple)
    assert sps.model is None
    assert sps.data_manager is not None
    assert sps.debug == debug
    assert sps.cache_manager is not None
    assert sps.mcmc_results is None and sps.advi_results is None


@pytest.mark.parametrize("debug", [True, False])
def test_build_model(tmp_path: Path, debug: bool, mock_crc_dm: dms.CrcDataManager):
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug, data_manager=mock_crc_dm)
    assert isinstance(sps, SpecletSimple)
    assert sps.model is None
    sps.build_model()
    assert sps.model is not None
    assert isinstance(sps.model, pm.Model)


@pytest.mark.slow
@pytest.mark.parametrize("debug", [True, False])
def test_mcmc_sample_model(
    tmp_path: Path, debug: bool, mock_crc_dm: dms.CrcDataManager
):
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug, data_manager=mock_crc_dm)
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


@pytest.mark.DEV
@pytest.mark.slow
@pytest.mark.parametrize("debug", [True, False])
def test_advi_sample_model(
    tmp_path: Path, debug: bool, mock_crc_dm: dms.CrcDataManager
):
    sps = SpecletSimple(fake.name(), tmp_path, debug=debug, data_manager=mock_crc_dm)
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
