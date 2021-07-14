from pathlib import Path
from time import time
from typing import Any, Dict, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from src.exceptions import CacheDoesNotExistError
from src.managers.model_data_managers import CrcDataManager, MockDataManager
from src.misc.test_helpers import do_nothing
from src.modeling import simulation_based_calibration_helpers as sbc
from src.models import speclet_model
from src.project_enums import MockDataSize, ModelFitMethod


class MockSpecletModelClass(speclet_model.SpecletModel):
    def __init__(self, name: str, root_cache_dir: Path, debug: bool) -> None:
        _data_manager = MockDataManager()
        super().__init__(
            name, _data_manager, root_cache_dir=root_cache_dir, debug=debug
        )

    def model_specification(self) -> Tuple[pm.Model, str]:
        data = self.data_manager.get_data()
        with pm.Model() as model:
            b = pm.Normal("b", 0, 10)
            a = pm.Normal("a", 0, 10)
            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal(  # noqa: F841
                "y", a + b * data["x"].values, sigma, observed=data["y"].values
            )
        return model, "y"


def assert_posterior_shape(res: az.InferenceData, shape: Tuple[int, int]) -> None:
    for p in ["a", "b", "sigma"]:
        assert res.posterior[p].shape == shape


class TestSpecletModel:
    @pytest.fixture(scope="function")
    def mock_sp_model(self, tmp_path: Path) -> MockSpecletModelClass:
        return MockSpecletModelClass(
            name="test-model",
            root_cache_dir=tmp_path,
            debug=False,
        )

    def test_mcmc_sample_model_fails_without_overriding(self, tmp_path: Path):
        sp = speclet_model.SpecletModel(
            "test-model", data_manager=MockDataManager(), root_cache_dir=tmp_path
        )
        with pytest.raises(AttributeError, match="Cannot sample: model is 'None'"):
            sp.mcmc_sample_model()

    @pytest.mark.slow
    def test_build_model(self, mock_sp_model: MockSpecletModelClass):
        assert mock_sp_model.model is None
        mock_sp_model.build_model()
        assert mock_sp_model.model is not None
        assert isinstance(mock_sp_model.model, pm.Model)

    @pytest.mark.slow
    def test_mcmc_sample_model(self, mock_sp_model: MockSpecletModelClass):
        mock_sp_model.build_model()
        mcmc_res = mock_sp_model.mcmc_sample_model(
            draws=50,
            tune=50,
            chains=2,
            cores=2,
            prior_pred_samples=25,
            post_pred_samples=50,
            random_seed=1,
        )
        assert isinstance(mcmc_res, az.InferenceData)
        assert_posterior_shape(mcmc_res, (2, 50))

        assert mock_sp_model.mcmc_results is not None

        tic = time()
        mcmc_res_2 = mock_sp_model.mcmc_sample_model(
            draws=50,
            tune=50,
            chains=2,
            cores=2,
            prior_pred_samples=25,
            post_pred_samples=50,
            random_seed=1,
        )
        toc = time()

        assert mcmc_res is mcmc_res_2
        assert toc - tic < 1
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(
                mcmc_res.posterior[p], mcmc_res_2.posterior[p]
            )

    def test_advi_sample_model_fails_without_model(self, tmp_path: Path):
        sp = speclet_model.SpecletModel(
            "test-model", data_manager=MockDataManager(), root_cache_dir=tmp_path
        )
        with pytest.raises(AttributeError, match="model is 'None'"):
            sp.advi_sample_model()

    @pytest.mark.slow
    def test_advi_sample_model(self, mock_sp_model: MockSpecletModelClass):
        mock_sp_model.build_model()
        advi_res, advi_approx = mock_sp_model.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=25,
            post_pred_samples=50,
            random_seed=1,
        )
        assert isinstance(advi_res, az.InferenceData)
        assert_posterior_shape(advi_res, (1, 100))

        assert mock_sp_model.advi_results is not None

        tic = time()
        advi_res_2, advi_approx_2 = mock_sp_model.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=25,
            post_pred_samples=50,
            random_seed=1,
        )
        toc = time()

        assert advi_res is advi_res_2
        assert advi_approx is advi_approx_2
        assert toc - tic < 1
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(
                advi_res.posterior[p], advi_res_2.posterior[p]
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("fit_method", [ModelFitMethod.ADVI, ModelFitMethod.MCMC])
    def test_run_simulation_based_calibration(
        self,
        tmp_path: Path,
        fit_method: ModelFitMethod,
        monkeypatch: pytest.MonkeyPatch,
        centered_eight: az.InferenceData,
    ):
        def mock_mcmc(*args, **kwargs) -> az.InferenceData:
            return centered_eight.copy()

        def mock_advi(*args, **kwargs) -> Tuple[az.InferenceData, str]:
            return centered_eight.copy(), "hi"

        monkeypatch.setattr(
            speclet_model.SpecletModel, "update_observed_data", do_nothing
        )
        monkeypatch.setattr(speclet_model.SpecletModel, "mcmc_sample_model", mock_mcmc)
        monkeypatch.setattr(speclet_model.SpecletModel, "advi_sample_model", mock_advi)
        sp = MockSpecletModelClass("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp.model is None

        fit_kwargs: Dict[str, Union[float, int]]

        if fit_method == ModelFitMethod.ADVI:
            fit_kwargs = {"n_iterations": 100, "draws": 10}
        else:
            fit_kwargs = {
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 2,
                "target_accept": 0.8,
            }

        fit_kwargs["prior_pred_samples"] = 10
        fit_kwargs["post_pred_samples"] = 10

        sp.run_simulation_based_calibration(
            results_path=tmp_path,
            fit_method=fit_method,
            size=MockDataSize.SMALL,
            fit_kwargs=fit_kwargs,
        )
        assert sp.model is not None
        assert (tmp_path / "inference-data.netcdf").exists()

    def _touch_sbc_results_files(self, sbc_fm: sbc.SBCFileManager) -> None:
        for f in (
            sbc_fm.inference_data_path,
            sbc_fm.priors_path_get,
            sbc_fm.posterior_summary_path,
        ):
            f.touch()

    def test_get_sbc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_crc_dm: CrcDataManager,
        iris: pd.DataFrame,
        centered_eight: az.InferenceData,
        centered_eight_post: pd.DataFrame,
    ):
        # setup SBC results
        sbc_dir = tmp_path / "sbc-results-dir"
        if not sbc_dir.exists():
            sbc_dir.mkdir()
        sbc_fm = sbc.SBCFileManager(sbc_dir)
        sbc_fm.save_sbc_data(iris)
        self._touch_sbc_results_files(sbc_fm)

        def mock_get_sbc_results(*args: Any, **kwargs: Any) -> sbc.SBCResults:
            return sbc.SBCResults(
                priors={},
                inference_obj=centered_eight,
                posterior_summary=centered_eight_post,
            )

        monkeypatch.setattr(sbc.SBCFileManager, "get_sbc_results", mock_get_sbc_results)
        monkeypatch.setattr(speclet_model.SpecletModel, "build_model", do_nothing)
        monkeypatch.setattr(CrcDataManager, "set_data", do_nothing)

        sp = speclet_model.SpecletModel(
            "testing-get-sbc", mock_crc_dm, root_cache_dir=tmp_path, debug=True
        )
        sim_df, sbc_res, sim_sbc_fm = sp.get_sbc(sbc_dir)
        assert isinstance(sim_df, pd.DataFrame)
        assert sim_df.shape == iris.shape
        assert isinstance(sbc_res, sbc.SBCResults)
        assert isinstance(sim_sbc_fm, sbc.SBCFileManager)
        assert sim_sbc_fm.dir == sbc_fm.dir

    def test_get_sbc_errors(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp_model = speclet_model.SpecletModel(
            "testing-model", mock_crc_dm, root_cache_dir=tmp_path
        )
        sbc_dir = tmp_path / "sbc-results"
        if not sbc_dir.exists():
            sbc_dir.mkdir()
        sbc_fm = sbc.SBCFileManager(sbc_dir)

        with pytest.raises(CacheDoesNotExistError):
            _ = sp_model.get_sbc(sbc_dir)

        sbc_fm.sbc_data_path.touch()
        with pytest.raises(CacheDoesNotExistError):
            _ = sp_model.get_sbc(sbc_dir)

        sbc_fm.sbc_data_path.unlink()
        self._touch_sbc_results_files(sbc_fm)
        with pytest.raises(CacheDoesNotExistError):
            _ = sp_model.get_sbc(sbc_dir)

    def test_changing_debug_status(self, mock_sp_model: MockSpecletModelClass):
        assert mock_sp_model.data_manager is not None
        assert not mock_sp_model.debug and not mock_sp_model.data_manager.debug
        mock_sp_model.debug = True
        assert mock_sp_model.debug and mock_sp_model.data_manager.debug

    def test_changing_mcmc_sampling_params(self, mock_sp_model: MockSpecletModelClass):
        mock_sp_model.mcmc_sampling_params.draws = 12
        mock_sp_model.mcmc_sampling_params.chains = 2
        mock_sp_model.mcmc_sampling_params.cores = 2
        mock_sp_model.mcmc_sampling_params.tune = 13
        mock_sp_model.mcmc_sampling_params.prior_pred_samples = 14
        mock_sp_model.mcmc_sampling_params.post_pred_samples = 3
        mock_sp_model.build_model()
        mcmc_res = mock_sp_model.mcmc_sample_model()
        assert mcmc_res.posterior["a"].shape == (2, 12)
        assert mcmc_res.posterior_predictive["y"].shape == (1, 3, 100)
        assert mcmc_res.prior["a"].shape == (1, 14)

    def test_changing_advi_sampling_params(self, mock_sp_model: MockSpecletModelClass):
        mock_sp_model.advi_sampling_params.draws = 17
        mock_sp_model.advi_sampling_params.n_iterations = 103
        mock_sp_model.advi_sampling_params.prior_pred_samples = 14
        mock_sp_model.advi_sampling_params.post_pred_samples = 3
        mock_sp_model.build_model()
        advi_res, approx = mock_sp_model.advi_sample_model()
        assert len(approx.hist) == 103
        assert advi_res.posterior["a"].shape == (1, 17)
        assert advi_res.posterior_predictive["y"].shape == (1, 3, 100)
        assert advi_res.prior["a"].shape == (1, 14)
