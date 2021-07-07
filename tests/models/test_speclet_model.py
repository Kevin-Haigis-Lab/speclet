from pathlib import Path
from time import time
from typing import Dict, Tuple, Union

import arviz as az
import numpy as np
import pymc3 as pm
import pytest

from src.managers.model_data_managers import MockDataManager
from src.models import speclet_model
from src.project_enums import ModelFitMethod


class MockSpecletModelClass(speclet_model.SpecletModel):
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


class TestSpecletModel:
    def test_mcmc_sample_model_fails_without_overriding(self, tmp_path: Path):
        sp = speclet_model.SpecletModel(
            "test-model", data_manager=MockDataManager(), root_cache_dir=tmp_path
        )
        with pytest.raises(AttributeError, match="Cannot sample: model is 'None'"):
            sp.mcmc_sample_model()

    @pytest.mark.slow
    def test_build_model(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            name="test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=MockDataManager(),
        )
        assert sp.model is None
        sp.build_model()
        assert sp.model is not None
        assert isinstance(sp.model, pm.Model)

    @pytest.mark.slow
    def test_mcmc_sample_model(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            "test-model", root_cache_dir=tmp_path, data_manager=MockDataManager()
        )
        sp.build_model()
        mcmc_res = sp.mcmc_sample_model(
            draws=100,
            tune=100,
            chains=2,
            cores=2,
            prior_pred_samples=100,
            post_pred_samples=10,
            random_seed=1,
        )
        assert isinstance(mcmc_res, az.InferenceData)
        for p in ["a", "b", "sigma"]:
            assert mcmc_res.posterior[p].shape == (2, 100)

        assert sp.mcmc_results is not None

        tic = time()
        mcmc_res_2 = sp.mcmc_sample_model(
            draws=100,
            tune=100,
            chains=2,
            cores=2,
            prior_pred_samples=100,
            post_pred_samples=10,
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
    def test_advi_sample_model(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            "test-model", root_cache_dir=tmp_path, data_manager=MockDataManager()
        )
        sp.build_model()
        advi_res, advi_approx = sp.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=100,
            post_pred_samples=10,
            random_seed=1,
        )
        assert isinstance(advi_res, az.InferenceData)
        for p in ["a", "b", "sigma"]:
            assert advi_res.posterior[p].shape == (1, 100)

        assert sp.advi_results is not None

        tic = time()
        advi_res_2, advi_approx_2 = sp.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=100,
            post_pred_samples=10,
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
    ):
        def identity(*args, **kwargs) -> None:
            return None

        monkeypatch.setattr(
            speclet_model.SpecletModel, "update_observed_data", identity
        )
        sp = MockSpecletModelClass(
            "test-model", root_cache_dir=tmp_path, data_manager=MockDataManager()
        )
        assert sp.model is None

        fit_kwargs: Dict[str, Union[float, int]]

        if fit_method == ModelFitMethod.ADVI:
            fit_kwargs = {"n_iterations": 100, "draws": 10}
        else:
            fit_kwargs = {
                "mcmc_draws": 10,
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
            size="small",
            fit_kwargs=fit_kwargs,
        )
        assert sp.model is not None
        assert (tmp_path / "inference-data.netcdf").exists()

    def test_changing_debug_status(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            "test-model",
            root_cache_dir=tmp_path,
            debug=False,
            data_manager=MockDataManager(debug=False),
        )
        assert sp.data_manager is not None
        assert not sp.debug and not sp.data_manager.debug
        sp.debug = True
        assert sp.debug and sp.data_manager.debug

    def test_changing_mcmc_sampling_params(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            "test-model",
            root_cache_dir=tmp_path,
            debug=False,
            data_manager=MockDataManager(debug=False),
        )
        sp.mcmc_sampling_params.draws = 12
        sp.mcmc_sampling_params.chains = 2
        sp.mcmc_sampling_params.cores = 2
        sp.mcmc_sampling_params.tune = 13
        sp.mcmc_sampling_params.prior_pred_samples = 14
        sp.build_model()
        mcmc_res = sp.mcmc_sample_model()
        assert mcmc_res.posterior["a"].shape == (2, 12)
        assert mcmc_res.posterior_predictive["y"].shape == (2, 12, 100)
        assert mcmc_res.prior["a"].shape == (1, 14)

    def test_changing_advi_sampling_params(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            "test-model",
            root_cache_dir=tmp_path,
            debug=False,
            data_manager=MockDataManager(debug=False),
        )
        sp.advi_sampling_params.draws = 17
        sp.advi_sampling_params.n_iterations = 103
        sp.advi_sampling_params.prior_pred_samples = 14
        sp.build_model()
        advi_res, approx = sp.advi_sample_model()
        assert len(approx.hist) == 103
        assert advi_res.posterior["a"].shape == (1, 17)
        assert advi_res.posterior_predictive["y"].shape == (1, 17, 100)
        assert advi_res.prior["a"].shape == (1, 14)
