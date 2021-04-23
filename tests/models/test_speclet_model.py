from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import pymc3 as pm
import pytest

from src.managers.model_data_managers import MockDataManager
from src.modeling import pymc3_sampling_api as pmapi
from src.models import speclet_model


class MockSpecletModelClass(speclet_model.SpecletModel):

    data_manager = MockDataManager()

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
    def test_build_model_fails_without_data_manager(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        with pytest.raises(AttributeError, match="without a data manager"):
            sp.build_model()

    def test_mcmc_sample_model_fails_without_overriding(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        with pytest.raises(AttributeError, match="Cannot sample: model is 'None'"):
            sp.mcmc_sample_model()

    def test_build_model(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            name="test-model", root_cache_dir=tmp_path, debug=True
        )
        assert sp.model is None
        sp.build_model()
        assert sp.model is not None
        assert isinstance(sp.model, pm.Model)

    @pytest.mark.slow
    def test_mcmc_sample_model(self, tmp_path: Path):
        sp = MockSpecletModelClass("test-model", root_cache_dir=tmp_path)
        sp.build_model()
        mcmc_res = sp.mcmc_sample_model(
            mcmc_draws=100,
            tune=100,
            chains=2,
            cores=2,
            prior_pred_samples=100,
            post_pred_samples=10,
            random_seed=1,
        )
        assert isinstance(mcmc_res, pmapi.MCMCSamplingResults)
        for p in ["a", "b", "sigma"]:
            assert mcmc_res.trace[p].shape == (100 * 2,)

        assert sp.mcmc_results is not None

        tic = time()
        mcmc_res_2 = sp.mcmc_sample_model(
            mcmc_draws=100,
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
            np.testing.assert_array_equal(mcmc_res.trace[p], mcmc_res_2.trace[p])

    def test_advi_sample_model_fails_without_model(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        with pytest.raises(AttributeError, match="model is 'None'"):
            sp.advi_sample_model()

    @pytest.mark.slow
    def test_advi_sample_model(self, tmp_path: Path):
        sp = MockSpecletModelClass("test-model", root_cache_dir=tmp_path)
        sp.build_model()
        advi_res = sp.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=100,
            post_pred_samples=10,
            random_seed=1,
        )
        assert isinstance(advi_res, pmapi.ApproximationSamplingResults)
        for p in ["a", "b", "sigma"]:
            assert advi_res.trace[p].shape == (100,)

        assert sp.advi_results is not None

        tic = time()
        advi_res_2 = sp.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=100,
            post_pred_samples=10,
            random_seed=1,
        )
        toc = time()

        assert advi_res is advi_res_2
        assert toc - tic < 1
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(advi_res.trace[p], advi_res_2.trace[p])

    @pytest.mark.slow
    def test_run_simulation_based_calibration(self, tmp_path: Path):
        sp = MockSpecletModelClass("test-model", root_cache_dir=tmp_path)
        assert sp.model is None
        sp.run_simulation_based_calibration(
            results_path=tmp_path,
            size="small",
            fit_kwargs={
                "n_iterations": 100,
                "draws": 10,
                "prior_pred_samples": 10,
                "post_pred_samples": 10,
            },
        )
        assert sp.model is not None
        assert (tmp_path / "inference-data.netcdf").exists()
