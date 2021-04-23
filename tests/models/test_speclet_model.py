from pathlib import Path
from time import time

import numpy as np
import pymc3 as pm
import pytest

from src.managers.model_data_managers import CrcDataManager, MockDataManager
from src.modeling import pymc3_sampling_api as pmapi
from src.models import speclet_model


class MockSpecletModelClass(speclet_model.SpecletModel):

    data_manager = MockDataManager()

    def model_specification(self) -> pm.Model:
        data = self.data_manager.get_data()
        with pm.Model() as model:
            b = pm.Normal("b", 0, 10)
            a = pm.Normal("a", 0, 10)
            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal(  # noqa: F841
                "y", a + b * data["x"].values, sigma, observed=data["y"].values
            )
        return model


class TestSpecletModel:
    def test_build_model_fails_without_data_manager(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        with pytest.raises(AttributeError, match="without a data manager"):
            sp.build_model()

    def test_build_model_fails_without_model_spec(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        sp.data_manager = CrcDataManager(debug=True)
        with pytest.raises(AttributeError, match="`model` attribute cannot be None"):
            sp.build_model()

    def test_build_model(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            name="test-model", root_cache_dir=tmp_path, debug=True
        )
        sp.build_model()
        assert sp.model is not None
        assert isinstance(sp.model, pm.Model)

    def test_mcmc_sample_model_fails_without_model(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        with pytest.raises(AttributeError, match="model is 'None'"):
            sp.mcmc_sample_model()

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
        assert toc - tic < 1
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(mcmc_res.trace[p], mcmc_res_2.trace[p])
