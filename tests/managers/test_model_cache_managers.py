#!/usr/bin/env python3

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.modeling import pymc3_sampling_api as pmapi


class TestPymc3ModelCacheManager:
    @pytest.fixture(scope="class")
    def data(self) -> pd.DataFrame:
        real_a = 1
        real_b = 2
        x = np.random.uniform(-10, 10, 100)
        real_sigma = 0.1
        y = real_a + real_b * x + np.random.normal(0, real_sigma, len(x))
        return pd.DataFrame({"x": x, "y_obs": y})

    @pytest.fixture(scope="class")
    def pm_model(self, data: pd.DataFrame) -> pm.Model:
        with pm.Model() as model:
            a = pm.Normal("a", 0, 5)
            b = pm.Normal("b", 0, 1)
            mu = a + b * data.x.values
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=data.y_obs.values)  # noqa: F841
        return model

    @pytest.fixture(scope="class")
    def mcmc_results(self, data: pd.DataFrame, pm_model: pm.Model) -> az.InferenceData:
        with pm_model:
            prior = pm.sample_prior_predictive(samples=100, random_seed=123)
            trace = pm.sample(
                1000,
                tune=1000,
                cores=2,
                chains=2,
                random_seed=123,
                return_inferencedata=False,
            )
            post = pm.sample_posterior_predictive(
                trace=trace, samples=100, random_seed=123
            )
        res = pmapi.MCMCSamplingResults(
            trace=trace, prior_predictive=prior, posterior_predictive=post
        )
        return pmapi.convert_samples_to_arviz(model=pm_model, res=res)

    @pytest.fixture(scope="class")
    def advi_fit(
        self, data: pd.DataFrame, pm_model: pm.Model
    ) -> pmapi.ApproximationSamplingResults:
        with pm_model:
            prior = pm.sample_prior_predictive(samples=100, random_seed=123)
            approx = pm.fit(
                100000,
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
                random_seed=123,
            )
            trace = approx.sample(draws=1000)
            post = pm.sample_posterior_predictive(
                trace=trace, samples=100, random_seed=123
            )
        res = pmapi.ApproximationSamplingResults(
            trace=trace,
            prior_predictive=prior,
            posterior_predictive=post,
            approximation=approx,
        )
        return res

    @pytest.fixture(scope="class")
    def advi_results(
        self, pm_model: pm.Model, advi_fit: pmapi.ApproximationSamplingResults
    ) -> az.InferenceData:
        return pmapi.convert_samples_to_arviz(model=pm_model, res=advi_fit)

    @pytest.mark.slow
    def test_writing_mcmc_cache(self, mcmc_results: az.InferenceData, tmp_path: Path):
        assert len(list(tmp_path.iterdir())) == 0
        cm = Pymc3ModelCacheManager(name="test-mcmc-model", root_cache_dir=tmp_path)
        assert len(list(tmp_path.iterdir())) == 1
        assert len(list(cm.cache_dir.iterdir())) == 2
        cm.write_mcmc_cache(mcmc_results)

        assert cm.mcmc_cache_exists()
        assert not cm.advi_cache_exists()

    @staticmethod
    def compare_inference_datas(id1: az.InferenceData, id2: az.InferenceData):
        for param in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(id1.prior[param], id2.prior[param])
            np.testing.assert_array_equal(id1.posterior[param], id2.posterior[param])

        np.testing.assert_array_equal(
            id1.prior_predictive["y"], id2.prior_predictive["y"]
        )
        np.testing.assert_array_equal(
            id1.posterior_predictive["y"],
            id2.posterior_predictive["y"],
        )

    @pytest.mark.slow
    def test_reading_mcmc_cache(
        self,
        pm_model: pm.Model,
        mcmc_results: az.InferenceData,
        tmp_path: Path,
    ):
        assert len(list(tmp_path.iterdir())) == 0
        cm = Pymc3ModelCacheManager(name="test-mcmc-model2", root_cache_dir=tmp_path)
        cm.write_mcmc_cache(mcmc_results)
        new_res = cm.get_mcmc_cache()
        TestPymc3ModelCacheManager.compare_inference_datas(mcmc_results, new_res)

    @pytest.mark.slow
    def test_writing_advi_cache(
        self,
        advi_fit: pmapi.ApproximationSamplingResults,
        advi_results: az.InferenceData,
        tmp_path: Path,
    ):
        assert len(list(tmp_path.iterdir())) == 0
        cm = Pymc3ModelCacheManager(name="test-advi-model", root_cache_dir=tmp_path)
        assert len(list(tmp_path.iterdir())) == 1
        assert len(list(cm.cache_dir.iterdir())) == 2
        cm.write_advi_cache(advi_results, approx=advi_fit.approximation)

        assert not cm.mcmc_cache_exists()
        assert cm.advi_cache_exists()

    @pytest.mark.slow
    def test_reading_advi_cache(
        self,
        advi_fit: pmapi.ApproximationSamplingResults,
        advi_results: az.InferenceData,
        tmp_path: Path,
    ):
        assert len(list(tmp_path.iterdir())) == 0
        cm = Pymc3ModelCacheManager(name="test-advi-model", root_cache_dir=tmp_path)
        cm.write_advi_cache(advi_results, approx=advi_fit.approximation)
        new_res, _ = cm.get_advi_cache()
        TestPymc3ModelCacheManager.compare_inference_datas(advi_results, new_res)
