#!/usr/bin/env python3

from pathlib import Path

import arviz as az
import numpy as np
import pymc3 as pm
import pytest

from src.managers.cache_managers import ArvizCacheManager, Pymc3CacheManager
from src.modeling import pymc3_sampling_api as pmapi
from src.project_enums import ModelFitMethod

UNOBSERVED_VARS = ["mu", "sigma"]
OBSERVED_VAR = "y"


@pytest.fixture(scope="module")
def pm_model() -> pm.Model:
    x = [1, 2, 3]
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        y = pm.Normal("y", mu, sigma, observed=x)  # noqa: F841
    return model


@pytest.fixture(scope="module")
def mcmc_results(pm_model: pm.Model) -> pmapi.MCMCSamplingResults:
    with pm_model:
        prior = pm.sample_prior_predictive(samples=100, random_seed=123)
        trace = pm.sample(
            100,
            tune=100,
            cores=2,
            chains=2,
            random_seed=123,
            return_inferencedata=False,
        )
        post = pm.sample_posterior_predictive(trace=trace, samples=100, random_seed=123)
    return pmapi.MCMCSamplingResults(
        trace=trace, prior_predictive=prior, posterior_predictive=post
    )


@pytest.fixture(scope="module")
def advi_results(pm_model: pm.Model) -> pmapi.ApproximationSamplingResults:
    with pm_model:
        prior = pm.sample_prior_predictive(samples=100, random_seed=123)
        approx = pm.fit(
            100,
            callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
            random_seed=123,
        )
        trace = approx.sample(draws=100)
        post = pm.sample_posterior_predictive(trace=trace, samples=100, random_seed=123)
    return pmapi.ApproximationSamplingResults(
        trace=trace,
        prior_predictive=prior,
        posterior_predictive=post,
        approximation=approx,
    )


class TestPymc3CacheManager:
    def test_cache_dir_is_made(self, tmp_path: Path):
        cache_dir = tmp_path / "pymc3-cache-dir"
        assert not cache_dir.exists()
        _ = Pymc3CacheManager(cache_dir=tmp_path / cache_dir)
        assert cache_dir.exists()

    def test_generation_of_cache_paths(self, tmp_path: Path):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        cache_paths = cm.get_cache_file_names()

        for path in cache_paths.dict().values():
            assert tmp_path == path.parent

        assert "prior" in cache_paths.prior_predictive_path.name
        assert "posterior" in cache_paths.posterior_predictive_path.name
        assert "trace" in cache_paths.trace_path.name
        assert "approx" in cache_paths.approximation_path.name

    @pytest.mark.slow
    def test_caching_mcmc(
        self, tmp_path: Path, mcmc_results: pmapi.MCMCSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(res=mcmc_results)
        cache_paths = cm.get_cache_file_names()
        assert (
            cache_paths.prior_predictive_path.exists()
            and cache_paths.prior_predictive_path.is_file()
        )
        assert (
            cache_paths.posterior_predictive_path.exists()
            and cache_paths.posterior_predictive_path.is_file()
        )
        assert cache_paths.trace_path.exists() and cache_paths.trace_path.is_dir()
        assert not cache_paths.approximation_path.exists()

    @pytest.mark.slow
    def test_caching_advi(
        self, tmp_path: Path, advi_results: pmapi.ApproximationSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(res=advi_results)
        cache_paths = cm.get_cache_file_names()
        assert (
            cache_paths.prior_predictive_path.exists()
            and cache_paths.prior_predictive_path.is_file()
        )
        assert (
            cache_paths.posterior_predictive_path.exists()
            and cache_paths.posterior_predictive_path.is_file()
        )
        assert (
            cache_paths.approximation_path.exists()
            and cache_paths.approximation_path.is_file()
        )
        assert not cache_paths.trace_path.exists()

    @pytest.mark.slow
    def test_read_mcmc_cache(
        self,
        tmp_path: Path,
        pm_model: pm.Model,
        mcmc_results: pmapi.MCMCSamplingResults,
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(res=mcmc_results)
        cached_res = cm.read_cached_sampling(model=pm_model)
        assert isinstance(cached_res, pmapi.MCMCSamplingResults)
        for p in UNOBSERVED_VARS:
            np.testing.assert_array_equal(
                mcmc_results.prior_predictive[p], cached_res.prior_predictive[p]
            )
            np.testing.assert_array_equal(mcmc_results.trace[p], cached_res.trace[p])

        y = OBSERVED_VAR
        np.testing.assert_array_equal(
            mcmc_results.prior_predictive[y], cached_res.prior_predictive[y]
        )
        np.testing.assert_array_equal(
            mcmc_results.posterior_predictive[y], cached_res.posterior_predictive[y]
        )

    @pytest.mark.slow
    def test_read_advi_cache(
        self, tmp_path: Path, advi_results: pmapi.ApproximationSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(res=advi_results)
        N_DRAWS = 120
        cached_res = cm.read_cached_approximation(draws=N_DRAWS)
        assert isinstance(cached_res, pmapi.ApproximationSamplingResults)
        for p in UNOBSERVED_VARS:
            np.testing.assert_array_equal(
                advi_results.prior_predictive[p], cached_res.prior_predictive[p]
            )

        np.testing.assert_array_equal(
            advi_results.approximation.hist, cached_res.approximation.hist
        )
        y = OBSERVED_VAR
        np.testing.assert_array_equal(
            advi_results.prior_predictive[y], cached_res.prior_predictive[y]
        )
        np.testing.assert_array_equal(
            advi_results.posterior_predictive[y], cached_res.posterior_predictive[y]
        )

        assert cached_res.trace[UNOBSERVED_VARS[0]].shape[0] == N_DRAWS
        cached_res = cm.read_cached_approximation(draws=53)
        assert cached_res.trace[UNOBSERVED_VARS[0]].shape[0] == 53

    @pytest.mark.slow
    def test_advi_cache_exists(
        self, tmp_path: Path, advi_results: pmapi.ApproximationSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        # Cache results and check existence.
        cm.cache_sampling_results(res=advi_results)
        assert cm.cache_exists(ModelFitMethod.advi)
        assert not cm.cache_exists(ModelFitMethod.mcmc)

    @pytest.mark.slow
    def test_mcmc_cache_exists(
        self, tmp_path: Path, mcmc_results: pmapi.MCMCSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        # Cache results and check existence.
        cm.cache_sampling_results(res=mcmc_results)
        assert not cm.cache_exists(ModelFitMethod.advi)
        assert cm.cache_exists(ModelFitMethod.mcmc)

    @pytest.mark.slow
    def test_clear_advi_cache(
        self, tmp_path: Path, advi_results: pmapi.ApproximationSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        cm.cache_sampling_results(res=advi_results)
        assert cm.cache_exists(ModelFitMethod.advi)
        cm.clear_cache()
        assert not cm.cache_exists(ModelFitMethod.advi)

    @pytest.mark.slow
    def test_clear_mcmc_cache(
        self, tmp_path: Path, mcmc_results: pmapi.MCMCSamplingResults
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        cm.cache_sampling_results(res=mcmc_results)
        assert cm.cache_exists(ModelFitMethod.mcmc)
        cm.clear_cache()
        assert not cm.cache_exists(ModelFitMethod.mcmc)

    @pytest.mark.slow
    def test_clear_mcmc_and_advi_cache(
        self,
        tmp_path: Path,
        mcmc_results: pmapi.MCMCSamplingResults,
        advi_results: pmapi.ApproximationSamplingResults,
    ):
        cm = Pymc3CacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        cm.cache_sampling_results(res=mcmc_results)
        cm.cache_sampling_results(res=advi_results)
        assert cm.cache_exists(ModelFitMethod.mcmc) and cm.cache_exists(
            ModelFitMethod.advi
        )
        cm.clear_cache()
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)


class TestArvizCacheManager:
    @pytest.fixture(scope="class")
    def mcmc_inf_data(
        self, pm_model: pm.Model, mcmc_results: pmapi.MCMCSamplingResults
    ) -> az.InferenceData:
        return az.from_pymc3(
            trace=mcmc_results.trace,
            prior=mcmc_results.prior_predictive,
            posterior_predictive=mcmc_results.posterior_predictive,
            model=pm_model,
        )

    @pytest.fixture(scope="class")
    def advi_inf_data(
        self, pm_model: pm.Model, advi_results: pmapi.ApproximationSamplingResults
    ) -> az.InferenceData:
        return az.from_pymc3(
            trace=advi_results.trace,
            prior=advi_results.prior_predictive,
            posterior_predictive=advi_results.posterior_predictive,
            model=pm_model,
        )

    def test_cache_dir_is_made(self, tmp_path: Path):
        cache_dir = tmp_path / "pymc3-cache-dir"
        assert not cache_dir.exists()
        _ = ArvizCacheManager(cache_dir=tmp_path / cache_dir)
        assert cache_dir.exists()

    def test_generation_of_cache_paths(self, tmp_path: Path):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cache_paths = cm.get_cache_file_names()

        for path in cache_paths.dict().values():
            assert tmp_path == path.parent

    @pytest.mark.slow
    def test_caching_mcmc(self, tmp_path: Path, mcmc_inf_data: az.InferenceData):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(mcmc_inf_data)
        cache_paths = cm.get_cache_file_names()
        assert (
            cache_paths.inference_data_path.exists()
            and cache_paths.inference_data_path.is_file()
        )
        assert not cache_paths.approximation_path.exists()

    @pytest.mark.slow
    def test_caching_advi(
        self,
        tmp_path: Path,
        advi_inf_data: az.InferenceData,
        advi_results: pmapi.ApproximationSamplingResults,
    ):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(
            advi_inf_data, approximation=advi_results.approximation
        )
        cache_paths = cm.get_cache_file_names()
        assert (
            cache_paths.inference_data_path.exists()
            and cache_paths.inference_data_path.is_file()
        )
        assert (
            cache_paths.approximation_path.exists()
            and cache_paths.approximation_path.is_file()
        )

    @staticmethod
    def compare_inference_datas(id1: az.InferenceData, id2: az.InferenceData):
        for p in UNOBSERVED_VARS:
            np.testing.assert_array_equal(id1.prior[p], id2.prior[p])
            np.testing.assert_array_equal(id1.posterior[p], id2.posterior[p])

        y = OBSERVED_VAR
        np.testing.assert_array_equal(id1.prior_predictive[y], id2.prior_predictive[y])
        np.testing.assert_array_equal(
            id1.posterior_predictive[y],
            id2.posterior_predictive[y],
        )

    @pytest.mark.slow
    def test_read_mcmc_cache(
        self,
        tmp_path: Path,
        pm_model: pm.Model,
        mcmc_inf_data: az.InferenceData,
    ):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(mcmc_inf_data)
        cached_res = cm.read_cached_sampling()
        assert isinstance(cached_res, az.InferenceData)
        TestArvizCacheManager.compare_inference_datas(mcmc_inf_data, cached_res)

    @pytest.mark.slow
    def test_read_advi_cache(
        self,
        tmp_path: Path,
        advi_inf_data: az.InferenceData,
        advi_results: pmapi.ApproximationSamplingResults,
    ):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(
            advi_inf_data, approximation=advi_results.approximation
        )
        cached_res, approx = cm.read_cached_approximation()
        assert isinstance(cached_res, az.InferenceData)
        assert isinstance(approx, pm.Approximation)
        TestArvizCacheManager.compare_inference_datas(advi_inf_data, cached_res)

    @pytest.mark.slow
    def test_advi_cache_exists(
        self,
        tmp_path: Path,
        advi_inf_data: az.InferenceData,
        advi_results: pmapi.ApproximationSamplingResults,
    ):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        # Cache results and check existence.
        cm.cache_sampling_results(
            advi_inf_data, approximation=advi_results.approximation
        )
        assert cm.cache_exists(ModelFitMethod.advi)

    @pytest.mark.slow
    def test_mcmc_cache_exists(self, tmp_path: Path, mcmc_inf_data: az.InferenceData):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        # Cache results and check existence.
        cm.cache_sampling_results(mcmc_inf_data)
        assert not cm.cache_exists(ModelFitMethod.advi)
        assert cm.cache_exists(ModelFitMethod.mcmc)

    @pytest.mark.slow
    def test_clear_advi_cache(
        self,
        tmp_path: Path,
        advi_inf_data: az.InferenceData,
        advi_results: pmapi.ApproximationSamplingResults,
    ):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        cm.cache_sampling_results(
            advi_inf_data, approximation=advi_results.approximation
        )
        assert cm.cache_exists(ModelFitMethod.advi)
        cm.clear_cache()
        assert not cm.cache_exists(ModelFitMethod.advi)

    @pytest.mark.slow
    def test_clear_mcmc_cache(self, tmp_path: Path, mcmc_inf_data: az.InferenceData):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        cm.cache_sampling_results(mcmc_inf_data)
        assert cm.cache_exists(ModelFitMethod.mcmc)
        cm.clear_cache()
        assert not cm.cache_exists(ModelFitMethod.mcmc)

    @pytest.mark.slow
    def test_clear_mcmc_and_advi_cache(
        self,
        tmp_path: Path,
        mcmc_inf_data: az.InferenceData,
        advi_inf_data: az.InferenceData,
        advi_results: pmapi.ApproximationSamplingResults,
    ):
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
        cm.cache_sampling_results(mcmc_inf_data)
        cm.cache_sampling_results(
            advi_inf_data, approximation=advi_results.approximation
        )
        assert cm.cache_exists(ModelFitMethod.mcmc) and cm.cache_exists(
            ModelFitMethod.advi
        )
        cm.clear_cache()
        assert not cm.cache_exists(method=ModelFitMethod.mcmc)
        assert not cm.cache_exists(method=ModelFitMethod.advi)
