from pathlib import Path

import arviz as az
import numpy as np
import pymc3 as pm
import pytest

from src.managers.cache_managers import ArvizCacheManager
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
def mcmc_results(pm_model: pm.Model) -> az.InferenceData:
    with pm_model:
        prior = pm.sample_prior_predictive(samples=100, random_seed=123)
        trace = pm.sample(
            100,
            tune=100,
            cores=1,
            chains=2,
            random_seed=123,
            return_inferencedata=True,
        )
        post = pm.sample_posterior_predictive(trace=trace, samples=100, random_seed=123)

    assert isinstance(trace, az.InferenceData)

    with pm_model:
        trace.extend(az.from_pymc3(prior=prior, posterior_predictive=post))

    return trace


@pytest.fixture(scope="module")
def advi_results(pm_model: pm.Model) -> pmapi.ApproximationSamplingResults:
    with pm_model:
        prior = pm.sample_prior_predictive(samples=100, random_seed=123)
        approx = pm.fit(
            100,
            callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)],
            random_seed=123,
        )
        trace = az.from_pymc3(trace=approx.sample(draws=100))
        post = pm.sample_posterior_predictive(trace=trace, samples=100, random_seed=123)

    assert isinstance(trace, az.InferenceData)

    with pm_model:
        trace.extend(az.from_pymc3(prior=prior, posterior_predictive=post))

    return pmapi.ApproximationSamplingResults(
        inference_data=trace, approximation=approx
    )


class TestArvizCacheManager:
    def test_cache_dir_is_made(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "pymc3-cache-dir"
        assert not cache_dir.exists()
        _ = ArvizCacheManager(cache_dir=tmp_path / cache_dir)
        assert cache_dir.exists()

    def test_generation_of_cache_paths(self, tmp_path: Path) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cache_paths = cm.get_cache_file_names()

        for path in cache_paths.dict().values():
            assert tmp_path == path.parent

    @pytest.mark.slow
    def test_caching_mcmc(self, tmp_path: Path, mcmc_results: az.InferenceData) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(mcmc_results)
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
        advi_results: pmapi.ApproximationSamplingResults,
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(
            advi_results.inference_data, advi_results.approximation
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
    def compare_inference_datas(id1: az.InferenceData, id2: az.InferenceData) -> None:
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
        mcmc_results: az.InferenceData,
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(mcmc_results)
        cached_res = cm.read_cached_sampling()
        assert isinstance(cached_res, az.InferenceData)
        TestArvizCacheManager.compare_inference_datas(mcmc_results, cached_res)

    @pytest.mark.slow
    def test_read_advi_cache(
        self,
        tmp_path: Path,
        advi_results: pmapi.ApproximationSamplingResults,
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(
            advi_results.inference_data, approximation=advi_results.approximation
        )
        cached_res = cm.read_cached_approximation()
        assert isinstance(cached_res.inference_data, az.InferenceData)
        assert isinstance(cached_res.approximation, pm.Approximation)
        TestArvizCacheManager.compare_inference_datas(
            advi_results.inference_data, cached_res.inference_data
        )

    @pytest.mark.slow
    def test_advi_cache_exists(
        self,
        tmp_path: Path,
        advi_results: pmapi.ApproximationSamplingResults,
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.MCMC)
        assert not cm.cache_exists(method=ModelFitMethod.ADVI)
        # Cache results and check existence.
        cm.cache_sampling_results(
            advi_results.inference_data, approximation=advi_results.approximation
        )
        assert cm.cache_exists(ModelFitMethod.ADVI)

    @pytest.mark.slow
    def test_mcmc_cache_exists(
        self, tmp_path: Path, mcmc_results: az.InferenceData
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.MCMC)
        assert not cm.cache_exists(method=ModelFitMethod.ADVI)
        # Cache results and check existence.
        cm.cache_sampling_results(mcmc_results)
        assert not cm.cache_exists(ModelFitMethod.ADVI)
        assert cm.cache_exists(ModelFitMethod.MCMC)

    @pytest.mark.slow
    def test_clear_advi_cache(
        self,
        tmp_path: Path,
        advi_results: pmapi.ApproximationSamplingResults,
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.ADVI)
        cm.cache_sampling_results(
            advi_results.inference_data, approximation=advi_results.approximation
        )
        assert cm.cache_exists(ModelFitMethod.ADVI)
        cm.clear_cache()
        assert not cm.cache_exists(ModelFitMethod.ADVI)

    @pytest.mark.slow
    def test_clear_mcmc_cache(
        self, tmp_path: Path, mcmc_results: az.InferenceData
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.MCMC)
        cm.cache_sampling_results(mcmc_results)
        assert cm.cache_exists(ModelFitMethod.MCMC)
        cm.clear_cache()
        assert not cm.cache_exists(ModelFitMethod.MCMC)

    @pytest.mark.slow
    def test_clear_mcmc_and_advi_cache(
        self,
        tmp_path: Path,
        mcmc_results: az.InferenceData,
        advi_results: pmapi.ApproximationSamplingResults,
    ) -> None:
        cm = ArvizCacheManager(cache_dir=tmp_path)
        assert not cm.cache_exists(method=ModelFitMethod.MCMC)
        assert not cm.cache_exists(method=ModelFitMethod.ADVI)
        cm.cache_sampling_results(mcmc_results)
        cm.cache_sampling_results(
            advi_results.inference_data, approximation=advi_results.approximation
        )
        assert cm.cache_exists(ModelFitMethod.MCMC) and cm.cache_exists(
            ModelFitMethod.ADVI
        )
        cm.clear_cache()
        assert not cm.cache_exists(method=ModelFitMethod.MCMC)
        assert not cm.cache_exists(method=ModelFitMethod.ADVI)
