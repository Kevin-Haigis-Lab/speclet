#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from src.modeling import cache_manager
from src.modeling import pymc3_sampling_api as pmapi


class TestPymc3CacheManager:
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
    def mcmc_results(self, pm_model: pm.Model) -> pmapi.MCMCSamplingResults:
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
        return pmapi.MCMCSamplingResults(
            trace=trace, prior_predictive=prior, posterior_predictive=post
        )

    @pytest.fixture(scope="class")
    def advi_results(self, pm_model: pm.Model) -> pmapi.ApproximationSamplingResults:
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
        return pmapi.ApproximationSamplingResults(
            trace=trace,
            prior_predictive=prior,
            posterior_predictive=post,
            approximation=approx,
        )

    def test_cache_dir_is_made(self, tmp_path: Path):
        cache_dir = tmp_path / "pymc3-cache-dir"
        assert not cache_dir.exists()
        _ = cache_manager.Pymc3CacheManager(cache_dir=tmp_path / cache_dir)
        assert cache_dir.exists()

    def test_writing_pickle(self, tmp_path: Path):
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
        fp = tmp_path / "my-pickle.pkl"
        p = "pickle"
        cm._write_pickle(p, fp)
        assert fp.exists()

    def test_reading_pickle(self, tmp_path: Path):
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
        fp = tmp_path / "my-pickle.pkl"
        p = "pickle"
        cm._write_pickle(p, fp)
        assert fp.exists()
        new_p = cm._get_pickle(fp)
        assert new_p == p

    def test_generation_of_cache_paths(self, tmp_path: Path):
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
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
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
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
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
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
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(res=mcmc_results)
        cached_res = cm.read_cached_sampling(model=pm_model)
        assert isinstance(cached_res, pmapi.MCMCSamplingResults)
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(
                mcmc_results.prior_predictive[p], cached_res.prior_predictive[p]
            )
            np.testing.assert_array_equal(mcmc_results.trace[p], cached_res.trace[p])

        np.testing.assert_array_equal(
            mcmc_results.prior_predictive["y"], cached_res.prior_predictive["y"]
        )
        np.testing.assert_array_equal(
            mcmc_results.posterior_predictive["y"], cached_res.posterior_predictive["y"]
        )

    @pytest.mark.slow
    def test_read_advi_cache(
        self, tmp_path: Path, advi_results: pmapi.ApproximationSamplingResults
    ):
        cm = cache_manager.Pymc3CacheManager(cache_dir=tmp_path)
        cm.cache_sampling_results(res=advi_results)
        cached_res = cm.read_cached_approximation(draws=1000)
        assert isinstance(cached_res, pmapi.ApproximationSamplingResults)
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(
                advi_results.prior_predictive[p], cached_res.prior_predictive[p]
            )

        np.testing.assert_array_equal(
            advi_results.approximation.hist, cached_res.approximation.hist
        )
        np.testing.assert_array_equal(
            advi_results.prior_predictive["y"], cached_res.prior_predictive["y"]
        )
        np.testing.assert_array_equal(
            advi_results.posterior_predictive["y"], cached_res.posterior_predictive["y"]
        )

        assert cached_res.trace["a"].shape[0] == 1000
        cached_res = cm.read_cached_approximation(draws=583)
        assert cached_res.trace["a"].shape[0] == 583
