#!/usr/bin/env python3

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from src.modeling import pymc3_sampling_api as pmapi
from src.models import speclet_model


class TestSpecletModel:
    def test_writing_pickle(self, tmp_path: Path):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        fp = tmp_path / "my-pickle.pkl"
        p = "pickle"
        model._write_pickle(p, fp)
        assert fp.exists()

    def test_reading_pickle(self, tmp_path: Path):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        fp = tmp_path / "my-pickle.pkl"
        p = "pickle"
        model._write_pickle(p, fp)
        assert fp.exists()
        new_p = model._get_pickle(fp)
        assert new_p == p

    def test_error_generation_of_cache_paths(self):
        with pytest.raises(ValueError):
            model = speclet_model.SpecletModel(cache_dir=None)
            _ = model.get_cache_file_names()

    def test_generation_of_cache_paths(self, tmp_path: Path):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        cache_paths = model.get_cache_file_names()

        for path in cache_paths.dict().values():
            assert tmp_path == path.parent

        assert "prior" in cache_paths.prior_predictive_path.name
        assert "posterior" in cache_paths.posterior_predictive_path.name
        assert "trace" in cache_paths.trace_path.name
        assert "approx" in cache_paths.approximation_path.name

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
            y = pm.Normal("y", mu, sigma, observed=data.y_obs.values)
        return model

    @pytest.fixture(scope="class")
    def mcmc_results(
        self, data: pd.DataFrame, pm_model: pm.Model
    ) -> pmapi.MCMCSamplingResults:
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
    def advi_results(
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
        return pmapi.ApproximationSamplingResults(
            trace=trace,
            prior_predictive=prior,
            posterior_predictive=post,
            approximation=approx,
        )

    @pytest.mark.slow
    def test_writing_mcmc_cache(
        self, mcmc_results: pmapi.MCMCSamplingResults, tmp_path: Path
    ):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        assert len(list(tmp_path.iterdir())) == 0
        model.cache_sampling_results(mcmc_results)
        assert len(list(tmp_path.iterdir())) > 0

        cache_paths = model.get_cache_file_names()
        assert (
            cache_paths.prior_predictive_path.is_file()
            and cache_paths.prior_predictive_path.exists()
        )
        assert (
            cache_paths.posterior_predictive_path.is_file()
            and cache_paths.posterior_predictive_path.exists()
        )
        assert cache_paths.trace_path.is_dir() and cache_paths.trace_path.exists()
        pickled_trace_dirs = [x.name for x in cache_paths.trace_path.iterdir()]
        for i in ["0", "1"]:
            assert i in pickled_trace_dirs

    @pytest.mark.slow
    def test_reading_mcmc_cache(
        self,
        pm_model: pm.Model,
        mcmc_results: pmapi.MCMCSamplingResults,
        tmp_path: Path,
    ):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        assert len(list(tmp_path.iterdir())) == 0
        model.cache_sampling_results(mcmc_results)
        assert len(list(tmp_path.iterdir())) > 0
        new_res = model.read_cached_sampling(model=pm_model)

        for param in ["a", "b", "sigma"]:
            np.testing.assert_almost_equal(
                mcmc_results.prior_predictive[param],
                new_res.prior_predictive[param],
                decimal=5,
            )
            np.testing.assert_almost_equal(
                mcmc_results.trace[param], new_res.trace[param], decimal=5
            )

        np.testing.assert_almost_equal(
            mcmc_results.prior_predictive["y"], new_res.prior_predictive["y"], decimal=5
        )
        np.testing.assert_almost_equal(
            mcmc_results.posterior_predictive["y"],
            new_res.posterior_predictive["y"],
            decimal=5,
        )

    @pytest.mark.slow
    def test_writing_advi_cache(
        self, advi_results: pmapi.ApproximationSamplingResults, tmp_path: Path
    ):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        assert len(list(tmp_path.iterdir())) == 0
        model.cache_sampling_results(advi_results)
        assert len(list(tmp_path.iterdir())) > 0

        cache_paths = model.get_cache_file_names()
        assert (
            cache_paths.prior_predictive_path.is_file()
            and cache_paths.prior_predictive_path.exists()
        )
        assert (
            cache_paths.posterior_predictive_path.is_file()
            and cache_paths.posterior_predictive_path.exists()
        )
        assert (
            cache_paths.approximation_path.is_file()
            and cache_paths.approximation_path.exists()
        )
        assert cache_paths.trace_path.is_dir() and cache_paths.trace_path.exists()

    @pytest.mark.slow
    def test_reading_advi_cache(
        self, advi_results: pmapi.ApproximationSamplingResults, tmp_path: Path
    ):
        model = speclet_model.SpecletModel(cache_dir=tmp_path)
        assert len(list(tmp_path.iterdir())) == 0
        model.cache_sampling_results(advi_results)
        assert len(list(tmp_path.iterdir())) > 0
        new_res = model.read_cached_approximation()

        for param in ["a", "b", "sigma"]:
            np.testing.assert_almost_equal(
                advi_results.prior_predictive[param],
                new_res.prior_predictive[param],
                decimal=5,
            )
            assert pytest.approx(
                np.mean(advi_results.trace[param]), abs=0.1
            ) == np.mean(new_res.trace[param])
            assert pytest.approx(np.std(advi_results.trace[param]), abs=0.1) == np.std(
                new_res.trace[param]
            )

        np.testing.assert_almost_equal(
            advi_results.prior_predictive["y"], new_res.prior_predictive["y"], decimal=5
        )
        np.testing.assert_almost_equal(
            advi_results.posterior_predictive["y"],
            new_res.posterior_predictive["y"],
            decimal=5,
        )
        np.testing.assert_almost_equal(
            advi_results.approximation.hist, new_res.approximation.hist, decimal=3
        )

    @pytest.mark.slow
    def test_convert_to_arviz(
        self, pm_model: pm.Model, mcmc_results: pmapi.MCMCSamplingResults
    ):
        az_obj = pmapi.convert_samples_to_arviz(model=pm_model, res=mcmc_results)
        assert isinstance(az_obj, az.InferenceData)
