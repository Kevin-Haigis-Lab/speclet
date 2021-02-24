#!/bin/env python3

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from analysis import pymc3_sampling_api as pmsample

#### ---- MCMC Sampling ---- ####


class TestPyMC3SamplingAPI:

    alpha = 1
    sigma = 1
    beta = [1, 2.5]

    def setup(self):
        # self.alpha = 1
        # self.sigma = 1
        # self.beta = [1, 2.5]
        return

    def teardown(self):
        return

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        np.random.seed(1027)
        size = 1000
        x1 = np.random.randn(size)
        x2 = np.random.randn(size) * 0.2

        y = (
            self.alpha
            + self.beta[0] * x1
            + self.beta[1] * x2
            + (np.random.randn(size) * self.sigma)
        )

        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    @pytest.fixture
    def model(self, data: pd.DataFrame) -> pm.Model:
        # y = alpha + beta_1 * x1 + beta_2 * x2 + noise
        with pm.Model() as m:
            alpha = pm.Normal("alpha", 0, 10)
            beta = pm.Normal("beta", 0, 10, shape=2)
            sigma = pm.HalfNormal("sigma", 1)
            mu = alpha + beta[0] * data["x1"].values + beta[1] * data["x2"].values
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data["y"].values)

        return m

    def test_can_mock_model(self, model: pm.Model):
        assert isinstance(model, pm.Model)


#### ---- MCMC ---- ####


class TestMCMCSampling(TestPyMC3SamplingAPI):
    @pytest.mark.slow
    def test_can_sample_model(self, model: pm.Model):
        sampling_res = pmsample.pymc3_sampling_procedure(
            model=model,
            num_mcmc=100,
            tune=100,
            prior_check_samples=100,
            ppc_samples=100,
            chains=2,
            cores=2,
            random_seed=123,
            cache_dir=None,
        )

        expect_keys = ["trace", "posterior_predictive", "prior_predictive"]
        for key in expect_keys:
            assert key in sampling_res.keys()

    @pytest.mark.slow
    def test_correctly_sampling_model(self, model: pm.Model):
        sampling_res = pmsample.pymc3_sampling_procedure(
            model=model,
            num_mcmc=1000,
            tune=1000,
            chains=2,
            cores=2,
            random_seed=123,
            cache_dir=None,
        )

        m_az = pmsample.samples_to_arviz(model, sampling_res)
        assert isinstance(m_az, az.InferenceData)

        trace = sampling_res["trace"]

        assert isinstance(trace, pm.backends.base.MultiTrace)
        for known_val, param in zip(
            [self.alpha, self.beta, self.sigma], ["alpha", "beta", "sigma"]
        ):
            np.testing.assert_almost_equal(
                trace[param].mean(axis=0), known_val, decimal=1
            )

    def test_cache_results(self, data: pd.DataFrame, model: pm.Model, tmp_path: Path):

        cache_dir = tmp_path / "pymc3_cache"

        original_res = pmsample.pymc3_sampling_procedure(
            model=model,
            num_mcmc=100,
            tune=100,
            chains=2,
            cores=2,
            random_seed=123,
            cache_dir=cache_dir,
            force=True,
        )

        cached_res = pmsample.read_cached_sampling(cache_dir, model=model)

        for param in ["alpha", "beta", "sigma"]:
            np.testing.assert_equal(
                original_res["trace"][param], cached_res["trace"][param]
            )


#### ---- ADVI ---- ####


class TestADVISampling(TestPyMC3SamplingAPI):
    @pytest.mark.slow
    def test_can_sample_model(self, model: pm.Model):
        sampling_res = pmsample.pymc3_advi_approximation_procedure(
            model=model,
            n_iterations=100,
            draws=100,
            prior_check_samples=100,
            post_check_samples=100,
            random_seed=123,
        )

        expect_keys = [
            "approximation",
            "trace",
            "posterior_predictive",
            "prior_predictive",
        ]
        for key in expect_keys:
            assert key in sampling_res.keys()

    @pytest.mark.slow
    def test_correctly_sampling_model(self, model: pm.Model):
        sampling_res = pmsample.pymc3_advi_approximation_procedure(
            model=model, callbacks=[pm.callbacks.CheckParametersConvergence()]
        )

        m_az = pmsample.samples_to_arviz(model, sampling_res)
        assert isinstance(m_az, az.InferenceData)

        assert isinstance(sampling_res["approximation"], pm.MeanField)

        trace = sampling_res["trace"]

        assert isinstance(trace, pm.backends.base.MultiTrace)
        for known_val, param in zip(
            [self.alpha, self.beta, self.sigma], ["alpha", "beta", "sigma"]
        ):
            np.testing.assert_almost_equal(
                trace[param].mean(axis=0), known_val, decimal=1
            )

    @pytest.mark.slow
    def test_correctly_sampling_fullrank(self, model: pm.Model):
        sampling_res = pmsample.pymc3_advi_approximation_procedure(
            model=model,
            method="fullrank_advi",
            callbacks=[pm.callbacks.CheckParametersConvergence()],
        )

        assert isinstance(sampling_res["approximation"], pm.FullRank)

    def test_cache_results(self, data: pd.DataFrame, model: pm.Model, tmp_path: Path):

        cache_dir = tmp_path / "pymc3_advi_cache"

        original_res = pmsample.pymc3_advi_approximation_procedure(
            model=model,
            callbacks=[pm.callbacks.CheckParametersConvergence()],
            random_seed=123,
            cache_dir=cache_dir,
            force=True,
        )

        cached_res = pmsample.read_cached_vi(cache_dir, draws=10000)

        np.testing.assert_equal(
            original_res["approximation"].hist, cached_res["approximation"].hist
        )

        for param in ["alpha", "beta", "sigma"]:
            np.testing.assert_almost_equal(
                original_res["trace"][param].mean(axis=0),
                cached_res["trace"][param].mean(axis=0),
                decimal=1,
            )
            np.testing.assert_almost_equal(
                original_res["trace"][param].std(axis=0),
                cached_res["trace"][param].std(axis=0),
                decimal=1,
            )
