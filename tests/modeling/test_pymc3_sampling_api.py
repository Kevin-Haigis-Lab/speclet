import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from src.modeling import pymc3_sampling_api as pmapi

#### ---- MCMC Sampling ---- ####


class TestPyMC3SamplingAPI:

    alpha = 1
    sigma = 1
    beta = [1, 2.5]

    @pytest.fixture(scope="class")
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

    @pytest.fixture(scope="class")
    def model(self, data: pd.DataFrame) -> pm.Model:
        # y = alpha + beta_1 * x1 + beta_2 * x2 + noise
        with pm.Model() as m:
            alpha = pm.Normal("alpha", 0, 10)
            beta = pm.Normal("beta", 0, 10, shape=2)
            sigma = pm.HalfNormal("sigma", 1)
            mu = alpha + beta[0] * data["x1"].values + beta[1] * data["x2"].values
            y_obs = pm.Normal(  # noqa: F841
                "y_obs", mu=mu, sigma=sigma, observed=data["y"].values
            )

        return m

    def test_can_mock_model(self, model: pm.Model) -> None:
        assert isinstance(model, pm.Model)


#### ---- MCMC ---- ####


class TestMCMCSampling(TestPyMC3SamplingAPI):
    @pytest.mark.slow
    def test_can_sample_model(self, model: pm.Model) -> None:
        sampling_res = pmapi.pymc3_sampling_procedure(
            model=model,
            mcmc_draws=100,
            tune=100,
            prior_pred_samples=100,
            chains=1,
            cores=2,
            random_seed=123,
        )

        assert isinstance(sampling_res, pmapi.MCMCSamplingResults)

    @pytest.mark.slow
    def test_correctly_sampling_model(self, model: pm.Model) -> None:
        sampling_res = pmapi.pymc3_sampling_procedure(
            model=model,
            mcmc_draws=1000,
            tune=1000,
            chains=1,
            cores=2,
            random_seed=123,
        )

        assert isinstance(sampling_res, pmapi.MCMCSamplingResults)
        m_az = pmapi.convert_samples_to_arviz(model, sampling_res)
        assert isinstance(m_az, az.InferenceData)

        trace = sampling_res.trace

        assert isinstance(trace, pm.backends.base.MultiTrace)
        for known_val, param in zip(
            [self.alpha, self.beta, self.sigma], ["alpha", "beta", "sigma"]
        ):
            np.testing.assert_almost_equal(
                trace[param].mean(axis=0), known_val, decimal=1
            )


#### ---- ADVI ---- ####


class TestADVISampling(TestPyMC3SamplingAPI):
    @pytest.mark.slow
    def test_can_sample_model(self, model: pm.Model) -> None:
        sampling_res = pmapi.pymc3_advi_approximation_procedure(
            model=model,
            n_iterations=100,
            draws=100,
            prior_pred_samples=100,
            random_seed=123,
        )
        assert isinstance(sampling_res, pmapi.ApproximationSamplingResults)

    @pytest.mark.slow
    def test_correctly_sampling_model(self, model: pm.Model) -> None:
        sampling_res = pmapi.pymc3_advi_approximation_procedure(
            model=model, callbacks=[pm.callbacks.CheckParametersConvergence()]
        )

        m_az = pmapi.convert_samples_to_arviz(model, sampling_res)
        assert isinstance(m_az, az.InferenceData)

        assert isinstance(sampling_res.approximation, pm.MeanField)

        trace = sampling_res.trace

        assert isinstance(trace, pm.backends.base.MultiTrace)
        for known_val, param in zip(
            [self.alpha, self.beta, self.sigma], ["alpha", "beta", "sigma"]
        ):
            np.testing.assert_almost_equal(
                trace[param].mean(axis=0), known_val, decimal=1
            )

    @pytest.mark.slow
    def test_correctly_sampling_fullrank(self, model: pm.Model) -> None:
        sampling_res = pmapi.pymc3_advi_approximation_procedure(
            model=model,
            method="fullrank_advi",
            callbacks=[pm.callbacks.CheckParametersConvergence()],
        )

        assert isinstance(sampling_res.approximation, pm.FullRank)
