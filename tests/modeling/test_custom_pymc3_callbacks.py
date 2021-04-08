#!/usr/bin/env python3

import numpy as np
import pymc3 as pm
import pytest

from src.modeling import custom_pymc3_callbacks as pymc3calls


class TestDivergenceFractionCallback:
    @pytest.fixture(scope="class")
    def mock_model(self) -> pm.Model:
        np.random.seed(1)
        X = np.ones(5)
        y = X * 2 + np.random.randn(len(X))

        with pm.Model() as m:
            a = pm.Normal("a", 50, 100)
            b = pm.Normal("b", 50, 100)
            mu = a + b * X
            sigma = pm.HalfCauchy("error", 1)
            obs = pm.Normal("obs", mu, sigma, observed=y)  # noqa: F841

        return m

    @pytest.mark.DEV
    def test_divergences(self, mock_model: pm.Model):
        n_tune_steps = 200
        cb = pymc3calls.DivergenceFractionCallback(
            n_tune_steps=n_tune_steps, max_frac=0.01
        )
        with pytest.raises(pymc3calls.TooManyDivergences):
            with mock_model:
                _ = pm.sample(
                    draws=1000,
                    tune=n_tune_steps,
                    chains=1,
                    cores=1,
                    callback=cb,
                    random_seed=404,
                )

    def test_min_samples_param(self, mock_model: pm.Model):
        n_tune_steps = 200
        cb = pymc3calls.DivergenceFractionCallback(
            n_tune_steps=n_tune_steps, min_samples=10000
        )
        with mock_model:
            trace = pm.sample(
                draws=1000,
                tune=n_tune_steps,
                chains=1,
                cores=1,
                callback=cb,
                random_seed=123,
            )

        assert trace["a"].shape[0] == 1000

    def test_max_frac_param(self, mock_model: pm.Model):
        n_tune_steps = 200
        cb = pymc3calls.DivergenceFractionCallback(
            n_tune_steps=n_tune_steps, max_frac=1.1
        )
        with mock_model:
            trace = pm.sample(
                draws=1000,
                tune=n_tune_steps,
                chains=1,
                cores=1,
                callback=cb,
                random_seed=123,
            )

        assert trace["a"].shape[0] == 1000
