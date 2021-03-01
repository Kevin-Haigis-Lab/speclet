#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pymc3 as pm
import pytest
from numpy.testing import assert_equal

import analysis.custom_pymc3_callbacks as pymc3calls


class TestDivergenceFractionCallback:
    @pytest.fixture(scope="class")
    def mock_model(self) -> pm.Model:
        np.random.seed(1)
        a, b = 1, 2
        x = np.random.uniform(0, 10, 100)
        y = a + b * x + np.random.randn(len(x))

        with pm.Model() as m:
            alpha = pm.Normal("alpha", 0, 200)
            gamma = pm.Normal("gamma", 0, 200)  # non-identifiable
            beta = pm.Normal("beta", 0, 200)
            mu = alpha + beta * x + gamma
            sigma = pm.HalfNormal("sigma", 100)
            y_pred = pm.Normal("y_pred", mu, sigma, observed=y)

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
                    random_seed=123,
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

        assert trace["alpha"].shape[0] == 1000

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

        assert trace["alpha"].shape[0] == 1000
