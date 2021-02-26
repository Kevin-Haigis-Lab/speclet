#!/bin/env python3

from typing import Dict, Tuple

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from analysis import pymc3_analysis as pmanal

MCMCResults = Tuple[pm.backends.base.MultiTrace, Dict[str, np.ndarray]]
ADVIResults = Tuple[
    pm.backends.base.MultiTrace, Dict[str, np.ndarray], pm.Approximation
]


class PyMC3AnalysisTesting:
    @pytest.fixture(scope="class")
    def mock_data(self) -> pd.DataFrame:
        self.a, self.b = 1, 2
        x = np.random.uniform(0, 100, 300)
        y = self.a + self.b * x + np.random.normal(0, 0.1, len(x))
        return pd.DataFrame({"x": x, "y": y})

    @pytest.fixture(scope="class")
    def mock_model(self, mock_data: pd.DataFrame) -> pm.Model:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", self.a, 2)
            beta = pm.Normal("beta", self.b, 2)
            mu = alpha + beta * mock_data.x.values
            sigma = pm.HalfNormal("sigma", 2)
            y_pred = pm.Normal("y_pred", mu, sigma, observed=mock_data.y.values)

        return model


class TestSummarizePosteriorPredictions(PyMC3AnalysisTesting):
    @pytest.fixture(scope="class")
    def mock_mcmc(self, mock_model: pm.Model) -> MCMCResults:
        with mock_model:
            trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2, random_seed=0)
            post_pred = pm.sample_posterior_predictive(
                trace, samples=500, random_seed=0
            )
        return trace, post_pred

    @pytest.fixture(scope="class")
    def mock_advi(self, mock_model: pm.Model) -> ADVIResults:
        with mock_model:
            approx = pm.fit(
                100000, callbacks=[pm.callbacks.CheckParametersConvergence()]
            )
            trace = approx.sample(1000)
            post_pred = pm.sample_posterior_predictive(
                trace, samples=500, random_seed=0
            )
        return trace, post_pred, approx

    def test_columnnames(self, mock_mcmc: MCMCResults):
        _, post_pred = mock_mcmc
        ppc_df = pmanal.summarize_posterior_predictions(post_pred["y_pred"])
        expected_columns = ["pred_mean", "pred_hdi_low", "pred_hdi_high"]
        for col in expected_columns:
            assert col in ppc_df.columns

    def test_hdi_parameter(self, mock_mcmc: MCMCResults):
        _, post_pred = mock_mcmc
        ppc_df_low = pmanal.summarize_posterior_predictions(
            post_pred["y_pred"], hdi_prob=0.5
        )
        ppc_df_high = pmanal.summarize_posterior_predictions(
            post_pred["y_pred"], hdi_prob=0.99
        )

        np.testing.assert_array_less(
            ppc_df_high["pred_hdi_low"], ppc_df_low["pred_hdi_low"]
        )
        np.testing.assert_array_less(
            ppc_df_low["pred_hdi_high"], ppc_df_high["pred_hdi_high"]
        )

    def test_data_merging(self, mock_mcmc: MCMCResults, mock_data: pd.DataFrame):
        _, post_pred = mock_mcmc
        ppc_df = pmanal.summarize_posterior_predictions(
            post_pred["y_pred"], merge_with=mock_data
        )
        expected_columns = ["x", "y", "pred_mean"]
        for col in expected_columns:
            assert col in ppc_df.columns

    def test_mcmc_ppc_accuracy(self, mock_mcmc: MCMCResults, mock_data: pd.DataFrame):
        _, post_pred = mock_mcmc
        ppc_df = pmanal.summarize_posterior_predictions(post_pred["y_pred"])
        assert ppc_df.shape[0] == mock_data.shape[0]
        np.testing.assert_allclose(
            mock_data.y.values, ppc_df["pred_mean"].values, atol=1
        )

    def test_advi_ppc_accuracy(self, mock_advi: ADVIResults, mock_data: pd.DataFrame):
        _, post_pred, _ = mock_advi
        ppc_df = pmanal.summarize_posterior_predictions(post_pred["y_pred"])
        assert ppc_df.shape[0] == mock_data.shape[0]
        np.testing.assert_allclose(
            mock_data.y.values, ppc_df["pred_mean"].values, atol=1
        )
