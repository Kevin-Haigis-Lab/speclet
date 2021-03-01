#!/bin/env python3

from string import ascii_lowercase, ascii_uppercase
from typing import Dict, Tuple

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import pytest

from analysis import pymc3_analysis as pmanal
from analysis.common_data_processing import get_indices, get_indices_and_count

MCMCResults = Tuple[pm.backends.base.MultiTrace, Dict[str, np.ndarray]]
ADVIResults = Tuple[
    pm.backends.base.MultiTrace, Dict[str, np.ndarray], pm.Approximation
]
PriorPrediction = Dict[str, np.ndarray]


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


class TestSummarizePosteriorPredictions(PyMC3AnalysisTesting):
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


class TestPlotVIHistory(PyMC3AnalysisTesting):
    def test_returns_plot(self, mock_advi: ADVIResults):
        _, _, approx = mock_advi
        hist_plot = pmanal.plot_vi_hist(approx)
        assert isinstance(hist_plot, gg.ggplot)

    def test_plot_data_has_correct_columns(self, mock_advi: ADVIResults):
        _, _, approx = mock_advi
        hist_plot = pmanal.plot_vi_hist(approx)
        for colname in ["loss", "step"]:
            assert colname in hist_plot.data.columns


class TestExtractMatrixVariableIndices:
    @pytest.fixture(scope="class")
    def mock_data(self) -> pd.DataFrame:
        np.random.seed(0)
        n_measures = 10
        i_groups = list(ascii_lowercase[:5])
        j_groups = list(ascii_uppercase[:3])
        i = np.repeat(i_groups, n_measures * len(j_groups))
        j = np.tile(j_groups, n_measures * len(i_groups))

        i_effect = np.random.randn(len(i_groups))
        j_effect = np.random.randn(len(j_groups))

        d = pd.DataFrame({"i": i, "j": j}, dtype="category")
        d["y"] = (
            i_effect[get_indices(d, "i")]
            + j_effect[get_indices(d, "j")]
            + np.random.normal(0, 0.1)
        )
        return d

    @pytest.fixture(scope="class")
    def mock_model(self, mock_data: pd.DataFrame) -> pm.Model:

        i_idx, num_i = get_indices_and_count(mock_data, "i")
        j_idx, num_j = get_indices_and_count(mock_data, "j")

        with pm.Model() as model:
            mu_a = pm.Normal("mu_a", 0, 1)
            sigma_a = pm.HalfNormal("sigma_a", 1)
            a = pm.Normal("a", mu_a, sigma_a, shape=(num_i, num_j))
            mu = a[i_idx, j_idx]
            sigma = pm.HalfNormal("sigma", 1)
            y = pm.Normal("y", mu, sigma, observed=mock_data.y.values)
        return model

    def test_extract_matrix_variable_indices(
        self, mock_model: pm.Model, mock_data: pd.DataFrame
    ):
        with mock_model:
            trace = pm.sample(100, tune=100, chains=2, cores=2, random_seed=123)

        i_groups = mock_data["i"].values.unique()
        j_groups = mock_data["j"].values.unique()

        model_az = az.from_pymc3(trace=trace, model=mock_model)
        summary = az.summary(model_az, var_names="a").reset_index()
        summary = pmanal.extract_matrix_variable_indices(
            summary,
            col="index",
            idx1=i_groups,
            idx2=j_groups,
            idx1name="i",
            idx2name="j",
        )

        np.testing.assert_equal(
            summary["i"].values.astype(str),
            np.repeat(i_groups, int(len(summary) / len(i_groups))).astype(str),
        )

        np.testing.assert_equal(
            summary["j"].values.astype(str),
            np.tile(j_groups, int(len(summary) / len(j_groups))).astype(str),
        )


class TestPlottingOfPriors(PyMC3AnalysisTesting):

    draws = 500

    @pytest.fixture(scope="class")
    def prior_pred(self, mock_model: pm.Model) -> PriorPrediction:
        with mock_model:
            prior_pred = pm.sample_prior_predictive(self.draws, random_seed=123)
        return prior_pred

    def test_prior_predictive_shape(
        self, prior_pred: PriorPrediction, mock_data: pd.DataFrame
    ):
        for k in ["alpha", "beta"]:
            assert len(prior_pred[k]) == self.draws
        assert prior_pred["y_pred"].shape == (self.draws, mock_data.shape[0])

    def test_returns_fig_and_axes(self, prior_pred: PriorPrediction):
        axes_shape = (2, 2)
        fig, axes = pmanal.plot_all_priors(prior_pred, axes_shape, (5, 5), samples=100)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == axes_shape

    def test_plots_correct_data(self, prior_pred: PriorPrediction):
        axes_shape = (2, 2)
        n_samples = 100
        _, axes = pmanal.plot_all_priors(
            prior_pred, axes_shape, (5, 5), samples=n_samples
        )

        for ax in axes.flatten():
            assert ax.lines[0].get_xdata().shape[0] == n_samples * 2
            assert ax.lines[0].get_ydata().shape[0] == n_samples * 2

    def test_subplots_have_titles(self, prior_pred: PriorPrediction):
        _, axes = pmanal.plot_all_priors(prior_pred, (2, 2), (5, 5), samples=100)
        axes_titles = [ax.get_title() for ax in axes.flatten()]
        for ax_title in axes_titles:
            assert ax_title in prior_pred.keys()
