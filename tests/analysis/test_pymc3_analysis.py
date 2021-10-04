#!/usr/bin/env python3

from itertools import product

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import pytest
from numpy.random import standard_normal

from src.analysis import pymc3_analysis as pmanal

MCMCResults = tuple[pm.backends.base.MultiTrace, dict[str, np.ndarray]]
ADVIResults = tuple[
    pm.backends.base.MultiTrace, dict[str, np.ndarray], pm.Approximation
]
PriorPrediction = dict[str, np.ndarray]


@pytest.fixture(scope="module")
def mock_data() -> pd.DataFrame:
    x = np.random.uniform(0, 100, 300)
    y = 1 + 2 * x + np.random.normal(0, 0.1, len(x))
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture(scope="module")
def mock_model(mock_data: pd.DataFrame) -> pm.Model:
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 1, 2)
        beta = pm.Normal("beta", 2, 2)
        mu = alpha + beta * mock_data.x.values
        sigma = pm.HalfNormal("sigma", 2)
        y_pred = pm.Normal(  # noqa: F841
            "y_pred", mu, sigma, observed=mock_data.y.values
        )
    return model


@pytest.fixture(scope="module")
def mock_advi(mock_model: pm.Model) -> ADVIResults:
    with mock_model:
        approx = pm.fit(1000, callbacks=[pm.callbacks.CheckParametersConvergence()])
        trace = approx.sample(100)
        post_pred = pm.sample_posterior_predictive(trace, samples=50, random_seed=0)
    return trace, post_pred, approx


class TestSummarizePosteriorPredictions:
    def test_columnnames(self) -> None:
        ppc_df = pmanal.summarize_posterior_predictions(standard_normal((100, 200)))
        expected_columns = ["pred_mean", "pred_hdi_low", "pred_hdi_high"]
        for col in expected_columns:
            assert col in ppc_df.columns

    def test_hdi_parameter(self, mock_data: pd.DataFrame) -> None:
        post_shape = (mock_data.shape[0], 100)
        ppc_df_low = pmanal.summarize_posterior_predictions(
            standard_normal(post_shape), hdi_prob=0.5
        )
        ppc_df_high = pmanal.summarize_posterior_predictions(
            standard_normal(post_shape), hdi_prob=0.99
        )

        np.testing.assert_array_less(
            ppc_df_high["pred_hdi_low"], ppc_df_low["pred_hdi_low"]
        )
        np.testing.assert_array_less(
            ppc_df_low["pred_hdi_high"], ppc_df_high["pred_hdi_high"]
        )

    def test_data_merging(self, mock_data: pd.DataFrame) -> None:
        ppc_df = pmanal.summarize_posterior_predictions(
            standard_normal((mock_data.shape[0], 100)), merge_with=mock_data
        )
        expected_columns = ["x", "y", "pred_mean"]
        for col in expected_columns:
            assert col in ppc_df.columns

    def test_calc_error(self, mock_data: pd.DataFrame) -> None:
        y_pred = standard_normal((mock_data.shape[0], 100))
        ppc_df = pmanal.summarize_posterior_predictions(y_pred)
        assert "error" not in ppc_df.columns

        ppc_df = pmanal.summarize_posterior_predictions(y_pred, calc_error=True)
        assert "error" not in ppc_df.columns

        ppc_df = pmanal.summarize_posterior_predictions(
            y_pred, calc_error=True, observed_y="y_pred"
        )
        assert "error" not in ppc_df.columns

        ppc_df = pmanal.summarize_posterior_predictions(y_pred, merge_with=mock_data)
        assert "error" not in ppc_df.columns

        ppc_df = pmanal.summarize_posterior_predictions(
            y_pred, merge_with=mock_data, calc_error=True
        )
        assert "error" not in ppc_df.columns

        with pytest.raises(TypeError):
            ppc_df = pmanal.summarize_posterior_predictions(
                y_pred,
                merge_with=mock_data,
                calc_error=True,
                observed_y="Not a real column",
            )

        ppc_df = pmanal.summarize_posterior_predictions(
            y_pred,
            merge_with=mock_data,
            calc_error=True,
            observed_y="y",
        )
        assert "error" in ppc_df.columns


@pytest.mark.plots
class TestPlotVIHistory:
    @pytest.mark.parametrize("y_log", (True, False))
    def test_returns_plot(self, mock_advi: ADVIResults, y_log: bool) -> None:
        _, _, approx = mock_advi
        hist_plot = pmanal.plot_vi_hist(approx, y_log=y_log)
        assert isinstance(hist_plot, gg.ggplot)

    @pytest.mark.parametrize("y_log", (True, False))
    def test_plot_data_has_correct_columns(
        self, mock_advi: ADVIResults, y_log: bool
    ) -> None:
        _, _, approx = mock_advi
        hist_plot = pmanal.plot_vi_hist(approx, y_log=y_log)
        for colname in ["loss", "step"]:
            assert colname in hist_plot.data.columns


def test_extract_matrix_variable_indices() -> None:

    n_i = 3
    n_j = 4
    i = list(range(n_i))
    i_groups = np.array([f"i_{x}" for x in i])
    j = list(range(n_j))
    j_groups = np.array([f"j_{x}" for x in j])
    var = [f"[{i},{j}]" for i, j in product(i, j)]
    post_summary = pd.DataFrame({"index": var})

    summary = pmanal.extract_matrix_variable_indices(
        post_summary,
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


@pytest.mark.plots
class TestPlottingOfPriors:

    draws = 500

    @pytest.fixture(scope="class")
    def prior_pred(self, mock_data: pd.DataFrame) -> PriorPrediction:
        return {
            "alpha": standard_normal(self.draws),
            "beta": standard_normal(self.draws),
            "y_pred": standard_normal((mock_data.shape[0], self.draws)),
        }

    def test_returns_fig_and_axes(self, prior_pred: PriorPrediction) -> None:
        axes_shape = (2, 2)
        fig, axes = pmanal.plot_all_priors(prior_pred, axes_shape, (5, 5), samples=100)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == axes_shape


def test_get_hdi_colnames_from_az_summary(centered_eight_post: pd.DataFrame) -> None:
    hdi_cols = pmanal.get_hdi_colnames_from_az_summary(centered_eight_post)
    assert hdi_cols == ("hdi_3%", "hdi_97%")


@pytest.mark.parametrize(
    "az_obj_name", ["centered_eight", "non_centered_eight", "radon", "rugby"]
)
def test_describe_mcmc(az_obj_name: str) -> None:
    az_obj = az.load_arviz_data(az_obj_name)
    assert isinstance(az_obj, az.InferenceData)
    mcmc_desc = pmanal.describe_mcmc(az_obj, plot=False)
    assert mcmc_desc is not None
