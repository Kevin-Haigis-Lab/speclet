import matplotlib
import numpy as np
import pandas as pd
import plotnine as gg
import pymc as pm
import pytest
from numpy.random import standard_normal
from pymc.backends.base import MultiTrace

from speclet.analysis import pymc_analysis as pmanal

MCMCResults = tuple[MultiTrace, dict[str, np.ndarray]]
ADVIResults = tuple[MultiTrace, dict[str, np.ndarray], pm.Approximation]
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
        post_pred = pm.sample_posterior_predictive(trace, random_seed=0)
    return trace, post_pred, approx


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
