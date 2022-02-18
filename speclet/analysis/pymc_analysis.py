"""Functions to aid in the analysis of PyMC3 models."""

import re
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc as pm
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from speclet.plot.color_pal import SeabornColor


def plot_all_priors(
    prior_predictive: dict[str, np.ndarray],
    subplots: tuple[int, int],
    figsize: tuple[float, float],
    samples: int = 1000,
    rm_var_regex: str = "log__|logodds_",
) -> tuple[Figure, Axes]:
    """Plot all priors of a PyMC3 model.

    Args:
        prior_predictive (dict[str, np.ndarray]): The results of sampling from the
          priors of a PyMC3 model.
        subplots (tuple[int, int]): How many subplots to create.
        figsize (tuple[float, float]): The size of the final figure.
        samples (int, optional): The number of samples from the distributions to use.
          This can help the performance of the plotting if there are many samples.
          Defaults to 1000.
        rm_var_regex (str, optional): A regular expression for variables to remove.
          Defaults to "log__|logodds_".

    Returns:
        tuple[Figure, Axes]: The matplotlib figure and array
          of axes.
    """
    model_vars: list[str] = []
    for x in prior_predictive:
        if not re.search(rm_var_regex, x):
            model_vars.append(x)

    model_vars.sort()
    model_vars.sort(key=lambda x: -len(x))

    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    for var, ax in zip(model_vars, axes.flatten()):
        sns.kdeplot(x=np.random.choice(prior_predictive[var].flatten(), samples), ax=ax)
        ax.set_title(var)

    fig.tight_layout()
    return fig, axes


def _plot_vi_axes_limits(
    yvals: np.ndarray, x_start: Optional[Union[int, float]]
) -> tuple[list[int], list[float]]:
    yvals = yvals.copy()[np.isfinite(yvals)]
    x_lims: list[int] = [0, len(yvals)]
    if x_start is None:
        pass
    elif isinstance(x_start, float) and x_start <= 1.0:
        x_lims[0] = int(x_start * len(yvals))
    else:
        x_lims[0] = int(x_start)

    y_lims: list[float] = [min(yvals), max(yvals[x_lims[0] :])]

    return x_lims, y_lims


def _advi_hist_rolling_avg(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df["loss_rolling_avg"] = (
        df[["loss"]].rolling(window=window, center=True).mean()["loss"]
    )
    return df


def plot_vi_hist(
    approx: pm.Approximation,
    y_log: bool = False,
    x_start: Optional[Union[int, float]] = None,
    rolling_window: int = 100,
) -> gg.ggplot:
    """Plot the history of fitting using Variational Inference.

    Args:
        approx (pm.variational.Approximation): The approximation attribute from the
          VI object.

    Returns:
        gg.ggplot: A plot showing the fitting history.
    """
    y = "np.log(loss)" if y_log else "loss"
    rolling_y = "np.log(loss_rolling_avg)" if y_log else "loss_rolling_avg"
    d = (
        pd.DataFrame({"loss": approx.hist})
        .assign(step=lambda d: np.arange(d.shape[0]))
        .pipe(_advi_hist_rolling_avg, window=rolling_window)
    )

    _x_lims, _y_lims = _plot_vi_axes_limits(approx.hist, x_start)
    if y_log:
        _y_lims = np.log(_y_lims).tolist()

    return (
        gg.ggplot(d, gg.aes(x="step"))
        + gg.geom_line(gg.aes(y=y), size=0.5, alpha=0.75, color="black")
        + gg.geom_line(gg.aes(y=rolling_y), size=0.5, alpha=0.9, color=SeabornColor.RED)
        + gg.scale_x_continuous(expand=(0, 0), limits=_x_lims)
        + gg.scale_y_continuous(expand=(0.02, 0, 0.02, 0), limits=_y_lims)
        + gg.labs(
            x="step",
            y="$\\log$ loss" if y_log else "loss",
            title="Approximation history",
        )
    )


def down_sample_ppc(
    ppc_ary: np.ndarray, n: int, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Down sample a PPC (or any numpy) array.

    Args:
        ppc_ary (np.ndarray): PPC array.
        n (int): Number of samples to return.
        axis (int, optional): The axis corresponding to the data samples. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: A numpy array with `n` samples and an array of
        the indices sampled.
    """
    r_idx = np.arange(ppc_ary.shape[axis])
    np.random.shuffle(r_idx)
    return ppc_ary[:, r_idx[:n]], r_idx
