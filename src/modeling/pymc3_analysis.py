#!/usr/bin/env python3

"""Functions to aid in the analysis of PyMC3 models."""

import re
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns

from src.plot.color_pal import SeabornColor


def plot_all_priors(
    prior_predictive: Dict[str, np.ndarray],
    subplots: Tuple[int, int],
    figsize: Tuple[float, float],
    samples: int = 1000,
    rm_var_regex: str = "log__|logodds_",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot all priors of a PyMC3 model.

    Args:
        prior_predictive (Dict[str, np.ndarray]): The results of sampling from the
          priors of a PyMC3 model.
        subplots (Tuple[int, int]): How many subplots to create.
        figsize (Tuple[float, float]): The size of the final figure.
        samples (int, optional): The number of samples from the distributions to use.
          This can help the performance of the plotting if there are many samples.
          Defaults to 1000.
        rm_var_regex (str, optional): A regular expression for variables to remove.
          Defaults to "log__|logodds_".

    Returns:
        Tuple[matplotlib.figure.Figure, np.ndarray]: The matplotlib figure and array
          of axes.
    """
    model_vars: List[str] = []
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


def extract_matrix_variable_indices(
    d: pd.DataFrame,
    col: str,
    idx1: np.ndarray,
    idx2: np.ndarray,
    idx1name: str,
    idx2name: str,
) -> pd.DataFrame:
    """Extract and annotating matrix (2D only) indices.

    Args:
        d (pd.DataFrame): The data frame produced by summarizing the posteriors of a
          PyMC3 model.
        col (str): The column with the 2D indices.
        idx1 (np.ndarray): The values to use for the first index.
        idx2 (np.ndarray): The values to use for the second index.
        idx1name (str): The name to give the first index.
        idx2name (str): The name to give the second index.

    Returns:
        pd.DataFrame: The modified data frame.
    """
    indices_list = [
        [int(x) for x in re.findall("[0-9]+", s)] for s in d[[col]].to_numpy().flatten()
    ]
    indices_array = np.asarray(indices_list)
    d[idx1name] = idx1[indices_array[:, 0]]
    d[idx2name] = idx2[indices_array[:, 1]]
    return d


def _reshape_mcmc_chains_to_2d(a: np.ndarray) -> np.ndarray:
    z = a.shape[2]
    return a.reshape(1, -1, z).squeeze()


def summarize_posterior_predictions(
    a: np.ndarray,
    hdi_prob: float = 0.89,
    merge_with: Optional[pd.DataFrame] = None,
    calc_error: bool = False,
    observed_y: Optional[str] = None,
) -> pd.DataFrame:
    """Summarizing PyMC3 PPCs.

    Args:
        a (np.ndarray): The posterior predictions.
        hdi_prob (float, optional): The HDI probability to use. Defaults to 0.89.
        merge_with (Optional[pd.DataFrame], optional): The original data to merge with
          the predictions. Defaults to None.
        calc_error (bool): Should the error (real - predicted) be calculated? This is
          only used if `merge_with` is not None. Default to false.
        observed_y: (Optional[str], optional): The column with the observed data. This
          is only used if `merge_with` is not None and `calc_error` is true. Default
          to None.

    Returns:
        pd.DataFrame: A data frame with one row per data point and columns describing
          the posterior predictions.
    """
    if len(a.shape) == 3:
        a = _reshape_mcmc_chains_to_2d(a)
    hdi = az.hdi(a, hdi_prob=hdi_prob)

    d = pd.DataFrame(
        {
            "pred_mean": a.mean(axis=0),
            "pred_hdi_low": hdi[:, 0],
            "pred_hdi_high": hdi[:, 1],
        }
    )

    if merge_with is not None:
        d = pd.merge(
            d, merge_with.reset_index(drop=True), left_index=True, right_index=True
        )
        if calc_error and observed_y is not None:
            if observed_y not in d.columns:
                raise TypeError(f"Column '{observed_y}' is not in data.")
            d["error"] = d[observed_y].values - d["pred_mean"].values

    return d


def _plot_vi_axes_limits(
    yvals: np.ndarray, x_start: Optional[Union[int, float]]
) -> Tuple[List[int], List[float]]:
    x_lims: List[int] = [0, len(yvals)]
    if x_start is None:
        pass
    elif isinstance(x_start, float) and x_start <= 1.0:
        x_lims[0] = int(x_start * len(yvals))
    else:
        x_lims[0] = int(x_start)

    y_lims: List[float] = [min(yvals), max(yvals[x_lims[0] :])]

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
        _y_lims = np.log(_y_lims)

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


def get_hdi_colnames_from_az_summary(df: pd.DataFrame) -> Tuple[str, str]:
    """Get the column names corresponding to the HDI from an ArviZ summary.

    Args:
        df (pd.DataFrame): ArviZ posterior summary data frame.

    Returns:
        Tuple[str, str]: The two column names.
    """
    cols: List[str] = [c for c in df.columns if "hdi_" in c]
    cols = [c for c in cols if "%" in c]
    assert len(cols) == 2
    return cols[0], cols[1]
