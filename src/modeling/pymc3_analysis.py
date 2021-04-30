#!/usr/bin/env python3

"""Functions to aid in the analysis of PyMC3 models."""

import re
from typing import Dict, List, Optional, Tuple

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns


def plot_all_priors(
    prior_predictive: Dict[str, np.ndarray],
    subplots: Tuple[int, int],
    figsize: Tuple[float, float],
    samples: int = 1000,
    rm_var_regex: str = "log__|logodds_",
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
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


def plot_vi_hist(approx: pm.variational.Approximation) -> gg.ggplot:
    """Plot the history of fitting using Variational Inference.

    Args:
        approx (pm.variational.Approximation): The approximation attribute from the
          VI object.

    Returns:
        gg.ggplot: A plot showing the fitting history.
    """
    d = pd.DataFrame({"loss": approx.hist}).assign(step=lambda d: np.arange(d.shape[0]))
    return (
        gg.ggplot(d, gg.aes(x="step", y="loss"))
        + gg.geom_line(size=0.5, alpha=0.75, color="black")
        + gg.scale_y_continuous(expand=(0.02, 0, 0.02, 0))
        + gg.scale_x_continuous(expand=(0.02, 0, 0.02, 0))
        + gg.labs(x="step", y="loss", title="Approximation history")
    )
