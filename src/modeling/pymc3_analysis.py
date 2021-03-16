#!/usr/bin/env python3

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

    model_vars: List[str] = []
    for x in prior_predictive.keys():
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
    indices_list = [
        [int(x) for x in re.findall("[0-9]+", s)] for s in d[[col]].to_numpy().flatten()
    ]
    indices_array = np.asarray(indices_list)
    d[idx1name] = idx1[indices_array[:, 0]]
    d[idx2name] = idx2[indices_array[:, 1]]
    return d


def summarize_posterior_predictions(
    a: np.ndarray, hdi_prob: float = 0.89, merge_with: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    hdi = az.hdi(a, hdi_prob=hdi_prob)

    d = pd.DataFrame(
        {
            "pred_mean": a.mean(axis=0),
            "pred_hdi_low": hdi[:, 0],
            "pred_hdi_high": hdi[:, 1],
        }
    )

    if not merge_with is None:
        d = pd.merge(
            d, merge_with.reset_index(drop=True), left_index=True, right_index=True
        )

    return d


def plot_vi_hist(approx: pm.variational.Approximation) -> gg.ggplot:
    d = pd.DataFrame({"loss": approx.hist}).assign(step=lambda d: np.arange(d.shape[0]))
    return (
        gg.ggplot(d, gg.aes(x="step", y="loss"))
        + gg.geom_line(size=0.5, alpha=0.75, color="black")
        + gg.scale_y_continuous(expand=(0.02, 0, 0.02, 0))
        + gg.scale_x_continuous(expand=(0.02, 0, 0.02, 0))
        + gg.labs(x="step", y="loss", title="Approximation history")
    )
