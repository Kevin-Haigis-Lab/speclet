import re
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns


def plot_all_priors(
    prior_predictive: Dict[str, np.ndarray],
    subplots: Tuple[int, int],
    figsize: Tuple[float, float],
    samples: int = 1000,
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:

    model_vars: List[str] = []
    for x in prior_predictive.keys():
        if not "log__" in x:
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
