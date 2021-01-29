from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
