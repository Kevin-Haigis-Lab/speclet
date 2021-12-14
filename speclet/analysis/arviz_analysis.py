"""Functions to aid in the analysis of ArviZ posterior data."""

import datetime
import re
from typing import Optional, Sequence, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from xarray import Dataset


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


def get_hdi_colnames_from_az_summary(df: pd.DataFrame) -> tuple[str, str]:
    """Get the column names corresponding to the HDI from an ArviZ summary.

    Args:
        df (pd.DataFrame): ArviZ posterior summary data frame.

    Returns:
        tuple[str, str]: The two column names.
    """
    cols: list[str] = [c for c in df.columns if "hdi_" in c]
    cols = [c for c in cols if "%" in c]
    assert len(cols) == 2
    return cols[0], cols[1]


def _pretty_bfmi(data: az.InferenceData, decimals: int = 3) -> list[str]:
    return np.round(az.bfmi(data), decimals).astype(str).tolist()


def get_average_step_size(data: az.InferenceData) -> list[float]:
    """Get the average step size for each chain of MCMC.

    Args:
        data (az.InferenceData): Data object.

    Returns:
        list[float]: list of average step sizes for each chain.
    """
    return data.sample_stats.step_size.mean(axis=1).values.tolist()


def _pretty_step_size(data: az.InferenceData, decimals: int = 3) -> list[str]:
    return np.round(get_average_step_size(data), decimals).astype(str).tolist()


def get_divergences(data: az.InferenceData) -> np.ndarray:
    """Get the number and percent of steps that were divergences of each MCMC chain.

    Args:
        data (az.InferenceData): Data object.

    Returns:
        tuple[list[int], list[float]]: A list of the number of divergent steps and a
        list of the percent of steps that were divergent.
    """
    return data.sample_stats.diverging.values


def get_divergence_summary(data: az.InferenceData) -> tuple[list[int], list[float]]:
    """Get the number and percent of steps that were divergences of each MCMC chain.

    Args:
        data (az.InferenceData): Data object.

    Returns:
        tuple[list[int], list[float]]: A list of the number of divergent steps and a
        list of the percent of steps that were divergent.
    """
    divs = data.sample_stats.diverging.values
    totals = divs.sum(axis=1)
    pct = divs.mean(axis=1) * 100
    return totals.tolist(), pct.tolist()


class MCMCDescription(BaseModel):
    """Descriptive information for a MCMC."""

    created: Optional[datetime.datetime]
    duration: Optional[datetime.timedelta]
    n_chains: int
    n_tuning_steps: Optional[int]
    n_draws: int
    n_divergences: list[int]
    pct_divergences: list[float]
    bfmi: list[float]
    avg_step_size: list[float]

    def _pretty_list(self, vals: Sequence[Union[int, float]], round: int = 3) -> str:
        return ", ".join(np.round(vals, round).astype(str).tolist())

    def __str__(self) -> str:
        """Nifty ol' string."""
        messages: list[str] = []
        if self.created is not None:
            messages.append(f"date created: {self.created:%Y-%m-%d %H:%M}")
        if self.duration is not None:
            _d_min = self.duration / datetime.timedelta(minutes=1)
            messages.append(f"time required: {_d_min:0.2f} minutes")
        _n_tuning_steps = (
            f"{self.n_tuning_steps:,}"
            if (self.n_tuning_steps is not None)
            else "(unknown)"
        )
        messages.append(
            f"sampled {self.n_chains} chains with {_n_tuning_steps} "
            + f"tuning steps and {self.n_draws:,} draws"
        )
        messages.append(f"num. divergences: {self._pretty_list(self.n_divergences)}")
        messages.append(
            f"percent divergences: {self._pretty_list(self.pct_divergences)}"
        )
        messages.append(f"BFMI: {self._pretty_list(self.bfmi)}")
        messages.append(f"avg. step size: {self._pretty_list(self.avg_step_size)}")
        return "\n".join(messages)


def describe_mcmc(
    data: az.InferenceData, silent: bool = False, plot: bool = True
) -> MCMCDescription:
    """Descriptive statistics and plots for MCMC.

    Prints out the following:

    1. Date of creation and how long the sampling took. ***
    2. The number of tuning and sampling steps. ***
    3. BFMI of each chain.
    4. Average step size of each chain.
    5. Number of divergences in each chain.
    6. Plot the energy transition distribution and marginal energy distribution.

    Args:
        data (az.InferenceData): Data object.
        silent (bool, optional): Silence the printing of the description? Defaults to
          False.
        plot (bool, optional): Include any plots? Default is True.
    """
    if not hasattr(data, "sample_stats"):
        print("Unable to get sampling stats.")
        raise AttributeError("Input data does not have a `sample_stats` attribute.")

    sample_stats = data.get("sample_stats")
    if not isinstance(sample_stats, Dataset):
        raise AttributeError("`sample_stats` attribute is not of type `xarray.Dataset`")

    # Date and duration.
    created_at = sample_stats.get("created_at")
    duration: Optional[datetime.timedelta] = None
    if (duration_sec := sample_stats.get("sampling_time")) is not None:
        duration = datetime.timedelta(seconds=duration_sec)

    # Sampling dimensions
    n_tuning_steps: Optional[int] = sample_stats.get("tuning_steps")
    n_draws: int = len(sample_stats.draw)
    n_chains: int = len(sample_stats.chain)

    # Divergences
    n_divergences, pct_divergences = get_divergence_summary(data)

    # BFMI.
    bfmi = az.bfmi(data).tolist()

    # Average step size.
    avg_step_size = get_average_step_size(data)

    mcmc_descr = MCMCDescription(
        created=created_at,
        duration=duration,
        n_tuning_steps=n_tuning_steps,
        n_chains=n_chains,
        n_draws=n_draws,
        n_divergences=n_divergences,
        pct_divergences=pct_divergences,
        bfmi=bfmi,
        avg_step_size=avg_step_size,
    )

    if not silent:
        print(mcmc_descr)

    if plot:
        az.plot_energy(data)
        plt.show()

    return mcmc_descr
