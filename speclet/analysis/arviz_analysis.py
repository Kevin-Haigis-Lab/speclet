"""Functions to aid in the analysis of ArviZ posterior data."""

import datetime
from typing import Any, Sequence

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from xarray import Dataset


def extract_coords_param_names(
    post_summ: pd.DataFrame, names: list[str] | str, col: str | None = None
) -> pd.DataFrame:
    """Extract coordinates from parameter names (from ArviZ summary).

    Args:
        post_summ (pd.DataFrame): Posterior summary from ArviZ.
        names (list[str] | str): Names for the coordinates.
        col (Optional[str], optional): Column containing the parameter names. If `None`
        (default), uses the row index for parameter names.

    Returns:
        pd.DataFrame: Modified data frame.
    """
    if col is None:
        coord_col = post_summ.index.tolist()
    else:
        coord_col = post_summ[col]

    if isinstance(names, str):
        names = [names]

    coords = [x.split("[")[1] for x in coord_col]
    coords = [x.replace("]", "") for x in coords]
    coords_split = [[y.strip() for y in x.split(",")] for x in coords]
    coords_ary = np.asarray(coords_split)
    assert coords_ary.shape == (len(post_summ), len(names))
    for j, name in enumerate(names):
        post_summ[name] = coords_ary[:, j]
    return post_summ


def summarize_posterior_predictions(
    ppc_ary: np.ndarray,
    hdi_prob: float | None = None,
    merge_with: pd.DataFrame | None = None,
    calc_error: bool = False,
    observed_y: str | None = None,
) -> pd.DataFrame:
    """Summarizing PyMC3 PPCs.

    Args:
        ppc_ary (np.ndarray): The posterior predictions.
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
    ppc_hdi = az.hdi(ppc_ary, hdi_prob=hdi_prob)
    d = pd.DataFrame(
        {
            "pred_mean": ppc_ary.mean(axis=(0, 1)),
            "pred_hdi_low": ppc_hdi[:, 0],
            "pred_hdi_high": ppc_hdi[:, 1],
        }
    )

    if merge_with is not None:
        d = pd.merge(
            d, merge_with.reset_index(drop=True), left_index=True, right_index=True
        )
        if calc_error and observed_y is not None:
            if observed_y not in d.columns:
                raise TypeError(f"Column '{observed_y}' is not in data.")
            d["error"] = d[observed_y] - d["pred_mean"]

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


def _get_average_sample_stat(data: az.InferenceData, name: str) -> list[float] | None:
    sample_stat = data.get("sample_stats")
    if sample_stat is None:
        return None
    values = sample_stat.get(name)
    if values is None:
        return None
    return values.mean(dim="draw").values.tolist()


def get_average_step_size(data: az.InferenceData) -> list[float] | None:
    """Get the average step size for each chain of MCMC.

    Args:
        data (az.InferenceData): Data object.

    Returns:
        list[float]: list of average step sizes for each chain.
    """
    possible_names = ["step_size", "stepsize"]
    for stat_name in possible_names:
        if (res := _get_average_sample_stat(data, stat_name)) is not None:
            return res
    return None


def get_average_tree_depth(data: az.InferenceData) -> list[float] | None:
    """Get the average tree depth if available."""
    return _get_average_sample_stat(data, "tree_depth")


def get_average_acceptance_prob(data: az.InferenceData) -> list[float] | None:
    """Get the average acceptance probability if available."""
    return _get_average_sample_stat(data, "acceptance_rate")


def get_divergences(data: az.InferenceData) -> np.ndarray:
    """Get the divergence values of each MCMC chain.

    Args:
        data (az.InferenceData): Data object.

    Returns:
        np.ndarray: Divergence values.
    """
    return data["sample_stats"].diverging.values


def get_divergence_summary(data: az.InferenceData) -> tuple[list[int], list[float]]:
    """Get the number and percent of steps that were divergences of each MCMC chain.

    Args:
        data (az.InferenceData): Data object.

    Returns:
        tuple[list[int], list[float]]: A list of the number of divergent steps and a
        list of the percent of steps that were divergent.
    """
    divs = data["sample_stats"].diverging.values
    totals = divs.sum(axis=1)
    pct = divs.mean(axis=1) * 100
    return totals.tolist(), pct.tolist()


class MCMCDescription(BaseModel):
    """Descriptive information for a MCMC."""

    created: datetime.datetime | None
    duration: datetime.timedelta | None
    n_chains: int
    n_tuning_steps: int | None
    n_draws: int
    n_divergences: list[int]
    pct_divergences: list[float]
    bfmi: list[float]
    avg_step_size: list[float] | None
    avg_accept_prob: list[float] | None
    avg_tree_depth: list[float] | None

    def _pretty_list(self, vals: Sequence[int | float], round: int = 3) -> str:
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
        if self.avg_step_size is None:
            messages.append("avg. step size: unknown")
        else:
            messages.append(f"avg. step size: {self._pretty_list(self.avg_step_size)}")

        if self.avg_accept_prob is None:
            messages.append("avg. accept prob.: unknown")
        else:
            messages.append(
                f"avg. accept prob.: {self._pretty_list(self.avg_accept_prob)}"
            )

        if self.avg_tree_depth is None:
            messages.append("avg. tree depth: unknown")
        else:
            messages.append(
                f"avg. tree depth: {self._pretty_list(self.avg_tree_depth)}"
            )
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
    created_at = getattr(sample_stats, "created_at", None)
    duration: datetime.timedelta | None = None

    if (duration_sec := getattr(sample_stats, "sampling_time", None)) is not None:
        duration = datetime.timedelta(seconds=duration_sec)

    # Sampling dimensions
    n_tuning_steps: int | None = getattr(sample_stats, "tuning_steps", None)
    n_draws: int = len(sample_stats.draw)
    n_chains: int = len(sample_stats.chain)

    # Divergences
    n_divergences, pct_divergences = get_divergence_summary(data)

    mcmc_descr = MCMCDescription(
        created=created_at,
        duration=duration,
        n_tuning_steps=n_tuning_steps,
        n_chains=n_chains,
        n_draws=n_draws,
        n_divergences=n_divergences,
        pct_divergences=pct_divergences,
        bfmi=az.bfmi(data).tolist(),
        avg_step_size=get_average_step_size(data),
        avg_accept_prob=get_average_acceptance_prob(data),
        avg_tree_depth=get_average_tree_depth(data),
    )

    if not silent:
        print(mcmc_descr)

    if plot:
        az.plot_energy(data)
        plt.show()

    return mcmc_descr


def rhat_table(
    trace: az.InferenceData, rhat_kwargs: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Get a table of R hat values.

    Args:
        trace (az.InferenceData): MCMC trace.
        rhat_kwargs (Optional[dict[str, Any]], optional): Keyword arguments to pass to
        ArviZ `rhat()` function. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    rhat_kwargs = rhat_kwargs if rhat_kwargs is not None else {}
    rhat: Dataset = az.rhat(trace, **rhat_kwargs)
    rhat_df_list = []
    for dv in rhat.data_vars:
        df = pd.DataFrame({"rhat": rhat[dv].values.flatten(), "var_name": dv})
        rhat_df_list.append(df)
    return pd.concat(rhat_df_list).reset_index(drop=True)


def summarize_rhat(
    trace: az.InferenceData | None = None,
    rhat_tbl: pd.DataFrame | None = None,
    ncol: int = 4,
    binwidth: float = 0.01,
) -> pd.DataFrame:
    """Summarize R hat values.

    Plot a histogram of R hat values for each variable and provide a data frame
    summarizing the R hat values for each variable.

    Args:
        trace (Optional[az.InferenceData], optional): MCMC trace. Defaults to None.
        rhat_tbl (Optional[pd.DataFrame], optional): R hat table for a MCMC trace.
        Defaults to None.
        ncol (int, optional): Number of columns in the histogram of R hat values.
        Defaults to 4.
        binwidth (float, optional): Bin width for the histogram of R hat values.
        Defaults to 0.01.

    Raises:
        BaseException: If neither the trace nor R hat table are provided (are both
        None).

    Returns:
        pd.DataFrame: Data frame of summary statistics of R hat values for each
        parameter.
    """
    if rhat_tbl is None and trace is None:
        msg = "The trace or R hat table must be provided"
        raise BaseException(msg)
    elif rhat_tbl is None:
        assert trace is not None
        rhat_tbl = rhat_table(trace)

    fg = sns.displot(
        data=rhat_tbl,
        x="rhat",
        col="var_name",
        binwidth=binwidth,
        col_wrap=ncol,
        facet_kws={"sharey": False},
    )
    fg.set_titles(col_template="{col_name}")
    plt.show()

    rhat_summary = rhat_tbl.groupby("var_name")["rhat"].describe()
    return rhat_summary
