"""Analyze simulation-based calibration results."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

import src.exceptions
import src.modeling.simulation_based_calibration_helpers as sbc
from src.modeling import pymc3_helpers as pmhelp

SBC_UNIFORMITY_THINNING_DRAWS = 100


class SBCAnalysis:
    """Analysis of SBC results."""

    root_dir: Path
    pattern: str
    n_simulations: Optional[int]

    simulation_results: Optional[list[sbc.SBCResults]] = None
    accuracy_test_results: Optional[pd.DataFrame] = None
    uniformity_test_results: Optional[pd.DataFrame] = None

    def __init__(
        self, root_dir: Path, pattern: str, n_simulations: Optional[int] = None
    ) -> None:
        """Create a `SBCAnalysis` object.

        Args:
            root_dir (Path): Path to the directory containing the results of all of
              the simulations.
            pattern (str): Pattern used for naming the simulations.
            n_simulations (Optional[int], optional): Number of simulations expected. If
              supplied, this number will be used to check the root dir for all of the
              results. Defaults to None.
        """
        self.root_dir = root_dir
        self.pattern = pattern
        self.n_simulations = n_simulations

    def _check_n_sims(self, ls: Sequence[Any]) -> None:
        if self.n_simulations is not None and self.n_simulations != len(ls):
            raise src.exceptions.IncorrectNumberOfFilesFoundError(
                expected=self.n_simulations, found=len(ls)
            )

    def get_simulation_directories(self) -> list[Path]:
        """Get the directories of all of the simulation results.

        Returns:
            list[Path]: List of paths.
        """
        return [p for p in self.root_dir.iterdir() if self.pattern in p.name]

    def get_simulation_file_managers(self) -> list[sbc.SBCFileManager]:
        """Get the file managers for each simulation.

        Returns:
            list[sbc.SBCFileManager]: List of SBC file managers.
        """
        return [sbc.SBCFileManager(p) for p in self.get_simulation_directories()]

    def get_simulation_results(
        self, multithreaded: bool = True
    ) -> list[sbc.SBCResults]:
        """Get all of the simulation results.

        Args:
            multithreaded (bool, optional): Should the results be collected using
              multiple threads? Defaults to True.

        Raises:
            src.exceptions.CacheDoesNotExistError: Raised if the results do not exist
            for a simulation.

        Returns:
            list[sbc.SBCResults]: List of all simulation results.
        """
        fms = self.get_simulation_file_managers()
        if multithreaded:
            results = self._get_simulation_results_multithreaded(fms)
        else:
            results = self._get_simulation_results_singlethreaded(fms)
        self._check_n_sims(results)
        self.simulation_results = results
        return results

    @staticmethod
    def _get_single_simulation_result(fm: sbc.SBCFileManager) -> sbc.SBCResults:
        if fm.all_data_exists():
            return fm.get_sbc_results()
        else:
            raise src.exceptions.CacheDoesNotExistError(fm.dir)

    def _get_simulation_results_singlethreaded(
        self, fms: list[sbc.SBCFileManager]
    ) -> list[sbc.SBCResults]:
        results = [SBCAnalysis._get_single_simulation_result(fm) for fm in fms]
        return results

    def _get_simulation_results_multithreaded(
        self, fms: list[sbc.SBCFileManager]
    ) -> list[sbc.SBCResults]:
        with ThreadPoolExecutor() as executor:
            results_iter = executor.map(SBCAnalysis._get_single_simulation_result, fms)

        return list(results_iter)

    def posterior_accuracy(
        self,
        simulation_posteriors_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Get the accuracy of the simulation results.

        Args:
          simulation_posteriors_df (Optional[pd.DataFrame], optional): Data frame of
            the summaries of the posterior summaries. If None is provided, then the data
            frame will be made first. Defaults to None.

        Returns:
            pd.DataFrame: Data frame of the accuracy of each parameter in the model.
        """
        if simulation_posteriors_df is None:
            simulation_posteriors_df = sbc.collate_sbc_posteriors(
                posterior_dirs=self.get_simulation_directories(),
                num_permutations=self.n_simulations,
            )

        self.accuracy_test_results = (
            simulation_posteriors_df.copy()
            .groupby(["parameter_name"])["within_hdi"]
            .mean()
            .reset_index(drop=False)
            .sort_values("within_hdi", ascending=False)
            .reset_index(drop=True)
        )
        return self.accuracy_test_results

    def uniformity_test(
        self, k_draws: int = SBC_UNIFORMITY_THINNING_DRAWS, multithreaded: bool = True
    ) -> pd.DataFrame:
        """Perform the SBC uniformity analysis.

        Args:
            k_draws (int, optional): Number of draws to thin the posterior samples down
              to. Defaults to 100.
            multithreaded (bool, optional): Should the data processing use multiple
              threads? Defaults to True.

        Returns:
            pd.DataFrame: A data frame of the rank statistic of each variable in each
            simulation.
        """
        sbc_file_managers = self.get_simulation_file_managers()
        self._check_n_sims(sbc_file_managers)

        def _calc_rank_stat(sbc_fm: sbc.SBCFileManager) -> pd.DataFrame:
            return calculate_parameter_rank_statistic(
                sbc_fm.get_sbc_results(), thin_to=k_draws
            )

        if multithreaded:
            with ThreadPoolExecutor() as executor:
                results = executor.map(_calc_rank_stat, sbc_file_managers)
            self.uniformity_test_results = pd.concat(list(results))
        else:
            self.uniformity_test_results = pd.concat(
                [_calc_rank_stat(fm) for fm in sbc_file_managers]
            )
        return self.uniformity_test_results

    def plot_uniformity(
        self,
        rank_stats: Optional[pd.DataFrame] = None,
        n_sims: Optional[int] = None,
        k_draws: int = SBC_UNIFORMITY_THINNING_DRAWS,
    ) -> matplotlib.axes.Axes:
        """Plot the results of the uniformity test.

        Args:
          rank_stats (Optional[pd.DataFrame]): Results of the uniformity test with the
            rank statistics for each variable. Defaults to None.
          n_sims (Optional[int], optional): Number of simulations performed If None
            (the default), then the value will be assumed to be the number of simulation
            directories. Defaults to None.
          k_draws (int, optional): Number of draws the posterior was thinned down to.
            Defaults to 100.
        """
        if rank_stats is None:
            if self.uniformity_test_results is None:
                raise src.exceptions.RequiredArgumentError(
                    "Parameter `rank_stats` must be passed because "
                    + "`self.uniformity_test_results` is None."
                )
            else:
                rank_stats = self.uniformity_test_results

        if n_sims is None:
            _sim_dirs = self.get_simulation_directories()
            self._check_n_sims(_sim_dirs)
            n_sims = len(_sim_dirs)

        ax = sns.histplot(data=rank_stats, x="rank_stat", binwidth=1)
        expected, lower, upper = expected_range_under_uniform(
            n_sims=n_sims, k_draws=k_draws
        )
        ax.fill_between(
            x=list(range(k_draws + 1)),
            y1=[lower] * (k_draws + 1),
            y2=[upper] * (k_draws + 1),
            color="#D3D3D3",
        )
        ax.axhline(expected, color="k", linestyle="-")
        return ax


def expected_range_under_uniform(
    n_sims: int, k_draws: int
) -> tuple[float, float, float]:
    """Use the expected distribution of rank statistics under a random binomial.

    Args:
        n_sims (int): Number of simulations.
        k_draws (int): Number of draws from the posterior.

    Returns:
        tuple[float, float, float]: The expected value and upper and lower 95% CI.
    """
    # Expected value.
    expected = n_sims / k_draws
    sd = np.sqrt((1 / k_draws) * (1 - 1 / k_draws) * n_sims)
    # 95% CI.
    upper = expected + 1.96 * sd
    lower = expected - 1.96 * sd
    return expected, lower, upper


def _fmt_tuple_to_label(tpl: tuple[int, ...]) -> str:
    if len(tpl) > 0:
        return "[" + ",".join([str(i) for i in tpl]) + "]"
    else:
        return ""


def _rank_statistic_to_dataframe(var_name: str, rank_stats: np.ndarray) -> pd.DataFrame:
    params: list[str] = []
    values: list[float] = []
    for idx, value in np.ndenumerate(rank_stats.squeeze()):
        params.append(var_name + _fmt_tuple_to_label(idx))
        values.append(value)
    return pd.DataFrame({"parameter": params, "rank_stat": values})


def calculate_parameter_rank_statistic(
    sbc_res: sbc.SBCResults, thin_to: int = 100
) -> pd.DataFrame:
    """Calculate the rank statistics for SBC uniformity analysis.

    Args:
        sbc_res (sbc.SBCResults): The results of an SBC simulation.
        thin_to (int, optional): How many draws to thin down to. Defaults to 100.

    Returns:
        pd.DataFrame: A data frame with the rank statistics for each parameter.
    """
    rank_stats_df = pd.DataFrame()
    var_names = pmhelp.get_posterior_names(sbc_res.inference_obj)
    for var_name in var_names:
        theta = sbc_res.priors[var_name]
        theta_prime = sbc_res.inference_obj["posterior"][var_name]
        theta_prime = pmhelp.thin_posterior(theta_prime, thin_to=thin_to)
        theta_prime = pmhelp.get_one_chain(theta_prime)
        rank_stat = (theta < theta_prime).values.sum(axis=0, keepdims=True)
        stat_df = _rank_statistic_to_dataframe(var_name, rank_stat)
        rank_stats_df = pd.concat([rank_stats_df, stat_df])
    rank_stats_df = rank_stats_df.reset_index(drop=True)
    return rank_stats_df
