"""Helpers for organizing simualtaion-based calibrations."""

import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import src.exceptions
from src.data_processing import vectors as vhelp
from src.modeling import pymc3_helpers as pmhelp
from src.modeling.pymc3_analysis import get_hdi_colnames_from_az_summary

#### ---- File management ---- ####


class SBCResults:
    """Results from a single round of SBC."""

    priors: dict[str, Any]
    inference_obj: az.InferenceData
    posterior_summary: pd.DataFrame

    def __init__(
        self,
        priors: dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ):
        """Create an instance of SBCResults.

        Args:
            priors (dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        self.priors = priors
        self.inference_obj = inference_obj
        self.posterior_summary = posterior_summary


class SBCFileManager:
    """Manages the results from a round of simulation-based calibration."""

    dir: Path
    sbc_data_path: Path
    inference_data_path: Path
    priors_path_set: Path
    priors_path_get: Path
    posterior_summary_path: Path

    sbc_data: Optional[pd.DataFrame] = None
    sbc_results: Optional[SBCResults] = None

    def __init__(self, dir: Path):
        """Create a SBCFileManager.

        Args:
            dir (Path): The directory where the data is stored.
        """
        if not dir.is_dir():
            raise NotADirectoryError(dir)
        self.dir = dir
        self.sbc_data_path = dir / "sbc-data.csv"
        self.inference_data_path = dir / "inference-data.netcdf"
        self.priors_path_set = dir / "priors"
        self.priors_path_get = dir / "priors.npz"
        self.posterior_summary_path = dir / "posterior-summary.csv"
        _ = self._check_dir_exists()

    def _check_dir_exists(self) -> bool:
        if self.dir.exists():
            return True
        else:
            self.dir.mkdir()
            return False

    def get_sbc_data(self, re_read: bool = False) -> pd.DataFrame:
        """Get the simulated data for the SBC.

        Args:
            re_read (bool, optional): Force re-reading from file. Defaults to False.

        Returns:
            pd.DataFrame: Saved simulated data frame.
        """
        if self.sbc_data is None or re_read:
            self.sbc_data = pd.read_csv(self.sbc_data_path)
        return self.sbc_data

    def save_sbc_data(
        self, data: pd.DataFrame, index: bool = False, **kwargs: dict[str, Any]
    ) -> None:
        """Save SBC dataframe to disk.

        Args:
            data (pd.DataFrame): Simulated data used for the SBC.
            index (bool, optional): Should the index be included in the file (CSV).
              Defaults to False.
            kwargs (dict[str, Any]): Additional keyword arguments for `data.to_csv()`.
        """
        self.sbc_data = data
        data.to_csv(self.sbc_data_path, index=index)

    def simulation_data_exists(self) -> bool:
        """Does the simulation dataframe file exist?

        Returns:
            bool: True if the dataframe file exists.
        """
        return self.sbc_data_path.exists()

    def clear_saved_data(self) -> None:
        """Clear save SBC dataframe file."""
        if self.simulation_data_exists():
            self.sbc_data_path.unlink()

    def save_sbc_results(
        self,
        priors: dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ) -> None:
        """Save the results from a round of SBC.

        Args:
            priors (dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        inference_obj.to_netcdf(self.inference_data_path.as_posix())
        np.savez(self.priors_path_set.as_posix(), **priors)
        posterior_summary.to_csv(
            self.posterior_summary_path.as_posix(), index_label="parameter"
        )

    def _tidy_numpy_files(self, files: Any) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {}
        for k in files.files:
            d[k] = files[k]
        return d

    def get_sbc_results(self, re_read: bool = False) -> SBCResults:
        """Retrieve results of a round of SBC.

        Args:
            re_read (bool, optional): Should the results be re-read from file?
              Defaults to False.

        Returns:
            SBCResults: The results from the round of SBC.
        """
        if self.sbc_results is not None and not re_read:
            return self.sbc_results

        inference_obj = az.from_netcdf(self.inference_data_path)
        priors_files = np.load(self.priors_path_get.as_posix())
        priors = self._tidy_numpy_files(priors_files)
        posterior_summary = pd.read_csv(
            self.posterior_summary_path, index_col="parameter"
        )

        self.sbc_results = SBCResults(
            priors=priors,
            inference_obj=inference_obj,
            posterior_summary=posterior_summary,
        )
        return self.sbc_results

    def all_data_exists(self) -> bool:
        """Confirm that all data exists.

        Returns:
            bool: True if all of the data exists, else false.
        """
        for p in [
            self.priors_path_get,
            self.posterior_summary_path,
            self.inference_data_path,
        ]:
            if not p.exists():
                return False
        return True

    def clear_results(self) -> None:
        """Clear the stored SBC results (if they exist)."""
        for f in (
            self.inference_data_path,
            self.priors_path_get,
            self.posterior_summary_path,
        ):
            if f.exists():
                f.unlink()


#### ---- Results analysis ---- ####

SBC_UNIFORMITY_THINNING_DRAWS = 100


class SBCAnalysis:
    """Analysis of SBC results."""

    root_dir: Path
    pattern: str
    n_simulations: Optional[int]

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

    def get_simulation_file_managers(self) -> list[SBCFileManager]:
        """Get the file managers for each simulation.

        Returns:
            list[SBCFileManager]: List of SBC file managers.
        """
        return [SBCFileManager(p) for p in self.get_simulation_directories()]

    def get_simulation_results(self) -> list[SBCResults]:
        """Get all of the simulation results.

        Raises:
            src.exceptions.CacheDoesNotExistError: Raised if the results do not exist
            for a simulation.

        Returns:
            list[SBCResults]: List of all simulation results.
        """
        fms = self.get_simulation_file_managers()
        results: list[SBCResults] = []
        for fm in fms:
            if fm.all_data_exists():
                results.append(fm.get_sbc_results())
            else:
                raise src.exceptions.CacheDoesNotExistError(fm.dir)

        self._check_n_sims(results)
        return results

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
            simulation_posteriors_df = collate_sbc_posteriors(
                posterior_dirs=self.get_simulation_directories(),
                num_permutations=self.n_simulations,
            )
        return (
            simulation_posteriors_df.copy()
            .groupby(["parameter_name"])["within_hdi"]
            .mean()
            .reset_index(drop=False)
            .sort_values("within_hdi", ascending=False)
            .reset_index(drop=True)
        )

    def uniformity_test(
        self, k_draws: int = SBC_UNIFORMITY_THINNING_DRAWS
    ) -> pd.DataFrame:
        """Perform the SBC uniformity analysis.

        Args:
            k_draws (int, optional): Number of draws to thin the posterior samples down
              to. Defaults to 100.

        Returns:
            pd.DataFrame: A data frame of the rank statistic of each variable in each
            simulation.
        """
        sbc_file_managers = self.get_simulation_file_managers()
        self._check_n_sims(sbc_file_managers)
        rank_statistics = pd.concat(
            [
                calculate_parameter_rank_statistic(
                    sbc_fm.get_sbc_results(), thin_to=k_draws
                )
                for sbc_fm in sbc_file_managers
            ]
        )
        return rank_statistics

    def plot_uniformity(
        self,
        rank_stats: pd.DataFrame,
        n_sims: Optional[int] = None,
        k_draws: int = SBC_UNIFORMITY_THINNING_DRAWS,
    ) -> matplotlib.axes.Axes:
        """Plot the results of the uniformity test.

        Args:
            rank_stats (pd.DataFrame): Results of the uniformity test with the rank
            statistics for each variable.
            n_sims (Optional[int], optional): Number of simulations performed If None
            (the default), then the value will be assumed to be the number of simulation
            directories. Defaults to None.
            k_draws (int, optional): Number of draws the posterior was thinned down to.
            Defaults to 100.
        """
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
    sbc_res: SBCResults, thin_to: int = 100
) -> pd.DataFrame:
    """Calculate the rank statistics for SBC uniformity analysis.

    Args:
        sbc_res (SBCResults): The results of an SBC simulation.
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


#### ---- Collate SBC ---- ####


def _split_parameter(p: str) -> list[str]:
    return [a for a in re.split("\\[|,|\\]", p) if a != ""]


def _get_prior_value_using_index_list(ary: np.ndarray, idx: list[int]) -> float:
    """Extract a prior value from an array by indexes in a list.

    The input array may be any number of dimensions and the indices in the list each
    correspond to a single dimension. The final result will be a single value extract
    from the array.

    Args:
        ary (np.ndarray): Input array of priors of any dimension.
        idx (list[int]): List of indices, one per dimension.

    Returns:
        float: The value in the array at a location.
    """
    if ary.shape == (1,):
        ary = np.asarray(ary[0])
    res = vhelp.index_array_by_list(ary, idx)
    assert len(res.shape) == 0
    return float(res)


def _make_priors_dataframe(
    priors: dict[str, np.ndarray], parameters: list[str]
) -> pd.DataFrame:
    df = pd.DataFrame({"parameter": parameters, "true_value": 0}).set_index("parameter")
    for parameter in parameters:
        split_p = _split_parameter(parameter)
        param = split_p[0]
        idx = [int(i) for i in split_p[1:]]
        value = _get_prior_value_using_index_list(priors[param][0], idx)
        df.loc[parameter] = value
    return df


def _is_true_value_within_hdi(
    low_hdi: pd.Series, true_vals: pd.Series, high_hdi: pd.Series
) -> np.ndarray:
    return (
        (low_hdi.values < true_vals.values).astype(int)
        * (true_vals.values < high_hdi.values).astype(int)
    ).astype(bool)


def _assign_column_for_within_hdi(
    df: pd.DataFrame, true_value_col: str = "true_value"
) -> pd.DataFrame:
    hdi_low, hdi_high = get_hdi_colnames_from_az_summary(df)
    df["within_hdi"] = _is_true_value_within_hdi(
        df[hdi_low], df["true_value"], df[hdi_high]
    )
    return df


class SBCResultsNotFoundError(FileNotFoundError):
    """SBC Results not found."""

    pass


def get_posterior_summary_for_file_manager(sbc_dir: Path) -> pd.DataFrame:
    """Create a summary of the results of an SBC sim. cached in a directory.

    Args:
        sbc_dir (Path): Directory with the results of a SBC simulation.

    Raises:
        SBCResultsNotFoundError: Raised if the SBC results are not found.

    Returns:
        pd.DataFrame: Dataframe with the summary of the simulation results.
    """
    sbc_fm = SBCFileManager(sbc_dir)
    if not sbc_fm.all_data_exists():
        raise SBCResultsNotFoundError(f"Not all output from '{sbc_fm.dir.name}' exist.")
    res = sbc_fm.get_sbc_results()
    true_values = _make_priors_dataframe(
        res.priors, parameters=res.posterior_summary.index.values.tolist()
    )
    return res.posterior_summary.merge(true_values, left_index=True, right_index=True)


def _index_and_concat_summaries(post_summaries: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            d.assign(simulation_id=f"sim_id_{str(i).rjust(4, '0')}")
            for i, d in enumerate(post_summaries)
        ]
    )


def _extract_parameter_names(df: pd.DataFrame) -> pd.DataFrame:
    df["parameter_name"] = [x.split("[")[0] for x in df.index.values]
    df = df.set_index("parameter_name", append=True)
    return df


def collate_sbc_posteriors(
    posterior_dirs: Iterable[Path], num_permutations: Optional[int] = None
) -> pd.DataFrame:
    """Collate many SBC posteriors.

    Args:
        posterior_dirs (Iterable[Path]): The directories containing the stored results
          of many SBC simulations.
        num_permutations (Optional[int], optional): Number of permutations expected. If
          supplied, this will be checked against the number of found simulations.
          Defaults to None.

    Raises:
        IncorrectNumberOfFilesFoundError: Raised if the number of found simulations is
        not equal to the number of expected simulations.

    Returns:
        pd.DataFrame: A single data frame with all of the results of the simulations.
    """
    tqdm_posterior_dirs = tqdm(posterior_dirs, total=num_permutations)
    simulation_posteriors: list[pd.DataFrame] = [
        get_posterior_summary_for_file_manager(d) for d in tqdm_posterior_dirs
    ]

    if num_permutations is not None and len(simulation_posteriors) != num_permutations:
        raise src.exceptions.IncorrectNumberOfFilesFoundError(
            num_permutations, len(simulation_posteriors)
        )

    simulation_posteriors_df = (
        _index_and_concat_summaries(simulation_posteriors)
        .pipe(_extract_parameter_names)
        .pipe(_assign_column_for_within_hdi)
    )

    return simulation_posteriors_df
