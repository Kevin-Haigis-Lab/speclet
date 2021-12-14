"""Helpers for organizing simulation-based calibrations."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm

import speclet.exceptions
from speclet.analysis.arviz_analysis import get_hdi_colnames_from_az_summary
from speclet.data_processing import vectors as vhelp

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


def _collate_sbc_posteriors_singlethreaded(
    posterior_dirs: Iterable[Path], n_perms: Optional[int]
) -> list[pd.DataFrame]:
    tqdm_posterior_dirs = tqdm(posterior_dirs, total=n_perms)
    return [get_posterior_summary_for_file_manager(d) for d in tqdm_posterior_dirs]


def _collate_sbc_posteriors_multithreaded(
    posterior_dirs: Iterable[Path], n_perms: Optional[int]
) -> list[pd.DataFrame]:
    simulation_posteriors = []
    with tqdm(total=n_perms) as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(get_posterior_summary_for_file_manager, d)
                for d in posterior_dirs
            ]
            for future in as_completed(futures):
                simulation_posteriors.append(future.result())
                pbar.update(1)
    return simulation_posteriors


def collate_sbc_posteriors(
    posterior_dirs: Iterable[Path],
    num_permutations: Optional[int] = None,
    multithreaded: bool = True,
) -> pd.DataFrame:
    """Collate many SBC posteriors.

    Args:
        posterior_dirs (Iterable[Path]): The directories containing the stored results
          of many SBC simulations.
        num_permutations (Optional[int], optional): Number of permutations expected. If
          supplied, this will be checked against the number of found simulations.
          Defaults to None.
        multithreaded (bool, optional): Should the results be collected using multiple
          threads? Defaults to True.

    Raises:
        IncorrectNumberOfFilesFoundError: Raised if the number of found simulations is
        not equal to the number of expected simulations.

    Returns:
        pd.DataFrame: A single data frame with all of the results of the simulations.
    """
    simulation_posteriors: list[pd.DataFrame]

    if multithreaded:
        simulation_posteriors = _collate_sbc_posteriors_multithreaded(
            posterior_dirs, n_perms=num_permutations
        )
    else:
        simulation_posteriors = _collate_sbc_posteriors_singlethreaded(
            posterior_dirs, n_perms=num_permutations
        )

    if num_permutations is not None and len(simulation_posteriors) != num_permutations:
        raise speclet.exceptions.IncorrectNumberOfFilesFoundError(
            num_permutations, len(simulation_posteriors)
        )

    simulation_posteriors_df = (
        _index_and_concat_summaries(simulation_posteriors)
        .pipe(_extract_parameter_names)
        .pipe(_assign_column_for_within_hdi)
    )

    return simulation_posteriors_df
