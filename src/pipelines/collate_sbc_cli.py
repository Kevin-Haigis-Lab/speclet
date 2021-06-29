#!/usr/bin/env python3

"""Collate the posteriors of separate simulations of SBC into a single file."""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from src.command_line_interfaces import cli_helpers
from src.exceptions import IncorrectNumberOfFilesFoundError, UnsupportedFileTypeError
from src.loggers import logger
from src.modeling.simulation_based_calibration_helpers import SBCFileManager

cli_helpers.configure_pretty()


def _split_parameter(p: str) -> List[str]:
    return [a for a in re.split("\\[|,|\\]", p) if a != ""]


def _get_prior_value_using_index_list(ary: np.ndarray, idx: List[int]) -> float:
    """Extract a prior value from an array by indexes in a list.

    The input array may be any number of dimensions and the indices in the list each
    correspond to a single dimension. The final result will be a single value extract
    from the array.

    # TODO (refactor): move this to vector helpers.

    Args:
        ary (np.ndarray): Input array of priors of any dimension.
        idx (List[int]): List of indices, one per dimension.

    Returns:
        float: The value in the array at a location.
    """
    #     ary = ary.astype(float)
    if len(idx) == 0:
        return float(ary)

    assert len(idx) == len(ary.shape)

    value: np.ndarray = ary.copy()
    for i in idx:
        value = value[i]
    return float(value)


def _make_priors_dataframe(
    priors: Dict[str, np.ndarray], parameters: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame({"parameter": parameters, "true_value": 0}).set_index("parameter")
    for parameter in parameters:
        split_p = _split_parameter(parameter)
        param = split_p[0]
        idx = [int(i) for i in split_p[1:]]
        value = _get_prior_value_using_index_list(priors[param][0], idx)
        df.loc[parameter] = value
    return df


def _get_hdi_colnames_from_az_summary(df: pd.DataFrame) -> Tuple[str, str]:
    cols: List[str] = [c for c in df.columns if "hdi_" in c]
    cols = [c for c in cols if "%" in c]
    assert len(cols) == 2
    return cols[0], cols[1]


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
    hdi_low, hdi_high = _get_hdi_colnames_from_az_summary(df)
    df["within_hdi"] = _is_true_value_within_hdi(
        df[hdi_low], df["true_value"], df[hdi_high]
    )
    return df


def get_posterior_summary_for_file_manager(sbc_dir: Path) -> pd.DataFrame:
    """Create a summary of the results of an SBC sim. cached in a directory.

    Args:
        sbc_dir (Path): Directory with the results of a SBC simulation.

    Raises:
        Exception: Raised if the path is not to a directory.

    Returns:
        pd.DataFrame: Dataframe with the summary of the simulation results.
    """
    sbc_fm = SBCFileManager(sbc_dir)
    if not sbc_fm.all_data_exists():
        raise Exception(f"Not all output from '{sbc_fm.dir.name}' exist.")
    res = sbc_fm.get_sbc_results()
    true_values = _make_priors_dataframe(
        res.priors, parameters=res.posterior_summary.index.values.tolist()
    )
    return res.posterior_summary.merge(true_values, left_index=True, right_index=True)


def _index_and_concat_summaries(post_summaries: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [
            d.assign(simulation_id=f"sim_id_{str(i).rjust(4, '0')}")
            for i, d in enumerate(post_summaries)
        ]
    )


def _extract_paramter_names(df: pd.DataFrame) -> pd.DataFrame:
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
    simulation_posteriors: List[pd.DataFrame] = [
        get_posterior_summary_for_file_manager(d) for d in tqdm_posterior_dirs
    ]

    if num_permutations is not None and len(simulation_posteriors) != num_permutations:
        raise IncorrectNumberOfFilesFoundError(
            num_permutations, len(simulation_posteriors)
        )

    simulation_posteriors_df = (
        _index_and_concat_summaries(simulation_posteriors)
        .pipe(_extract_paramter_names)
        .pipe(_assign_column_for_within_hdi)
    )

    return simulation_posteriors_df


def collate_sbc_posteriors_cli(
    root_perm_dir: Path,
    output_path: Path,
    num_permutations: Optional[int] = None,
) -> None:
    """Command line interface for collating the results of many SBC simulations.

    Args:
        root_perm_dir (Path): Path to the root directory containing the subdirectories
          with the results of all of the simulations.
        output_path (Path): Path to save the results to. Can either be saved as a pickle
          or CSV.
        num_permutations (Optional[int], optional):  Number of permutations expected. If
          supplied, this will be checked against the number of found simulations.
          Defaults to None.

    Raises:
        UnsupportedOutputFileTypeError: Raised if the output path has an unsupported
        extension.
    """
    if not root_perm_dir.is_dir():
        raise NotADirectoryError()

    logger.info(
        f"Collating {num_permutations} SBC simulations in '{root_perm_dir.as_posix()}'."
    )
    df = collate_sbc_posteriors(
        posterior_dirs=root_perm_dir.iterdir(), num_permutations=num_permutations
    )
    if output_path.suffix == ".pkl":
        logger.info(f"Saving final results to pickle: '{output_path.as_posix()}'.")
        df.to_pickle(output_path.as_posix())
    elif output_path.suffix == ".csv":
        logger.info(f"Saving final results to CSV: '{output_path.as_posix()}'.")
        df.to_csv(output_path)
    else:
        raise UnsupportedFileTypeError(output_path.suffix)


if __name__ == "__main__":
    typer.run(collate_sbc_posteriors_cli)


# TODO:
# 1. docstrings for above functions ✅
# 2. add logging ✅
# 3. test behavior ✅
# 4. (in pipeline) add resource requirements for SLURM in config
# 5. run SBC pipeline with a few simulations (need to remove files in reports)
# 6. debug
