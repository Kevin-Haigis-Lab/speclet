#!/usr/bin/env python3

"""Command line interface for collating SBC simulation results."""

from pathlib import Path
from typing import Optional

import typer

from src.command_line_interfaces import cli_helpers
from src.exceptions import UnsupportedFileTypeError
from src.loggers import logger
from src.modeling import simulation_based_calibration_helpers as sbc

cli_helpers.configure_pretty()


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
        raise NotADirectoryError(root_perm_dir)

    logger.info(
        f"Collating {num_permutations} SBC simulations in '{root_perm_dir.as_posix()}'."
    )
    df = sbc.collate_sbc_posteriors(
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
