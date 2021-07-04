#!/usr/bin/env python3

"""A helpful command line interface for simulation-based calibration."""

from pathlib import Path

import typer

from src.command_line_interfaces import cli_helpers
from src.loggers import logger
from src.models.configuration import instantiate_and_configure_model
from src.project_enums import ModelFitMethod, ModelOption

cli_helpers.configure_pretty()


def run_sbc(
    model_class: ModelOption,
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    sim_number: int,
    data_size: str,
) -> None:
    """CLI for running a round of simulation-based calibration for a model.

    Args:
        model_class (ModelOption): Name of the model to use.
        name (str): Unique identifiable name for the model.
        config_path (Path): Path to the model configuration file.
        fit_method (ModelFitMethod): Fitting method.
        cache_dir (Path): Where to store the results.
        sim_number (int): Simulation number.
        data_size (str): Which data size to use. See the actual methods
          for details and options.

    Returns:
        None: None
    """
    logger.info(f"Running SBC for model '{model_class.value}' named '{name}'.")
    name = cli_helpers.clean_model_names(name)
    sp_model = instantiate_and_configure_model(
        model_opt=model_class,
        name=name,
        root_cache_dir=cache_dir,
        debug=False,
        config_path=config_path,
    )
    sp_model.run_simulation_based_calibration(
        cache_dir,
        fit_method=fit_method,
        random_seed=sim_number,
        size=data_size,
    )
    logger.info("SBC finished.")
    return None


if __name__ == "__main__":
    typer.run(run_sbc)
