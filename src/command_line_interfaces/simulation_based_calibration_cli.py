#!/usr/bin/env python3

"""A helpful command line interface for simulation-based calibration."""

from pathlib import Path

import typer

from src.command_line_interfaces import cli_helpers
from src.io import model_config
from src.loggers import logger
from src.models import configuration
from src.project_enums import MockDataSize, ModelFitMethod, SpecletPipeline

cli_helpers.configure_pretty()


def run_sbc(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    sim_number: int,
    data_size: MockDataSize,
) -> None:
    """CLI for running a round of simulation-based calibration for a model.

    Args:
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
    sp_model = configuration.get_config_and_instantiate_model(
        config_path=config_path, name=name, root_cache_dir=cache_dir
    )
    fit_kwargs = model_config.get_sampling_kwargs(
        config_path=config_path,
        name=name,
        pipeline=SpecletPipeline.SBC,
        fit_method=fit_method,
    )
    if fit_kwargs != {}:
        logger.info("Found specific fitting keyword arguments.")

    logger.info(f"Running SBC for model '{sp_model.__class__}' - '{name}'.")
    sp_model.run_simulation_based_calibration(
        cache_dir,
        fit_method=fit_method,
        random_seed=sim_number,
        size=data_size,
        fit_kwargs=fit_kwargs,
    )
    logger.info("SBC finished.")
    return None


if __name__ == "__main__":
    typer.run(run_sbc)
