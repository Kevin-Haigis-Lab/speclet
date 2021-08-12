#!/usr/bin/env python3

"""A helpful command line interface for simulation-based calibration."""

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import xarray

from src.command_line_interfaces import cli_helpers
from src.io import model_config
from src.loggers import logger
from src.modeling import simulation_based_calibration_helpers as sbc
from src.models import configuration
from src.project_enums import MockDataSize, ModelFitMethod, SpecletPipeline

cli_helpers.configure_pretty()

app = typer.Typer()


@app.command()
def make_mock_data(
    name: str,
    config_path: Path,
    data_size: MockDataSize,
    save_path: Path,
    random_seed: Optional[int] = None,
) -> None:
    """Generate mock data for a model.

    Args:
        name (str): Unique identifiable name for the model.
        config_path (Path): Path to the model configuration file.
        save_path (Optional[Path]): Path to save the data frame to as a CSV.
        data_size (Optional[MockDataSize]): What size of mock data should be generated?
          Is ignored if a path is supplied to pre-existing mock data in the
          `mock_data_path` argument. Defaults to None.
        save_path (Optional[Path]): Path to save the data frame to as a CSV.
        random_seed (Optional[int]): Random seed for data generation process.
    """
    sp_model = configuration.get_config_and_instantiate_model(
        config_path=config_path, name=name, root_cache_dir=Path(tempfile.gettempdir())
    )
    mock_data = sp_model.generate_mock_data(size=data_size, random_seed=random_seed)
    mock_data.to_csv(save_path)


@app.command()
def run_sbc(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    sim_number: int,
    mock_data_path: Optional[Path] = None,
    data_size: Optional[MockDataSize] = None,
    check_results: bool = True,
) -> None:
    """CLI for running a round of simulation-based calibration for a model.

    Args:
        name (str): Unique identifiable name for the model.
        config_path (Path): Path to the model configuration file.
        fit_method (ModelFitMethod): Fitting method.
        cache_dir (Path): Where to store the results.
        sim_number (int): Simulation number.
        mock_data_path (Optional[Path]): Path to pre-existing mock data (formatted as a
          CSV) to use in the SBC. If novel data should be generated instead, pass the
          desied size to the `data_size` argument. Defaults to None.
        data_size (Optional[MockDataSize]): What size of mock data should be generated?
          Is ignored if a path is supplied to pre-existing mock data in the
          `mock_data_path` argument. Defaults to None.
        check_results (bool, optional): Should the results be checked for completeness
          (based off of known issues with the SBC pipeline's fidelity)? Defaults to True

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

    mock_data: Optional[pd.DataFrame] = None
    if mock_data_path is not None:
        logger.info("Loading pre-existing mock data for the SBC.")
        mock_data = pd.read_csv(mock_data_path)

    logger.info(f"Running SBC for model '{sp_model.__class__}' - '{name}'.")
    sp_model.run_simulation_based_calibration(
        cache_dir,
        fit_method=fit_method,
        random_seed=sim_number,
        mock_data=mock_data,
        size=data_size,
        fit_kwargs=fit_kwargs,
    )

    logger.info("SBC finished.")

    if check_results:
        logger.info("Checking SBC results...")
        _check_sbc_results(cache_dir=cache_dir, fit_method=fit_method)
    return None


def _check_sbc_results(cache_dir: Path, fit_method: ModelFitMethod) -> bool:
    sbc_fm = sbc.SBCFileManager(cache_dir)
    res = sbc_fm.get_sbc_results()

    assert hasattr(res.inference_obj, "sample_stats")
    assert isinstance(res.inference_obj.get("sample_stats"), xarray.Dataset)

    return True


if __name__ == "__main__":
    app()
