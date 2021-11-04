#!/usr/bin/env python3

"""A helpful command line interface for simulation-based calibration."""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import theano
import typer
import xarray

from src import model_configuration as model_config
from src.command_line_interfaces import cli_helpers
from src.loggers import logger
from src.modeling import simulation_based_calibration_helpers as sbc
from src.pipelines.theano_flags import get_theano_compile_dir
from src.project_enums import MockDataSize, ModelFitMethod, SpecletPipeline

cli_helpers.configure_pretty()

app = typer.Typer()


class FailedSBCCheckError(BaseException):
    """Failed SBC check error."""

    pass


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
    sp_model = model_config.get_config_and_instantiate_model(
        config_path=config_path, name=name, root_cache_dir=Path(tempfile.gettempdir())
    )
    mock_data = sp_model.generate_mock_data(size=data_size, random_seed=random_seed)
    mock_data.to_csv(save_path)


def _check_theano_config() -> None:
    logger.info(f"theano.config.compile__wait: {theano.config.compile__wait}")
    logger.info(f"theano.config.compile__timeout: {theano.config.compile__timeout}")
    return None


def _remove_thenao_comp_dir() -> None:
    _theano_comp_dir = get_theano_compile_dir()
    logger.info(f"Removing Theano compilation directory: '{_theano_comp_dir}'.")
    shutil.rmtree(_theano_comp_dir, ignore_errors=True)


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
    remove_theano_comp_dir: bool = False,
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
          (based off of known issues with the SBC pipeline's fidelity)? Defaults to
          True.
        remove_theano_comp_dir (bool, optional): Should the Theano compilation
          directory be removed when the SBC finishes? Only use this if you are
          certain that no other job is using this compilation directory. Defaults to
          False.

    Returns:
        None: None
    """
    _check_theano_config()
    sp_model = model_config.get_config_and_instantiate_model(
        config_path=config_path, name=name, root_cache_dir=cache_dir
    )
    config_sampling_kwargs = model_config.get_sampling_kwargs(
        config_path=config_path,
        name=name,
        pipeline=SpecletPipeline.SBC,
        fit_method=fit_method,
    )

    fit_kwargs: dict[str, Any]
    if config_sampling_kwargs is not None:
        config_sampling_kwargs.random_seed = sim_number
        fit_kwargs = config_sampling_kwargs.dict()
    else:
        fit_kwargs = {"random_seed": sim_number}

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
        sbc_check = _check_sbc_results(cache_dir=cache_dir, fit_method=fit_method)
        if sbc_check.result:
            logger.info("SBC check successful with no errors detected.")
        else:
            logger.error(f"SBC check failed: {sbc_check.message}")
            logger.warn("Clearing cached results of SBC.")
            sbc_check.sbc_file_manager.clear_results()
            sbc_check.sbc_file_manager.clear_saved_data()
            sp_model.cache_manager.clear_all_caches()
            if remove_theano_comp_dir:
                _remove_thenao_comp_dir()
            raise FailedSBCCheckError(sbc_check.message)

    if remove_theano_comp_dir:
        _remove_thenao_comp_dir()

    return None


@dataclass
class SBCCheckResult:
    """Results of the SBC check."""

    sbc_file_manager: sbc.SBCFileManager
    result: bool
    message: str


def _check_sbc_results(cache_dir: Path, fit_method: ModelFitMethod) -> SBCCheckResult:
    sbc_fm = sbc.SBCFileManager(cache_dir)

    if not sbc_fm.all_data_exists():
        return SBCCheckResult(
            sbc_fm, result=False, message="Not all result files exist."
        )
    if not sbc_fm.simulation_data_exists():
        return SBCCheckResult(
            sbc_fm, result=False, message="Mock data file does not exist."
        )

    sbc_res = sbc_fm.get_sbc_results()

    if fit_method is ModelFitMethod.MCMC:
        if not hasattr(sbc_res.inference_obj, "sample_stats"):
            return SBCCheckResult(
                sbc_fm,
                result=False,
                message="No sampling statistics.",
            )
        if not isinstance(sbc_res.inference_obj.get("sample_stats"), xarray.Dataset):
            return SBCCheckResult(
                sbc_fm,
                result=False,
                message="Sampling statistics is not a xarray.Dataset.",
            )

    return SBCCheckResult(sbc_fm, True, "")


if __name__ == "__main__":
    app()
