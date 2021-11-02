#!/usr/bin/env python3

"""Command line interface for merging MCMC chains for a single model."""

from pathlib import Path
from typing import Optional

import arviz as az
import typer

from src import model_configuration as model_config
from src.command_line_interfaces import cli_helpers
from src.loggers import logger

cli_helpers.configure_pretty()


def combine_mcmc_chains(
    name: str,
    config_path: Path,
    chain_dirs: list[Path],
    combined_root_cache_dir: Path,
    touch: Optional[Path] = None,
) -> None:
    """Combine multiple MCMC chains for a single model.

    Args:
        name (str): Name of the model.
        config_path (Path): Path to the model configuration file.
        chain_dirs (List[Path]): The paths to the individual chains' cache directories.
        combined_root_cache_dir (Path): Cache directory for combined model's MCMC.
        touch (bool, optional): Should there be a file touched to indicate that the
          sampling process is complete? This is helpful for telling pipelines/workflows
          that this step is complete. Defaults to False.

    Raises:
        TypeError: Raised if something has gone wrong in the merge.
    """
    n_chains = len(chain_dirs)
    logger.info(f"Expecting {n_chains} separate chains.")

    all_arviz_data_objects: list[az.InferenceData] = []
    for chain_dir in chain_dirs:
        logger.info(f"Gather MCMC chain '{chain_dir.as_posix()}'.")
        sp_model = model_config.get_config_and_instantiate_model(
            config_path=config_path, name=name, root_cache_dir=chain_dir
        )

        logger.debug(f"Cache directory: {sp_model.cache_manager.cache_dir.as_posix()}")
        logger.debug(
            f"Cache directory exists: {sp_model.cache_manager.cache_dir.exists()}"
        )
        _mcmc_cache_dir = sp_model.cache_manager.mcmc_cache_delegate.cache_dir
        logger.debug(f"MCMC cache directory: {_mcmc_cache_dir.as_posix()}")
        logger.debug(f"MCMC cache directory exists: {_mcmc_cache_dir.exists()}")

        if sp_model.cache_manager.mcmc_cache_exists():
            logger.info(f"Found chain '{chain_dir.as_posix()}'.")
            sp_model.build_model()
            assert sp_model.model is not None
            mcmc_res = sp_model.mcmc_sample_model()
            all_arviz_data_objects.append(mcmc_res)
        else:
            logger.error(
                f"Cache for SpecletModel '{sp_model.name}' does not exist - skipping."
            )

    if len(all_arviz_data_objects) != n_chains:
        logger.error(f"Expected {n_chains} but found {len(all_arviz_data_objects)}.")
    else:
        logger.info(f"Collected {len(all_arviz_data_objects)} chains.")

    combined_chains = az.concat(all_arviz_data_objects, dim="chain")
    combined_sp_model = sp_model = model_config.get_config_and_instantiate_model(
        config_path=config_path, name=name, root_cache_dir=combined_root_cache_dir
    )
    if isinstance(combined_chains, az.InferenceData):
        combined_sp_model.mcmc_results = combined_chains
    else:
        logger.error(f"`combined_chains` of type '{type(combined_chains)}'")
        raise TypeError("`combined_chains` was not of type 'arviz.InferenceData'.")
    combined_sp_model.write_mcmc_cache()

    logger.info(f"Finished combining chains for model '{combined_sp_model.name}'")

    if touch is not None:
        logger.info(f"Touching file: '{touch.as_posix()}'.")
        touch.touch()

    return None


if __name__ == "__main__":
    typer.run(combine_mcmc_chains)
