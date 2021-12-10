#!/usr/bin/env python3

"""Command line interface for merging MCMC chains for a single model."""

from pathlib import Path
from typing import Optional

import arviz as az
import typer

from speclet.command_line_interfaces import cli_helpers
from speclet.exceptions import CacheDoesNotExistError
from speclet.loggers import logger
from speclet.managers.cache_manager import PosteriorManager, get_posterior_cache_name
from speclet.model_configuration import get_configuration_for_model
from speclet.project_enums import ModelFitMethod

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
    model_config = get_configuration_for_model(config_path=config_path, name=name)
    assert model_config is not None
    for chain_dir in chain_dirs:
        logger.info(f"Gather MCMC chain '{chain_dir.as_posix()}'.")
        posterior_manager = PosteriorManager(
            get_posterior_cache_name(
                model_name=name, fit_method=ModelFitMethod.STAN_MCMC
            ),
            cache_dir=chain_dir,
        )
        if posterior_manager.cache_exists:
            _posterior = posterior_manager.get()
            assert _posterior is not None
            all_arviz_data_objects.append(_posterior)
        else:
            logger.error(f"Cache for model '{name}' does not exist.")
            raise CacheDoesNotExistError(posterior_manager.cache_path)

    if len(all_arviz_data_objects) != n_chains:
        msg = f"Expected {n_chains} but found {len(all_arviz_data_objects)}."
        logger.error(msg)
        raise BaseException(msg)
    else:
        logger.info(f"Collected {len(all_arviz_data_objects)} chains.")

    combined_chains = az.concat(all_arviz_data_objects, dim="chain")
    assert isinstance(combined_chains, az.InferenceData)
    _ = PosteriorManager(id=name, cache_dir=combined_root_cache_dir).put(
        combined_chains
    )

    if touch is not None:
        logger.info(f"Touching file: '{touch.as_posix()}'.")
        touch.touch()

    return None


if __name__ == "__main__":
    typer.run(combine_mcmc_chains)
