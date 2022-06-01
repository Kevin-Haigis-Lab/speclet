#!/usr/bin/env python3

"""Command line interface for merging MCMC chains for a single model."""

from pathlib import Path

import arviz as az
from typer import Typer

from speclet.cli import cli_helpers
from speclet.exceptions import CacheDoesNotExistError
from speclet.loggers import logger
from speclet.managers.cache_manager import PosteriorManager, get_posterior_cache_name
from speclet.model_configuration import get_configuration_for_model
from speclet.project_enums import ModelFitMethod

cli_helpers.configure_pretty()
app = Typer()


@app.command()
def combine_mcmc_chains(
    name: str,
    fit_method: ModelFitMethod,
    n_chains: int,
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> None:
    """Combine multiple MCMC chains for a single model.

    Args:
        name (str): Name of the model.
        fit_method (ModelFitMethod): Method used to fit the model (should be MCMC).
        n_chains (int): Number of chains.
        config_path (Path): Path to the model configuration file.
        cache_dir (Path): Directory with the separate chains.
        output_dir (Path): Cache directory for combined model's MCMC.

    Raises:
        TypeError: Raised if something has gone wrong in the merge.
    """
    logger.info(f"Expecting {n_chains} separate chains.")

    model_config = get_configuration_for_model(config_path=config_path, name=name)
    assert model_config is not None

    cache_name = get_posterior_cache_name(model_name=name, fit_method=fit_method)
    all_idata_objects: list[az.InferenceData] = []
    for chain_i in range(n_chains):
        logger.info(f"Gather MCMC chain #{chain_i}.")
        chain_id = f"{cache_name}_chain{chain_i}"
        post_man = PosteriorManager(id=chain_id, cache_dir=cache_dir)
        if post_man.posterior_cache_exists:
            _posterior = post_man.get()
            assert _posterior is not None
            all_idata_objects.append(_posterior)
        else:
            logger.error(f"Cache for chain #{chain_i} '{chain_id}' does not exist.")
            raise CacheDoesNotExistError(post_man.posterior_path)

    if len(all_idata_objects) != n_chains:
        msg = f"Expected {n_chains} but found {len(all_idata_objects)}."
        logger.error(msg)
        raise BaseException(msg)
    else:
        logger.info(f"Collected {len(all_idata_objects)} chains.")

    logger.info("Merging all chains.")
    combined_chains = az.concat(all_idata_objects, dim="chain")
    assert isinstance(combined_chains, az.InferenceData)
    logger.info("Writing posterior samples to file.")
    PosteriorManager(id=cache_name, cache_dir=output_dir).put(combined_chains)
    logger.info("Finished writing posterior samples to file.")
    return None


if __name__ == "__main__":
    app()
