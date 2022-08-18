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
    merged_idata: az.InferenceData | None = None
    for chain_i in range(n_chains):
        logger.info(f"Gather MCMC chain #{chain_i}.")
        chain_id = f"{cache_name}_chain{chain_i}"
        post_man = PosteriorManager(id=chain_id, cache_dir=cache_dir)
        if not post_man.posterior_cache_exists:
            logger.error(f"Cache for chain #{chain_i} '{chain_id}' does not exist.")
            raise CacheDoesNotExistError(post_man.posterior_path)
        _new_idata = post_man.get()
        assert _new_idata is not None, "Posterior object not loaded."
        if merged_idata is None:
            logger.info("First posterior object - setting as `merged_idata`.")
            merged_idata = _new_idata
        else:
            logger.info("Merging chains.")
            az.concat(merged_idata, _new_idata, dim="chain", inplace=True)

    logger.info("Iterated through all chains.")
    assert merged_idata is not None, "Merged IData objects unsuccessful."
    n_chains_final = len(merged_idata["posterior"].coords["chain"])
    assert n_chains == n_chains_final, f"Expected {n_chains}, found {n_chains_final}."
    logger.info("Writing posterior samples to file.")
    PosteriorManager(id=cache_name, cache_dir=output_dir).put(merged_idata)
    logger.info("Finished writing posterior samples to file.")
    return None


if __name__ == "__main__":
    app()
