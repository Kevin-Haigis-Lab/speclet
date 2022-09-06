#!/usr/bin/env python3

"""Command line interface for merging MCMC chains for a single model."""

import gc
from pathlib import Path
from typing import Callable

import arviz as az
from typer import Typer

from speclet.cli import cli_helpers
from speclet.exceptions import CacheDoesNotExistError
from speclet.loggers import logger, set_console_handler_level
from speclet.managers.cache_manager import PosteriorManager, get_posterior_cache_name
from speclet.model_configuration import get_configuration_for_model
from speclet.modeling import pymc_helpers
from speclet.project_enums import ModelFitMethod

set_console_handler_level("DEBUG")
cli_helpers.configure_pretty()
app = Typer()


class IncorrectNumberOfChainsError(BaseException):
    """Incorrect number of chains."""

    ...


# --- InferenceData modifiers ---

IDataModifier = Callable[[az.InferenceData], az.InferenceData]


def _del_posterior_predictive(idata: az.InferenceData) -> az.InferenceData:
    if hasattr(idata, "posterior_predictive"):
        logger.debug("Removing posterior predictive distribution.")
        del idata.posterior_predictive
    return idata


def _del_observed_data(idata: az.InferenceData) -> az.InferenceData:
    if hasattr(idata, "observed_data"):
        logger.debug("Removing observed data.")
        del idata.observed_data
    return idata


def _del_posterior(idata: az.InferenceData) -> az.InferenceData:
    if hasattr(idata, "posterior"):
        logger.debug("Removing posterior data.")
        del idata.posterior
    return idata


def _del_posterior_predictive_and_observed_data(
    idata: az.InferenceData,
) -> az.InferenceData:
    idata = _del_posterior_predictive(idata)
    idata = _del_observed_data(idata)
    return idata


def _del_posterior_and_observed_data(idata: az.InferenceData) -> az.InferenceData:
    idata = _del_posterior(idata)
    idata = _del_observed_data(idata)
    return idata


# --- Combine chains ---


def _thin_posterior(trace: az.InferenceData, by: int | None) -> az.InferenceData:
    if hasattr(trace, "posterior") and by is not None and 1 < by:
        return pymc_helpers.thin_posterior(trace, by)
    return trace


def _thin_posterior_predictive(
    trace: az.InferenceData, by: int | None
) -> az.InferenceData:
    if hasattr(trace, "posterior_predictive") and by is not None and 1 < by:
        return pymc_helpers.thin_posterior_predictive(trace, by)
    return trace


def _combine_mcmc_chains(
    cache_name: str,
    cache_dir: Path,
    n_chains: int,
    chain_mod_fxn: IDataModifier | None = None,
    thin_posterior: int | None = None,
    thin_post_pred: int | None = None,
) -> az.InferenceData:
    all_idata_objects: list[az.InferenceData] = []
    for chain_i in range(n_chains):
        logger.info(f"Gather MCMC chain #{chain_i}.")
        chain_id = f"{cache_name}_chain{chain_i}"
        post_man = PosteriorManager(id=chain_id, cache_dir=cache_dir)
        if post_man.posterior_cache_exists:
            idata = post_man.get_posterior()
            assert idata is not None, "Failed to get data."
            if chain_mod_fxn is not None:
                idata = chain_mod_fxn(idata)
            idata = _thin_posterior(idata, thin_posterior)
            idata = _thin_posterior_predictive(idata, thin_post_pred)
            all_idata_objects.append(idata)
        else:
            logger.error(f"Cache for chain #{chain_i} '{chain_id}' does not exist.")
            raise CacheDoesNotExistError(post_man.posterior_path)

    if len(all_idata_objects) != n_chains:
        msg = f"Expected {n_chains} but found {len(all_idata_objects)}."
        logger.error(msg)
        raise IncorrectNumberOfChainsError(msg)
    else:
        logger.info(f"Collected {len(all_idata_objects)} chains.")

    logger.info("Merging all chains.")
    combined_chains = az.concat(all_idata_objects, dim="chain")
    assert isinstance(combined_chains, az.InferenceData), "Failed to combine chains."
    return combined_chains


@app.command()
def combine_mcmc_chains(
    name: str,
    fit_method: ModelFitMethod,
    n_chains: int,
    config_path: Path,
    cache_dir: Path,
    output_dir: Path,
    thin_posterior: int = 1,
    thin_post_pred: int = 1,
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
    assert model_config is not None, "Model configuration not found."
    model_config.split_posterior_when_combining_chains

    cache_name = get_posterior_cache_name(model_name=name, fit_method=fit_method)
    if not model_config.split_posterior_when_combining_chains:
        combined_chains = _combine_mcmc_chains(
            cache_name=cache_name, cache_dir=cache_dir, n_chains=n_chains
        )
        logger.info("Writing posterior data to file.")
        PosteriorManager(id=cache_name, cache_dir=output_dir).put_posterior(
            combined_chains
        )
        logger.info("Finished writing posterior data to file.")
    else:
        logger.info("Merging and saving posterior and post. pred. separately.")

        # Merge and save posterior distribution data.
        logger.info("Merging and saving posterior.")
        combined_chains_posterior = _combine_mcmc_chains(
            cache_name=cache_name,
            cache_dir=cache_dir,
            n_chains=n_chains,
            chain_mod_fxn=_del_posterior_predictive_and_observed_data,
        )
        logger.info("Writing posterior data to file.")
        PosteriorManager(id=cache_name, cache_dir=output_dir).put_posterior(
            combined_chains_posterior
        )
        logger.info("Finished writing posterior data to file.")

        # Free up memory.
        logger.debug("Deleting posterior object.")
        del combined_chains_posterior
        logger.debug("Calling garbage collection.")
        gc.collect()

        # Merge and save posterior predictive distribution data.
        logger.info("Merging and saving posterior predictive.")
        combined_chains_posterior = _combine_mcmc_chains(
            cache_name=cache_name,
            cache_dir=cache_dir,
            n_chains=n_chains,
            chain_mod_fxn=_del_posterior_and_observed_data,
        )
        logger.info("Writing posterior predictive data to file.")
        PosteriorManager(id=cache_name, cache_dir=output_dir).put_posterior_predictive(
            combined_chains_posterior
        )
        logger.info("Finished writing posterior predictive data to file.")

    return None


if __name__ == "__main__":
    app()
