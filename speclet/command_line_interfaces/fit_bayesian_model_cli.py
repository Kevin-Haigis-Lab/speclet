#!/usr/bin/env python3

"""CLI for standardized fitting Bayesian models."""


from pathlib import Path
from time import time
from typing import Optional

import pandas as pd
from typer import Typer

from speclet import io
from speclet import model_configuration as model_config
from speclet.bayesian_models import get_bayesian_model
from speclet.command_line_interfaces import cli_helpers
from speclet.loggers import logger
from speclet.managers.cache_manager import cache_posterior, get_posterior_cache_name
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.model_configuration import ModelingSamplingArguments
from speclet.modeling.fitting_arguments import (
    Pymc3SampleArguments,
    StanMCMCSamplingArguments,
)
from speclet.modeling.model_fitting_api import fit_model
from speclet.project_enums import ModelFitMethod

cli_helpers.configure_pretty()
app = Typer()

#### ---- Main ---- ####


def _read_crispr_screen_data(file: io.DataFile) -> pd.DataFrame:
    """Read in CRISPR screen data."""
    return CrisprScreenDataManager(data_file=file).get_data()


def _augment_sampling_kwargs(
    sampling_kwargs: Optional[ModelingSamplingArguments],
    mcmc_chains: int,
    mcmc_cores: int,
) -> Optional[ModelingSamplingArguments]:
    if sampling_kwargs is None:
        sampling_kwargs = ModelingSamplingArguments()

    if sampling_kwargs.stan_mcmc is not None:
        sampling_kwargs.stan_mcmc.num_chains = mcmc_chains
    else:
        sampling_kwargs.stan_mcmc = StanMCMCSamplingArguments(num_chains=mcmc_chains)

    if sampling_kwargs.pymc3_mcmc is not None:
        sampling_kwargs.pymc3_mcmc.chains = mcmc_chains
        sampling_kwargs.pymc3_mcmc.cores = mcmc_cores
    else:
        sampling_kwargs.pymc3_mcmc = Pymc3SampleArguments(
            chains=mcmc_chains, cores=mcmc_cores
        )

    return sampling_kwargs


@app.command()
def fit_bayesian_model(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    mcmc_chains: int = 4,
    mcmc_cores: int = 4,
    cache_name: Optional[str] = None,
) -> None:
    """Sample a Bayesian model.

    The parameters for the MCMC cores and chains is because I often fit the chains in
    separate jobs to help with memory management.

    Args:
        name (str): Name of the model configuration.
        config_path (Path): Path to the configuration file.
        fit_method (ModelFitMethod): Model fitting method to use.
        cache_dir (Path): Directory in which to cache the results.
        mcmc_chains (int, optional): Number of MCMC chains. Defaults to 4.
        mcmc_cores (int, optional): Number of MCMC cores. Defaults to 4.
        cache_name (Optional[str], optional): A specific name to use for the posterior
        cache ID. Defaults to None which results in using the `name` for the cache name.
    """
    tic = time()
    logger.info("Reading model configuration.")
    config = model_config.get_configuration_for_model(
        config_path=config_path, name=name
    )
    assert config is not None
    logger.info("Loading data.")
    data = _read_crispr_screen_data(config.data_file)
    logger.info("Retrieving Bayesian model object.")
    model = get_bayesian_model(config.model)()

    logger.info("Augmenting sampling kwargs (MCMC chains and cores).")
    sampling_kwargs_adj = _augment_sampling_kwargs(
        config.sampling_kwargs,
        mcmc_chains=mcmc_chains,
        mcmc_cores=mcmc_cores,
    )
    logger.info("Sampling model.")
    trace = fit_model(
        model=model,
        data=data,
        fit_method=fit_method,
        sampling_kwargs=sampling_kwargs_adj,
    )

    logger.info("Sampling finished.")
    print(trace.posterior.data_vars)

    if cache_name is None:
        logger.warn("No cache name provided - one will be generated automatically.")
        cache_name = get_posterior_cache_name(model_name=name, fit_method=fit_method)

    logger.info(f"Caching posterior data: '{str(cache_name)}'")
    cache_posterior(trace, id=cache_name, cache_dir=cache_dir)

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    app()
