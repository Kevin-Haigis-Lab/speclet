#!/usr/bin/env python3

"""CLI for standardized fitting Bayesian models."""


from pathlib import Path
from time import time
from typing import Optional

import pandas as pd
import typer

from speclet import io
from speclet import model_configuration as model_config
from speclet.bayesian_models import get_bayesian_model
from speclet.command_line_interfaces import cli_helpers
from speclet.loggers import logger
from speclet.managers.cache_manager import cache_posterior
from speclet.model_configuration import ModelingSamplingArguments
from speclet.modeling.model_fitting_api import fit_model
from speclet.project_enums import ModelFitMethod

cli_helpers.configure_pretty()


#### ---- Main ---- ####


def read_crispr_screen_data(file: io.DataFile) -> pd.DataFrame:
    """Read in CRISPR screen data.

    # TODO: move this to a better place once the data manager system is figured out.

    Args:
        file (io.DataFile): Path to the data file.

    Returns:
        pd.DataFrame: Pandas dataframe of the CRISPR screen data.
    """
    path = io.data_path(file)
    return pd.read_csv(path)


def _augment_sampling_kwargs(
    sampling_kwargs: Optional[ModelingSamplingArguments],
    mcmc_chains: int,
    mcmc_cores: int,
) -> Optional[ModelingSamplingArguments]:
    if sampling_kwargs is None:
        return None
    if sampling_kwargs.stan_mcmc is not None:
        sampling_kwargs.stan_mcmc.num_chains = mcmc_chains
    if sampling_kwargs.pymc3_mcmc is not None:
        sampling_kwargs.pymc3_mcmc.chains = mcmc_chains
        sampling_kwargs.pymc3_mcmc.cores = mcmc_cores
    return sampling_kwargs


def fit_bayesian_model(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    mcmc_chains: int = 4,
    mcmc_cores: int = 4,
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
    """
    tic = time()

    config = model_config.get_configuration_for_model(
        config_path=config_path, name=name
    )
    assert config is not None
    data = read_crispr_screen_data(config.data_file)
    model = get_bayesian_model(config.model)()

    sampling_kwargs_adj = _augment_sampling_kwargs(
        config.sampling_kwargs, mcmc_chains=mcmc_chains, mcmc_cores=mcmc_cores
    )
    posterior = fit_model(
        model=model,
        data=data,
        fit_method=fit_method,
        sampling_kwargs=sampling_kwargs_adj,
    )
    cache_posterior(
        posterior=posterior, name=name, fit_method=fit_method, cache_dir=cache_dir
    )

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    typer.run(fit_bayesian_model)
