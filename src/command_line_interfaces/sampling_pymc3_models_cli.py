#!/usr/bin/env python3

"""CLI for standardized sampling from predefined PyMC3 models."""


from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import typer

from src.command_line_interfaces import cli_helpers
from src.io import cache_io
from src.loggers import logger
from src.models.configuration import instantiate_and_configure_model
from src.models.speclet_model import SpecletModel
from src.project_enums import ModelFitMethod, ModelOption

cli_helpers.configure_pretty()


#### ---- Main ---- ####

PYMC3_CACHE_DIR = cache_io.default_cache_dir()


def sample_speclet_model(
    model: ModelOption,
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod = ModelFitMethod.ADVI,
    mcmc_chains: int = 4,
    mcmc_cores: int = 4,
    sample: bool = True,
    ignore_cache: bool = False,
    debug: bool = False,
    random_seed: Optional[int] = None,
    touch: Optional[Path] = None,
) -> SpecletModel:
    """Fit and sample from a variety of predefined PyMC3 models.

    Args:
        model (ModelOption): The name of the model.
        name (str): Custom name for the model. This is useful for creating multiple
          caches of the same model.
        config_path (Path): Path to the model configuration file.
        fit_method (ModelFitMethod, optional): Fitting method. Defaults to
          ModelFitMethod.ADVI.
        mcmc_chains (int, optional): Number of MCMC chains to run. Defaults to 4.
        mcmc_cores (int, optional): Number of compute cores. Defaults to 4.
        sample (bool, optional): Should the model be sampled? Defaults to True.
        ignore_cache (bool, optional): Should the cache be ignored? Defaults to False.
        debug (bool, optional): In debug mode? Defaults to False.
        random_seed (Optional[int], optional): Random seed. Defaults to None.
        touch (bool, optional): Should there be a file touched to indicate that the
          sampling process is complete? This is helpful for telling pipelines/workflows
          that this step is complete. Defaults to False.

    Returns:
        SpecletModel: An instance of the requested model with the PyMC3 model
          built and fit.
    """
    tic = time()

    name = cli_helpers.clean_model_names(name)
    logger.info(f"Cache directory: {PYMC3_CACHE_DIR.as_posix()}")

    if random_seed:
        np.random.seed(random_seed)
    if debug:
        logger.info("Sampling in debug mode.")

    logger.info(f"Sampling '{model}' with custom name '{name}'")
    speclet_model = instantiate_and_configure_model(
        model_opt=model,
        name=name,
        root_cache_dir=PYMC3_CACHE_DIR,
        debug=debug,
        config_path=config_path,
    )

    logger.info("Running model build method.")
    speclet_model.build_model()

    if sample:
        if fit_method == ModelFitMethod.ADVI:
            logger.info("Running ADVI fitting method.")
            _ = speclet_model.advi_sample_model(
                random_seed=random_seed,
                ignore_cache=ignore_cache,
            )

        elif fit_method == ModelFitMethod.MCMC:
            logger.info("Running MCMC fitting method.")
            _ = speclet_model.mcmc_sample_model(
                chains=mcmc_chains,
                cores=mcmc_cores,
                random_seed=random_seed,
                ignore_cache=ignore_cache,
            )
        else:
            raise Exception(f"Unknown fit method '{fit_method.value}'.")

    if touch is not None:
        logger.info(f"Touching file: '{touch.as_posix()}'.")
        touch.touch()

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return speclet_model


if __name__ == "__main__":
    typer.run(sample_speclet_model)
