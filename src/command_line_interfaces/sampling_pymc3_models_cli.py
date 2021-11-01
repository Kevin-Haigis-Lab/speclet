#!/usr/bin/env python3

"""CLI for standardized sampling from predefined PyMC3 models."""


from pathlib import Path
from time import time
from typing import Any, Optional

import numpy as np
import typer

from src import model_configuration as model_config
from src.command_line_interfaces import cli_helpers
from src.loggers import logger
from src.models.speclet_model import SpecletModel
from src.project_enums import ModelFitMethod, SpecletPipeline, assert_never

cli_helpers.configure_pretty()


#### ---- Main ---- ####


def _update_sampling_kwargs(kwargs: dict[str, Any], chains: int, cores: int) -> None:
    sample_kwargs = kwargs.get("sample_kwargs", {})
    for key, value in {"chains": chains, "cores": cores}.items():
        if key in kwargs.keys():
            logger.warn(f"Overriding configured '{key}'.")
        sample_kwargs[key] = value
    kwargs["sample_kwargs"] = sample_kwargs
    return None


def sample_speclet_model(
    name: str,
    config_path: Path,
    fit_method: ModelFitMethod,
    cache_dir: Path,
    mcmc_chains: int = 4,
    mcmc_cores: int = 4,
    sample: bool = True,
    ignore_cache: bool = False,
    random_seed: Optional[int] = None,
    touch: Optional[Path] = None,
) -> SpecletModel:
    """Fit and sample from a variety of predefined PyMC3 models.

    Args:
        name (str): Custom name for the model. This is useful for creating multiple
          caches of the same model.
        config_path (Path): Path to the model configuration file.
        fit_method (ModelFitMethod, optional): Fitting method.
        cache_dir (Path): Root caching directory.
        mcmc_chains (int, optional): Number of MCMC chains to run. Defaults to 4.
        mcmc_cores (int, optional): Number of compute cores. Defaults to 4.
        sample (bool, optional): Should the model be sampled? Defaults to True.
        ignore_cache (bool, optional): Should the cache be ignored? Defaults to False.
        random_seed (Optional[int], optional): Random seed. Defaults to None.
        touch (bool, optional): Should there be a file touched to indicate that the
          sampling process is complete? This is helpful for telling pipelines/workflows
          that this step is complete. Defaults to False.

    Returns:
        SpecletModel: An instance of the requested model with the PyMC3 model
          built and fit.
    """
    tic = time()
    if random_seed:
        logger.info(f"Setting random seed ({random_seed}).")
        np.random.seed(random_seed)

    logger.info(f"Model config file: '{config_path.as_posix()}'.")
    logger.info(f"Root cache directory: {cache_dir.as_posix()}")
    sp_model = model_config.get_config_and_instantiate_model(
        config_path=config_path, name=name, root_cache_dir=cache_dir
    )
    logger.info(f"Sampling '{sp_model.__class__}' with name '{sp_model.name}'")
    logger.info(
        f"Model's cache directory: '{sp_model.cache_manager.cache_dir.as_posix()}'."
    )

    sampling_kwargs: dict[str, Any] = model_config.get_sampling_kwargs(
        config_path=config_path,
        name=name,
        pipeline=SpecletPipeline.FITTING,
        fit_method=fit_method,
    )
    _update_sampling_kwargs(sampling_kwargs, chains=mcmc_chains, cores=mcmc_cores)

    logger.info("Running model build method.")
    sp_model.build_model()

    if sample:
        if fit_method is ModelFitMethod.ADVI:
            logger.info("Running ADVI fitting method.")
            _ = sp_model.advi_sample_model(
                random_seed=random_seed, ignore_cache=ignore_cache, **sampling_kwargs
            )

        elif fit_method is ModelFitMethod.MCMC:
            logger.info("Running MCMC fitting method.")
            _ = sp_model.mcmc_sample_model(
                random_seed=random_seed,
                ignore_cache=ignore_cache,
                **sampling_kwargs,
            )
        else:
            assert_never(fit_method)

    if touch is not None:
        logger.info(f"Touching file: '{touch.as_posix()}'.")
        touch.touch()

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return sp_model


if __name__ == "__main__":
    typer.run(sample_speclet_model)
