#!/usr/bin/env python3

"""CLI for standardized sampling from predefined PyMC3 models."""


from pathlib import Path
from time import time
from typing import Optional, Type

import numpy as np
import typer

from src.command_line_interfaces import cli_helpers
from src.command_line_interfaces.cli_helpers import ModelOption
from src.io import cache_io
from src.loggers import get_logger
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_model import SpecletModel

logger = get_logger()

cli_helpers.configure_pretty()


#### ---- File IO ---- ####


def make_cache_name(root_cache_dir: Path, name: str) -> Path:
    """Make a cache path.

    Args:
        name (str): Name of the model.

    Returns:
        Path: The path for the cache.
    """
    return root_cache_dir / name


def touch_file(cache_dir: Path, model: str, name: str) -> None:
    """Touch a file.

    Args:
        model (str): The model.
        name (str): The custom name of the model.
    """
    p = cache_dir / (model + "_" + name + ".txt")
    p.touch()
    return None


#### ---- Main ---- ####

PYMC3_CACHE_DIR = cache_io.default_cache_dir() / "pymc3_model_cache"


def sample_speclet_model(
    model: ModelOption,
    name: str,
    cores: int = 1,
    sample: bool = True,
    ignore_cache: bool = False,
    debug: bool = False,
    random_seed: Optional[int] = None,
    touch: bool = False,
) -> Type[SpecletModel]:
    """Fit and sample from a variety of predefined PyMC3 models.

    Args:
        model (ModelOption): The name of the model.
        name (str): Custom name for the model. This is useful for creating multiple
          caches of the same model.
        cores (int, optional): Number of compute cores. Defaults to 1.
        sample (bool, optional): Should the model be sampled? Defaults to True.
        ignore_cache (bool, optional): Should the cache be ignored? Defaults to False.
        debug (bool, optional): In debug mode? Defaults to False.
        random_seed (Optional[int], optional): Random seed. Defaults to None.
        touch (bool, optional): Should there be a file touched to indicate that the
          sampling process is complete? This is helpful for telling pipelines/workflows
          that this step is complete. Defaults to False.

    Returns:
        Type[SpecletModel]: An instance of the requested model with the PyMC3 model
          built and fit.
    """
    tic = time()

    name = cli_helpers.clean_model_names(name)
    cache_dir = make_cache_name(root_cache_dir=PYMC3_CACHE_DIR, name=name)
    logger.info(f"Cache directory: {cache_dir.as_posix()}")

    if random_seed:
        np.random.seed(random_seed)
    if debug:
        logger.info("Sampling in debug mode.")

    logger.info(f"Sampling '{model}' with custom name '{name}'")
    ModelClass = cli_helpers.get_model_class(model_opt=model)
    speclet_model = ModelClass(name=name, root_cache_dir=cache_dir, debug=debug)

    assert isinstance(speclet_model, SpecletModel)

    if model == ModelOption.crc_ceres_mimic and isinstance(speclet_model, CeresMimic):
        cli_helpers.modify_ceres_model_by_name(
            model=speclet_model, name=name, logger=logger
        )

    logger.info("Running model build method.")
    speclet_model.build_model()

    if sample:
        logger.info("Running ADVI fitting method.")
        _ = speclet_model.advi_sample_model(random_seed=random_seed)

    if touch:
        logger.info("Touching output file.")
        touch_file(cache_dir=cache_dir, model=model, name=name)

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return speclet_model


if __name__ == "__main__":
    typer.run(sample_speclet_model)
