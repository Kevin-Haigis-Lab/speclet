#!/usr/bin/env python3

"""Standardized sampling from predefined PyMC3 models (with CLI)."""


from enum import Enum
from pathlib import Path
from time import time
from typing import Optional, Type

import numpy as np
import pretty_errors
import typer

from src.io import cache_io
from src.loggers import get_logger
from src.modeling.sampling_metadata_models import SamplingArguments
from src.models.crc_ceres_mimic_one import CrcCeresMimic
from src.models.crc_model_one import CrcModelOne
from src.models.speclet_model import SpecletModel

logger = get_logger()

#### ---- Pretty Errors ---- ####


pretty_errors.configure(
    filename_color=pretty_errors.BLUE,
    code_color=pretty_errors.BLACK,
    exception_color=pretty_errors.BRIGHT_RED,
    exception_arg_color=pretty_errors.RED,
    line_color=pretty_errors.BRIGHT_BLACK,
)


#### ---- General ---- ####


def clean_model_names(n: str) -> str:
    """Clean a custom model name.

    Args:
        n (str): Custom model name.

    Returns:
        str: Cleaned model name.
    """
    return n.replace(" ", "-")


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


class ModelOption(str, Enum):
    """Models that are available for sampling."""

    crc_m1 = "crc_m1"
    crc_ceres_mimic = "crc_ceres-mimic"


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

    Raises:
        Exception: The model from the user is not recognized.

    Returns:
        [None]: None
    """
    tic = time()

    name = clean_model_names(name)
    cache_dir = make_cache_name(root_cache_dir=PYMC3_CACHE_DIR, name=name)
    sampling_args = SamplingArguments(
        name=name,
        cores=cores,
        sample=sample,
        ignore_cache=ignore_cache,
        debug=debug,
        random_seed=random_seed,
    )

    logger.debug(f"Cache directory: {cache_dir.as_posix()}")

    if random_seed:
        np.random.seed(random_seed)

        if debug:
            logger.debug("Sampling in debug mode.")

    if model == ModelOption.crc_m1:
        logger.info(f"Sampling '{model}' with custom name '{name}'")
        speclet_model = CrcModelOne(name=name, root_cache_dir=cache_dir, debug=debug)
        logger.debug("Running model build method.")
        speclet_model.build_model()
        logger.debug("Running ADVI fitting method.")
        _ = speclet_model.advi_sample_model(sampling_args=sampling_args)
    elif model == ModelOption.crc_ceres_mimic:
        logger.info(f"Sampling '{model}' with custom name '{name}'")
        speclet_model = CrcCeresMimic(name=name, root_cache_dir=cache_dir, debug=debug)
        logger.debug("Running model build method.")
        speclet_model.build_model()
        logger.debug("Running ADVI fitting method.")
        _ = speclet_model.advi_sample_model(sampling_args=sampling_args)
    else:
        logger.error("Unknown model: '{model}'")
        raise Exception("Unrecognized model 🤷🏻‍♂️")

    if touch:
        logger.info("Touching output file.")
        touch_file(cache_dir=cache_dir, model=model, name=name)

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return speclet_model


if __name__ == "__main__":
    typer.run(sample_speclet_model)
