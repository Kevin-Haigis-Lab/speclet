#!/usr/bin/env python3

"""CLI for sampling from predefined PyMC3 models."""


from enum import Enum
from time import time
from typing import Optional

import numpy as np
import pretty_errors
import typer

from src.logging.loggers import logger
from src.modeling import sampling_pymc3_models as sampling
from src.modeling.sampling_pymc3_models import SamplingArguments

#### ---- Pretty Errors ---- ####


pretty_errors.configure(
    filename_color=pretty_errors.BLUE,
    code_color=pretty_errors.BLACK,
    exception_color=pretty_errors.BRIGHT_RED,
    exception_arg_color=pretty_errors.RED,
    line_color=pretty_errors.BRIGHT_BLACK,
)


#### ---- Main ---- ####


class ModelOption(str, Enum):
    """Models that are available for sampling."""

    crc_m1 = "crc-m1"


def main(
    model: ModelOption,
    name: str,
    sample: bool = True,
    ignore_cache: bool = False,
    debug: bool = False,
    random_seed: Optional[int] = None,
    touch: bool = False,
) -> None:
    """Fit and sample from a variety of predefined PyMC3 models.

    Args:
        model (ModelOption): The name of the model.
        name (str): Custom name for the model. This is useful for creating multiple caches of the same model.
        sample (bool, optional): Should the model be sampled? Defaults to True.
        ignore_cache (bool, optional): Should the cache be ignored? Defaults to False.
        debug (bool, optional): In debug mode? Defaults to False.
        random_seed (Optional[int], optional): Random seed. Defaults to None.
        touch (bool, optional): Should there be a file touched to indicate that the sampling process is complete? This is helpful for telling pipelines/workflows that this step is complete. Defaults to False.

    Raises:
        Exception: The model from the user is not recognized.

    Returns:
        [None]: None
    """
    tic = time()

    name = sampling.clean_model_names(name)
    cache_dir = sampling.make_cache_name(name)
    sampling_args = SamplingArguments(
        name=name,
        sample=sample,
        ignore_cache=ignore_cache,
        debug=debug,
        random_seed=random_seed,
        cache_dir=cache_dir,
    )

    logger.debug(f"Cache directory: {cache_dir.as_posix()}")

    if random_seed:
        np.random.seed(random_seed)

        if debug:
            logger.debug("Sampling in debug mode.")

    if model == ModelOption.crc_m1:
        logger.info(f"Sampling '{model}' with custom name '{name}'")
        _ = sampling.crc_model1(sampling_args=sampling_args)
    else:
        logger.error("Unknown model: '{model}'")
        raise Exception("Unrecognized model 🤷🏻‍♂️")

    if touch:
        logger.info("Touching output file.")
        sampling.touch_file(model, name)

    toc = time()
    logger.info(f"finished; execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    typer.run(main)
