#!/usr/bin/env python3

"""Standardized sampling from predefined PyMC3 models."""


from pathlib import Path
from typing import Optional

import pretty_errors
from pydantic import BaseModel

from src.io import cache_io

#### ---- Data Paths ---- ####

# TODO: Move this responsibility to 'io/' module.
PYMC3_CACHE_DIR = cache_io.default_cache_dir()


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


def make_cache_name(name: str) -> Path:
    """Make a cache path.

    Args:
        name (str): Name of the model.

    Returns:
        Path: The path for the cache.
    """
    return PYMC3_CACHE_DIR / name


def touch_file(model: str, name: str) -> None:
    """Touch a file.

    Args:
        model (str): The model.
        name (str): The custom name of the model.
    """
    p = make_cache_name(name) / (model + "_" + name + ".txt")
    p.touch()
    return None


#### ---- Common sampling arguments model ---- ####


class SamplingArguments(BaseModel):
    """Organize arguments/parameters often used for sampling."""

    name: str
    cores: int = 1
    sample: bool = True
    ignore_cache: bool = False
    cache_dir: Optional[Path] = None
    debug: bool = False
    random_seed: Optional[int] = None
