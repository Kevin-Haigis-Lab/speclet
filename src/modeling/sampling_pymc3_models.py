#!/usr/bin/env python3

"""Standardized sampling from predefined PyMC3 models."""


from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
from pydantic import BaseModel
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.loggers import get_logger
from src.modeling import pymc3_sampling_api
from src.models import crc_models

logger = get_logger()

#### ---- Data Paths ---- ####


PYMC3_CACHE_DIR = Path("models", "model_cache") / pymc3_sampling_api.default_cache_dir

MODELING_DATA_DIR = Path("modeling_data")

DEPMAP_MODELING_DATA = MODELING_DATA_DIR / "depmap_modeling_dataframe.csv"
DEPMAP_SUBSAMPLE_DATA = MODELING_DATA_DIR / "depmap_modeling_dataframe_subsample.csv"

CRC_MODELING_DATA = MODELING_DATA_DIR / "depmap_CRC_data.csv"
CRC_SUBSAMPLING_DATA = MODELING_DATA_DIR / "depmap_CRC_data_subsample.csv"


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


#### ---- CRC Model Helpers ---- ####


def load_crc_data(debug: bool) -> pd.DataFrame:
    """Load CRC data.

    Args:
        debug (bool): In debug mode?

    Returns:
        pd.DataFrame: CRC Achilles data.
    """
    f = CRC_SUBSAMPLING_DATA if debug else CRC_MODELING_DATA
    return achelp.read_achilles_data(f, low_memory=False)


def crc_batch_size(debug: bool) -> int:
    """Decide on the minibatch size for a CRC data set.

    Args:
        debug (bool): In debug mode?

    Returns:
        int: Batch size.
    """
    if debug:
        return 1000
    else:
        return 10000


#### ---- Common sampling arguments model ---- ####


class SamplingArguments(BaseModel):
    """Organize arguments/parameters often used for sampling."""

    name: str
    sample: bool = True
    ignore_cache: bool = False
    cache_dir: Optional[Path] = None
    debug: bool = False
    random_seed: Optional[int] = None


#### ---- CRC Model 1 ---- ####

ReplacementsDict = Dict[TTShared, Union[pm.Minibatch, np.ndarray]]


def sample_crc_model1(
    model: pm.Model, args: SamplingArguments, replacements: ReplacementsDict
) -> None:
    """Sample CRC Model 1.

    Args:
        model (pm.Model): CRC Model 1.
        args (SamplingArguments): Arguments for the sampling method.
        replacements (ReplacementsDict): Variable replacements for sampling.
    """
    logger.info("Fitting and sampling 'crc-m1'.")
    _ = pymc3_sampling_api.pymc3_advi_approximation_procedure(
        model=model,
        n_iterations=100000,
        callbacks=[
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ],
        random_seed=args.random_seed,
        cache_dir=args.cache_dir,
        force=args.ignore_cache,
        fit_kwargs={"more_replacements": replacements},
    )


def crc_model1(
    sampling_args: SamplingArguments,
) -> Tuple[pm.Model, Dict[str, TTShared], pd.DataFrame]:
    """Build CRC Model 1.

    Args:
        sampling_args (SamplingArguments): Arguments to use for sampling.

    Returns:
        Tuple[pm.Model, Dict[str, TTShared], pd.DataFrame]: A collection of the generated model, shared variables, and the CRC Achilles data.
    """
    # Data
    logger.info("Loading CRC data.")
    data = load_crc_data(sampling_args.debug)

    batch_size = crc_batch_size(sampling_args.debug)
    logger.debug(f"Using batch size {batch_size}")

    # Indices
    indices_dict = achelp.common_indices(data)

    # Batched data
    sgrna_idx_batch = pm.Minibatch(indices_dict["sgrna_idx"], batch_size=batch_size)
    cellline_idx_batch = pm.Minibatch(
        indices_dict["cellline_idx"], batch_size=batch_size
    )
    batch_idx_batch = pm.Minibatch(indices_dict["batch_idx"], batch_size=batch_size)
    lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

    # Construct model
    logger.info("Building 'crc-m1' and shared variables.")
    crc_m1, shared_vars = crc_models.model_1(
        sgrna_idx=indices_dict["sgrna_idx"],
        sgrna_to_gene_idx=indices_dict["sgrna_to_gene_idx"],
        cellline_idx=indices_dict["cellline_idx"],
        batch_idx=indices_dict["batch_idx"],
        lfc_data=data.lfc.values,
    )

    data_replacements: ReplacementsDict = {
        shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
        shared_vars["cellline_idx_shared"]: cellline_idx_batch,
        shared_vars["batch_idx_shared"]: batch_idx_batch,
        shared_vars["lfc_shared"]: lfc_data_batch,
    }

    if sampling_args.sample:
        sample_crc_model1(
            model=crc_m1, args=sampling_args, replacements=data_replacements
        )

    logger.info("Finished building and sampling 'crc-m1'.")
    return crc_m1, shared_vars, data
