#!/usr/bin/env python3

from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union

import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pymc3_sampling_api
import typer
from colorama import Back, Fore, Style, init
from pydantic import BaseModel
from pymc3_models import crc_models
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

init(autoreset=True)


pretty_errors.configure(
    filename_color=pretty_errors.BLUE,
    code_color=pretty_errors.BLACK,
    exception_color=pretty_errors.BRIGHT_RED,
    exception_arg_color=pretty_errors.RED,
    line_color=pretty_errors.BRIGHT_BLACK,
)

#### ---- Data Paths ---- ####


PYMC3_CACHE_DIR = Path("analysis") / pymc3_sampling_api.default_cache_dir

MODELING_DATA_DIR = Path("modeling_data")

DEPMAP_MODELING_DATA = MODELING_DATA_DIR / "depmap_modeling_dataframe.csv"
DEPMAP_SUBSAMPLE_DATA = MODELING_DATA_DIR / "depmap_modeling_dataframe_subsample.csv"

CRC_MODELING_DATA = MODELING_DATA_DIR / "depmap_CRC_data.csv"
CRC_SUBSAMPLING_DATA = MODELING_DATA_DIR / "depmap_CRC_data_subsample.csv"


#### ---- General ---- ####


def print_model(n: str):
    print(Fore.WHITE + Back.BLUE + " " + n + " ")
    return None


def info(m: str):
    print(Fore.BLACK + Style.DIM + m)
    return None


def done():
    print(Fore.GREEN + Style.BRIGHT + "Done")
    return None


#### ---- File IO ---- ####


def make_cache_name(name: str) -> Path:
    return PYMC3_CACHE_DIR / name


def touch_file(n: str) -> None:
    p = make_cache_name(n) / (n + ".txt")
    p.touch()
    return None


#### ---- CRC Model Helpers ---- ####


def load_crc_data(debug: bool) -> pd.DataFrame:
    if debug:
        f = CRC_SUBSAMPLING_DATA
    else:
        f = CRC_MODELING_DATA
    return dphelp.read_achilles_data(f, low_memory=False)


def crc_batch_size(debug: bool) -> int:
    if debug:
        return 1000
    else:
        return 10000


#### ---- Common sampling arguments model ---- ####


class SamplingArguments(BaseModel):
    name: str
    force_sampling: bool
    cache_dir: Path
    debug: bool
    random_seed: Optional[int]


#### ---- CRC Model 1 ---- ####


def crc_model1(sampling_args: SamplingArguments) -> None:
    print_model("CRC Model 1")

    # Data
    info("Loading data...")
    data = load_crc_data(sampling_args.debug)

    batch_size = crc_batch_size(sampling_args.debug)

    # Indices
    indices_dict = dphelp.common_indices(data)

    # Batched data
    sgrna_idx_batch = pm.Minibatch(indices_dict["sgrna_idx"], batch_size=batch_size)
    cellline_idx_batch = pm.Minibatch(
        indices_dict["cellline_idx"], batch_size=batch_size
    )
    batch_idx_batch = pm.Minibatch(indices_dict["batch_idx"], batch_size=batch_size)
    lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

    # Construct model
    crc_m1, shared_vars = crc_models.model_1(
        sgrna_idx=indices_dict["sgrna_idx"],
        sgrna_to_gene_idx=indices_dict["sgrna_to_gene_idx"],
        cellline_idx=indices_dict["cellline_idx"],
        batch_idx=indices_dict["batch_idx"],
        lfc_data=data.lfc.values,
    )

    _ = pymc3_sampling_api.pymc3_advi_approximation_procedure(
        model=crc_m1,
        n_iterations=100000,
        callbacks=[
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ],
        random_seed=sampling_args.random_seed,
        cache_dir=sampling_args.cache_dir,
        force=sampling_args.force_sampling,
        fit_kwargs={
            "more_replacements": {
                shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
                shared_vars["cellline_idx_shared"]: cellline_idx_batch,
                shared_vars["batch_idx_shared"]: batch_idx_batch,
                shared_vars["lfc_shared"]: lfc_data_batch,
            }
        },
    )

    done()
    return


#### ---- MAIN ---- ####


class ModelOption(str, Enum):
    crc_m1 = "crc-m1"


def clean_model_names(n: str) -> str:
    return n.replace(" ", "-")


def main(
    model: ModelOption,
    name: str,
    force_sample: bool = False,
    debug: bool = False,
    random_seed: Optional[int] = None,
    touch: bool = False,
):
    tic = time()

    name = clean_model_names(name)
    cache_dir = make_cache_name(name)
    sampling_args = SamplingArguments(
        name=name,
        force_sample=force_sample,
        debug=debug,
        random_seed=random_seed,
        cache_dir=cache_dir,
    )

    if random_seed:
        np.random.seed(random_seed)

    if debug:
        print(Fore.RED + "(🪲 debug mode)")

    if model == ModelOption.crc_m1:
        crc_model1(sampling_args=sampling_args)
    else:
        raise Exception("Unrecognized model 🤷🏻‍♂️")

    if touch:
        info("Touching output file.")
        touch_file(name)

    toc = time()
    info(f"execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    typer.run(main)
