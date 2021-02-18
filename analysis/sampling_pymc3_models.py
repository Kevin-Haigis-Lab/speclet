#!/bin/env python3

import argparse
import warnings
from pathlib import Path
from time import time
from typing import Optional

import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pymc3_sampling_api
from colorama import Back, Fore, Style, init
from pymc3_models import ceres_models, crc_models

init(autoreset=True)

#### ---- Data Paths ---- ####

PYMC3_CACHE_DIR = Path("analysis") / pymc3_sampling_api.default_cache_dir

MODELING_DATA_DIR = Path("modeling_data")

DEPMAP_MODELING_DATA = MODELING_DATA_DIR / "depmap_modeling_dataframe.csv"
DEPMAP_SUBSAMPLE_DATA = MODELING_DATA_DIR / "depmap_modeling_dataframe_subsample.csv"

CRC_MODELING_DATA = MODELING_DATA_DIR / "depmap_CRC_data.csv"


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


#### ---- CERES Model 1 ---- ####


def ceres_model1(
    name: str,
    debug: bool = False,
    force_sampling: bool = False,
    random_seed: Optional[int] = None,
) -> None:
    print_model("CERES Model 1")

    # Data
    info("Loading data...")
    if debug:
        data_path = DEPMAP_SUBSAMPLE_DATA
    else:
        data_path = DEPMAP_MODELING_DATA

    data = dphelp.read_achilles_data(data_path)

    # Indices
    sgrna_idx = dphelp.get_indices(data, "sgrna")
    gene_idx = dphelp.get_indices(data, "hugo_symbol")
    cell_idx = dphelp.get_indices(data, "depmap_id")

    # Construct model
    ceres_m1 = ceres_models.construct_ceres_m1(
        sgrna_idx=sgrna_idx,
        gene_idx=gene_idx,
        cell_idx=cell_idx,
        cn_data=data.z_log2_cn.to_numpy(),
        lfc_data=data.lfc.to_numpy(),
    )

    ceres_m1_cache = PYMC3_CACHE_DIR / name
    _ = pymc3_sampling_api.pymc3_sampling_procedure(
        model=ceres_m1,
        num_mcmc=1000,
        tune=1000,
        chains=2,
        cores=2,
        random_seed=random_seed,
        cache_dir=ceres_m1_cache,
        force=force_sampling,
        sample_kwargs={
            "init": "advi+adapt_diag",
            "n_init": 50000,
            "target_accept": 0.9,
        },
    )

    done()


#### ---- CERES Model 2 ---- ####


def ceres_model2(
    name: str,
    debug: bool = False,
    force_sampling: bool = False,
    random_seed: Optional[int] = None,
) -> None:
    print_model("CERES Model 2")

    # Data
    info("Loading data...")
    if debug:
        data_path = DEPMAP_SUBSAMPLE_DATA
    else:
        data_path = DEPMAP_MODELING_DATA

    data = dphelp.read_achilles_data(data_path)

    # Indices
    sgrna_idx = dphelp.get_indices(data, "sgrna")
    gene_idx = dphelp.get_indices(data, "hugo_symbol")
    cell_idx = dphelp.get_indices(data, "depmap_id")

    # Construct model
    ceres_m2 = ceres_models.construct_ceres_m2(
        sgrna_idx=sgrna_idx,
        gene_idx=gene_idx,
        cell_idx=cell_idx,
        cn_data=data.z_log2_cn.to_numpy(),
        lfc_data=data.lfc.to_numpy(),
    )

    ceres_m2_cache = PYMC3_CACHE_DIR / name
    _ = pymc3_sampling_api.pymc3_sampling_procedure(
        model=ceres_m2,
        num_mcmc=1000,
        tune=1000,
        chains=2,
        cores=2,
        random_seed=random_seed,
        cache_dir=ceres_m2_cache,
        force=force_sampling,
        sample_kwargs={
            "init": "advi+adapt_diag",
            "n_init": 100000,
            "target_accept": 0.9,
        },
    )

    done()


#### ---- CRC Model 1 ---- ####


def crc_model1(
    name: str,
    debug: bool = False,
    force_sampling: bool = False,
    random_seed: Optional[int] = None,
) -> None:
    print_model("CRC Model 1")

    # Data
    info("Loading data...")
    data = pd.DataFrame()
    if debug:
        data = dphelp.read_achilles_data(
            CRC_MODELING_DATA, low_memory=False, set_categorical_cols=False
        )
        info("Subsampling data...")
        data = dphelp.subsample_achilles_data(data)
        data = dphelp.set_achilles_categorical_columns(data)
    else:
        data = dphelp.read_achilles_data(CRC_MODELING_DATA, low_memory=False)

    batch_size = -1
    if debug:
        batch_size = 1000
    else:
        batch_size = 10000

    # Indices
    gene_idx = dphelp.get_indices(data, "hugo_symbol")

    # Construct model
    crc_m1 = crc_models.model_1(
        gene_idx=gene_idx, lfc_data=data.lfc.values, batch_size=batch_size
    )

    # Sample and cache
    crc_m1_cache = PYMC3_CACHE_DIR / name

    _ = pymc3_sampling_api.pymc3_sampling_procedure(
        model=crc_m1,
        num_mcmc=2000,
        tune=2000,
        chains=2,
        cores=2,
        random_seed=random_seed,
        cache_dir=crc_m1_cache,
        force=force_sampling,
        sample_kwargs={
            "init": "advi+adapt_diag",
            "n_init": 200000,
            "target_accept": 0.9,
        },
    )

    done()


#### ---- Finish ---- ####


def parse_cli_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "-m",
        "--model",
        help="model to sample from",
        type=str,
        choices=["ceres-m1", "ceres-m2", "crc-m1", "crc-m2"],
    )
    parser.add_argument("-n", "--name", help="model name", type=str)
    parser.add_argument(
        "--force-sample",
        help="ignore cached results and sample from model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d", "--debug", help="debug mode", action="store_true", default=False
    )
    parser.add_argument(
        "-s",
        "--random-seed",
        help="random seed for all processes",
        type=int,
        nargs="?",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    args = parse_cli_arguments(parser)

    tic = time()

    np.random.seed(args.random_seed)

    if args.debug:
        print(Fore.RED + "(🪲 debug mode)")

    if args.model == "ceres-m1":
        ceres_model1(
            name=args.name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "ceres-m2":
        ceres_model2(
            name=args.name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "crc-m1":
        crc_model1(
            name=args.name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "crc-m2":
        warnings.warn("CRC model 2 is not yet implemented 😥")
    else:
        warnings.warn("Unrecognized model 🤷🏻‍♂️")

    toc = time()
    info(f"execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    main()
