#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from time import time
from typing import Any, List, Optional

import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pymc3_sampling_api
from colorama import Back, Fore, Style, init
from pymc3_models import ceres_models, crc_models

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


def touch(n: str) -> None:
    p = make_cache_name(n) / (n + ".txt")
    p.touch()
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

    ceres_m1_cache = make_cache_name(name)
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
    return


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

    ceres_m2_cache = make_cache_name(name)
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
    return


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


def make_sgrna_to_gene_mapping_df(
    data: pd.DataFrame, sgrna_col: str = "sgrna", gene_col: str = "hugo_symbol"
) -> pd.DataFrame:
    return (
        data[[sgrna_col, gene_col]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values(sgrna_col)
        .reset_index(drop=True)
    )


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
    data = load_crc_data(debug)

    batch_size = crc_batch_size(debug)

    # Indices
    gene_idx = dphelp.get_indices(data, "hugo_symbol")

    # Batched data
    gene_idx_batch = pm.Minibatch(gene_idx, batch_size=batch_size)
    lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

    # Construct model
    crc_m1, gene_idx_shared, lfc_data_shared = crc_models.model_1(
        gene_idx=gene_idx, lfc_data=data.lfc.values
    )

    # Sample and cache
    crc_m1_cache = make_cache_name(name)

    _ = pymc3_sampling_api.pymc3_advi_approximation_procedure(
        model=crc_m1,
        n_iterations=100000,
        callbacks=[
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ],
        random_seed=random_seed,
        cache_dir=crc_m1_cache,
        force=force_sampling,
        fit_kwargs={
            "more_replacements": {
                gene_idx_shared: gene_idx_batch,
                lfc_data_shared: lfc_data_batch,
            }
        },
    )

    done()
    return


#### ---- CRC Model 2 ---- ####


def crc_model2(
    name: str,
    debug: bool = False,
    force_sampling: bool = False,
    random_seed: Optional[int] = None,
) -> None:
    print_model("CRC Model 2")

    # Data
    info("Loading data...")
    data = load_crc_data(debug)

    batch_size = crc_batch_size(debug)

    # Indices
    sgrna_idx = dphelp.get_indices(data, "sgrna")
    sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(data)
    sgrna_to_gene_idx = dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol")

    # Batched data
    sgrna_idx_batch = pm.Minibatch(sgrna_idx, batch_size=batch_size)
    lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

    # Construct model
    crc_m2, shared_vars = crc_models.model_2(
        sgrna_idx=sgrna_idx,
        sgrna_to_gene_idx=sgrna_to_gene_idx,
        lfc_data=data.lfc.values,
    )

    # Sample and cache
    crc_m2_cache = make_cache_name(name)

    _ = pymc3_sampling_api.pymc3_advi_approximation_procedure(
        model=crc_m2,
        n_iterations=100000,
        callbacks=[
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ],
        random_seed=random_seed,
        cache_dir=crc_m2_cache,
        force=force_sampling,
        fit_kwargs={
            "more_replacements": {
                shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
                shared_vars["lfc_shared"]: lfc_data_batch,
            }
        },
    )

    done()
    return


#### ---- CRC Model 3 ---- ####


def crc_model3(
    name: str,
    debug: bool = False,
    force_sampling: bool = False,
    random_seed: Optional[int] = None,
) -> None:
    print_model("CRC Model 3")

    # Data
    info("Loading data...")
    data = load_crc_data(debug)

    batch_size = crc_batch_size(debug)

    # Indices
    sgrna_idx = dphelp.get_indices(data, "sgrna")
    sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(data)
    sgrna_to_gene_idx = dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol")
    cell_idx = dphelp.get_indices(data, "depmap_id")

    # Batched data
    sgrna_idx_batch = pm.Minibatch(sgrna_idx, batch_size=batch_size)
    cell_idx_batch = pm.Minibatch(cell_idx, batch_size=batch_size)
    lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

    # Construct model
    crc_m3, shared_vars = crc_models.model_3(
        sgrna_idx=sgrna_idx,
        sgrna_to_gene_idx=sgrna_to_gene_idx,
        cell_idx=cell_idx,
        lfc_data=data.lfc.values,
    )

    # Sample and cache
    crc_m3_cache = make_cache_name(name)

    _ = pymc3_sampling_api.pymc3_advi_approximation_procedure(
        model=crc_m3,
        callbacks=[
            pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
        ],
        random_seed=random_seed,
        cache_dir=crc_m3_cache,
        force=force_sampling,
        fit_kwargs={
            "more_replacements": {
                shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
                shared_vars["cell_idx_shared"]: cell_idx_batch,
                shared_vars["lfc_shared"]: lfc_data_batch,
            }
        },
    )

    done()
    return


#### ---- MAIN ---- ####

MODELS = ["ceres-m1", "ceres-m2", "crc-m1", "crc-m2", "crc-m3"]


def parse_cli_arguments(
    parser: argparse.ArgumentParser, args: Optional[List[Any]] = None
) -> argparse.Namespace:

    if args is None:
        args = sys.argv[1:]

    parser.add_argument(
        "-m",
        "--model",
        help="model to sample from",
        type=str,
        choices=MODELS,
        required=True,
    )
    parser.add_argument("-n", "--name", help="model name", type=str, required=True)
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
    parser.add_argument(
        "--touch",
        help="touch a file with the name of the model when sampling has finished",
        action="store_true",
        default=False,
    )

    return parser.parse_args(args)


def clean_model_names(n: str) -> str:
    return n.replace(" ", "-")


def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    args = parse_cli_arguments(parser)

    tic = time()

    np.random.seed(args.random_seed)

    if args.debug:
        print(Fore.RED + "(🪲 debug mode)")

    model_name = clean_model_names(args.name)

    if args.model == "ceres-m1":
        ceres_model1(
            name=model_name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "ceres-m2":
        ceres_model2(
            name=model_name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "crc-m1":
        crc_model1(
            name=model_name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "crc-m2":
        crc_model2(
            name=model_name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    elif args.model == "crc-m3":
        crc_model3(
            name=model_name,
            debug=args.debug,
            force_sampling=args.force_sample,
            random_seed=args.random_seed,
        )
    else:
        raise Exception("Unrecognized model 🤷🏻‍♂️")

    if args.touch and args.model in MODELS:
        info("Touching output file.")
        touch(args.model)

    toc = time()
    info(f"execution time: {(toc - tic) / 60:.2f} minutes")
    return None


if __name__ == "__main__":
    main()
