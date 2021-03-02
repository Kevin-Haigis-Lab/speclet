#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union

import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pymc3_sampling_api
from colorama import Back, Fore, Style, init
from pymc3_models import crc_models

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


def common_indices(
    achilles_df: pd.DataFrame,
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    sgrna_idx = dphelp.get_indices(achilles_df, "sgrna")
    sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(achilles_df)
    sgrna_to_gene_idx = dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol")
    cellline_idx = dphelp.get_indices(achilles_df, "depmap_id")
    batch_idx = dphelp.get_indices(achilles_df, "pdna_batch")

    return {
        "sgrna_idx": sgrna_idx,
        "sgrna_to_gene_map": sgrna_to_gene_map,
        "sgrna_to_gene_idx": sgrna_to_gene_idx,
        "cellline_idx": cellline_idx,
        "batch_idx": batch_idx,
    }


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
    indices_dict = common_indices(data)

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

MODELS = ["crc-m1"]


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

    if args.model == "crc-m1":
        crc_model1(
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
