import argparse
from pathlib import Path
from time import time

import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3_helpers as pmhelp
from colorama import Fore, init
from pymc3_models import ceres_models, crc_models

#### ---- Argument parsing ---- ####

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--models",
    help="model(s) to sample from",
    nargs="+",
    type=str,
    choices=["ceres-m1", "ceres-m2"],
)
parser.add_argument(
    "--force-sample",
    help="ignore cached results and sample from model",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-d", "--debug", help="debug mode", action="store_true", default=False
)

args = parser.parse_args()


#### ---- Setup ---- ####

script_tic = time()

init(autoreset=True)

RANDOM_SEED = 1146
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("analysis") / pmhelp.default_cache_dir

DEBUG = args.debug
if DEBUG:
    print(Fore.RED + "(debug mode)")


#### ---- CERES Model 1 ---- ####

if "ceres-m1" in args.models:
    print(Fore.BLUE + "CERES Model 1")

    # Data
    print("Loading data...")
    if DEBUG:
        data_path = Path("modeling_data/depmap_modeling_dataframe_subsample.csv")
    else:
        data_path = Path("modeling_data/depmap_modeling_dataframe.csv")

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

    ceres_m1_cache = pymc3_cache_dir / "mimic-ceres-m1"
    _ = pmhelp.pymc3_sampling_procedure(
        model=ceres_m1,
        num_mcmc=1000,
        tune=1000,
        chains=2,
        cores=2,
        random_seed=RANDOM_SEED,
        cache_dir=ceres_m1_cache,
        force=args.force_sample,
        sample_kwargs={
            "init": "advi+adapt_diag",
            "n_init": 50000,
            "target_accept": 0.9,
        },
    )

    print(Fore.GREEN + "Done")


#### ---- CERES Model 2 ---- ####

if "ceres-m2" in args.models:
    print(Fore.BLUE + "CERES Model 2")

    # Data
    print("Loading data...")
    if DEBUG:
        data_path = Path("modeling_data/depmap_modeling_dataframe_subsample_medium.csv")
    else:
        data_path = Path("modeling_data/depmap_modeling_dataframe.csv")

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

    ceres_m2_cache = pymc3_cache_dir / "mimic-ceres-m2"
    _ = pmhelp.pymc3_sampling_procedure(
        model=ceres_m2,
        num_mcmc=1000,
        tune=1000,
        chains=2,
        cores=2,
        random_seed=RANDOM_SEED,
        cache_dir=ceres_m2_cache,
        force=args.force_sample,
        sample_kwargs={
            "init": "advi+adapt_diag",
            "n_init": 100000,
            "target_accept": 0.9,
        },
    )

    print(Fore.GREEN + "Done")


#### ---- Finish ---- ####

script_toc = time()
print(Fore.CYAN + f"execution time: {(script_toc - script_tic) / 60:.2f} minutes")
