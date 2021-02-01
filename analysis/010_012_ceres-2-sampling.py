from pathlib import Path
from time import time

import ceres_models
import common_data_processing as dphelp
import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3_helpers as pmhelp
from colorama import Fore, init

#### ---- Setup ---- ####

script_tic = time()

init(autoreset=True)

RANDOM_SEED = 1146
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("analysis") / pmhelp.default_cache_dir

DEBUG = True

if DEBUG:
    print(Fore.RED + "(debug mode)")


#### ---- Data prepration ---- ####


def get_data(debug: bool) -> pd.DataFrame:
    if debug:
        data_path = Path("modeling_data/depmap_modeling_dataframe_subsample.csv")
    else:
        data_path = Path("modeling_data/depmap_modeling_dataframe.csv")

    return dphelp.read_achilles_data(data_path)


#### ---- Sampling models ---- ####

# Data
print("Loading data...")
data = get_data(debug=DEBUG)

# Indices
sgrna_idx = dphelp.get_indices(data, "sgrna")
gene_idx = dphelp.get_indices(data, "hugo_symbol")
cell_idx = dphelp.get_indices(data, "depmap_id")


## CERES Model 1
print(Fore.BLUE + "CERES Model 1")

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
    force=True,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 200000, "target_accept": 0.9},
)

print(Fore.GREEN + "Done")


## CERES Model 2
print(Fore.BLUE + "CERES Model 2")

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
    force=True,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 200000, "target_accept": 0.9},
)

print(Fore.GREEN + "Done")


#### ---- Finish ---- ####

script_toc = time()
print(Fore.CYAN + f"execution time: {(script_toc - script_tic) / 60:.2f} minutes")
