#!/usr/bin/env python3

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared


def nunique(a: np.ndarray) -> int:
    return len(np.unique(a))


#### ---- Model 1 ---- ####


def model_1(
    sgrna_idx: np.ndarray,
    sgrna_to_gene_idx: np.ndarray,
    cellline_idx: np.ndarray,
    batch_idx: np.ndarray,
    lfc_data: np.ndarray,
) -> Tuple[pm.model.Model, Dict[str, TTShared]]:

    total_size = len(lfc_data)
    n_sgrnas = nunique(sgrna_idx)
    n_genes = nunique(sgrna_to_gene_idx)
    n_lines = nunique(cellline_idx)
    n_batches = nunique(batch_idx)

    # Shared Theano variables
    sgrna_idx_shared = theano.shared(sgrna_idx)
    sgrna_to_gene_idx_shared = theano.shared(sgrna_to_gene_idx)
    cellline_idx_shared = theano.shared(cellline_idx)
    batch_idx_shared = theano.shared(batch_idx)
    lfc_shared = theano.shared(lfc_data)

    with pm.Model() as model:
        # Hyper-priors
        μ_g = pm.Normal("μ_g", np.mean(lfc_data), 1)
        σ_g = pm.HalfNormal("σ_g", 2)
        σ_σ_α = pm.HalfNormal("σ_σ_α", 1)

        # Prior per gene that sgRNAs are sampled from.
        μ_α = pm.Normal("μ_α", μ_g, σ_g, shape=n_genes)
        σ_α = pm.HalfNormal("σ_α", σ_σ_α, shape=n_genes)
        μ_β = pm.Normal("μ_β", 0, 0.2)
        σ_β = pm.HalfNormal("σ_β", 1)
        μ_η = pm.Normal("μ_η", 0, 0.2)
        σ_η = pm.HalfNormal("σ_η", 1)

        # Prior per sgRNA
        α_s = pm.Normal(
            "α_s",
            μ_α[sgrna_to_gene_idx_shared],
            σ_α[sgrna_to_gene_idx_shared],
            shape=n_sgrnas,
        )
        β_l = pm.Normal("β_l", μ_β, σ_β, shape=n_lines)
        η_b = pm.Normal("η_b", μ_η, σ_η, shape=n_batches)

        # Main model level
        μ = pm.Deterministic(
            "μ",
            α_s[sgrna_idx_shared] + β_l[cellline_idx_shared] + η_b[batch_idx_shared],
        )
        σ = pm.HalfNormal("σ", 2)

        # Likelihood
        lfc = pm.Normal("lfc", μ, σ, observed=lfc_shared, total_size=total_size)

    shared_vars = {
        "sgrna_idx_shared": sgrna_idx_shared,
        "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
        "cellline_idx_shared": cellline_idx_shared,
        "batch_idx_shared": batch_idx_shared,
        "lfc_shared": lfc_shared,
    }
    return model, shared_vars


#### ---- Model 3 ---- ####


def model_3(
    sgrna_idx: np.ndarray,
    sgrna_to_gene_idx: np.ndarray,
    cell_idx: np.ndarray,
    lfc_data: np.ndarray,
) -> Tuple[pm.model.Model, Dict[str, TTShared]]:

    total_size = len(lfc_data)
    n_sgrnas = nunique(sgrna_idx)
    n_cells = nunique(cell_idx)
    n_genes = nunique(sgrna_to_gene_idx)

    # Shared Theano variables
    sgrna_idx_shared = theano.shared(sgrna_idx)
    sgrna_to_gene_idx_shared = theano.shared(sgrna_to_gene_idx)
    cell_idx_shared = theano.shared(cell_idx)
    lfc_shared = theano.shared(lfc_data)

    with pm.Model() as model:
        # Hyper-priors
        μ_g = pm.Normal("μ_g", 0, 5)
        σ_g = pm.HalfNormal("σ_g", 2)
        σ_σ_α = pm.HalfNormal("σ_σ_α", 2)

        # Prior per gene that sgRNAs are sampled from.
        μ_α = pm.Normal("μ_α", μ_g, σ_g, shape=n_genes)
        σ_α = pm.HalfNormal("σ_α", σ_σ_α, shape=n_genes)
        μ_β = pm.Normal("μ_β", 0, 0.5)
        σ_β = pm.HalfNormal("σ_β", 1)

        # Prior per sgRNA
        α_s = pm.Normal(
            "α_s",
            μ_α[sgrna_to_gene_idx_shared],
            σ_α[sgrna_to_gene_idx_shared],
            shape=n_sgrnas,
        )
        β_c = pm.Normal("β_c", μ_β, σ_β, shape=n_cells)

        # Main model level
        μ = pm.Deterministic("μ", α_s[sgrna_idx_shared] + β_c[cell_idx_shared])
        σ = pm.HalfNormal("σ", 5)

        # Likelihood
        lfc = pm.Normal("lfc", μ, σ, observed=lfc_shared, total_size=total_size)

    shared_vars = {
        "sgrna_idx_shared": sgrna_idx_shared,
        "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
        "cell_idx_shared": cell_idx_shared,
        "lfc_shared": lfc_shared,
    }

    return model, shared_vars
