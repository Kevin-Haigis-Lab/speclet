#!/usr/bin/env python3

"""Builders for CRC PyMC3 models."""

from typing import Dict, Tuple

import numpy as np
import pretty_errors
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing.common import nunique

#### ---- Model 1 ---- ####


def model_1(
    sgrna_idx: np.ndarray,
    sgrna_to_gene_idx: np.ndarray,
    cellline_idx: np.ndarray,
    batch_idx: np.ndarray,
    lfc_data: np.ndarray,
) -> Tuple[pm.Model, Dict[str, TTShared]]:
    """Build CRC Model 1.

    Args:
        sgrna_idx (np.ndarray): sgRNA index.
        sgrna_to_gene_idx (np.ndarray): sgRNA to gene index.
        cellline_idx (np.ndarray): Cell line index.
        batch_idx (np.ndarray): pDNA batch index.
        lfc_data (np.ndarray): Log-fold change (LFC) data.

    Returns:
        Tuple[pm.Model, Dict[str, TTShared]]: A collection of the model and shared variables.
    """
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
