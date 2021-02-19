from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

#### ---- Model 1 ---- ####


def model_1(
    gene_idx: np.ndarray, lfc_data: np.ndarray
) -> Tuple[pm.model.Model, TTShared, TTShared]:

    n_genes = len(np.unique(gene_idx))

    # Shared Theano variables.
    gene_idx_shared = theano.shared(gene_idx)
    lfc_shared = theano.shared(lfc_data)

    with pm.Model() as model:

        # Hyper-priors
        μ_α = pm.Normal("μ_α", 0, 5)
        σ_α = pm.Exponential("σ_α", 2)

        # Priors
        α_g = pm.Normal("α_g", μ_α, σ_α, shape=n_genes)

        μ = pm.Deterministic("μ", α_g[gene_idx_shared])
        σ = pm.HalfNormal("σ", 5)

        lfc = pm.Normal("lfc", μ, σ, observed=lfc_shared, total_size=len(lfc_data))

    return model, gene_idx_shared, lfc_shared


#### ---- Model 2 ---- ####


def model_2(
    sgrna_idx: np.ndarray, sgrna_to_gene_idx: np.ndarray, lfc_data: np.ndarray
) -> Tuple[pm.model.Model, TTShared, TTShared, TTShared]:

    total_size = len(lfc_data)
    n_sgrnas = len(np.unique(sgrna_idx))
    n_genes = len(np.unique(sgrna_to_gene_idx))

    # Shared Theano variables
    sgrna_idx_shared = theano.shared(sgrna_idx)
    sgrna_to_gene_idx_shared = theano.shared(sgrna_to_gene_idx)
    lfc_shared = theano.shared(lfc_data)

    with pm.Model() as model:
        # Hyper-priors
        μ_g = pm.Normal("μ_g", 0, 5)
        σ_g = pm.HalfNormal("σ_g", 2)
        σ_σ_α = pm.HalfNormal("σ_σ_α", 2)

        # Prior per gene that sgRNAs are sampled from.
        μ_α = pm.Normal("μ_α", μ_g, σ_g, shape=n_genes)
        σ_α = pm.HalfNormal("σ_α", σ_σ_α, shape=n_genes)

        # Prior per sgRNA
        α_s = pm.Normal(
            "α_s",
            μ_α[sgrna_to_gene_idx_shared],
            σ_α[sgrna_to_gene_idx_shared],
            shape=n_sgrnas,
        )

        # Main model level
        μ = α_s[sgrna_idx_shared]
        σ = pm.HalfNormal("σ", 5)

        # Likelihood
        lfc = pm.Normal("lfc", μ, σ, observed=lfc_shared, total_size=total_size)

    return model, sgrna_idx_shared, sgrna_to_gene_idx_shared, lfc_shared
