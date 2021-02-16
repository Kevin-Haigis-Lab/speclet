import numpy as np
import pandas as pd
import pymc3 as pm

#### ---- Model 1 ---- ####


def model_1(gene_idx: np.ndarray, lfc_data: np.ndarray) -> pm.model.Model:

    n_genes = len(np.unique(gene_idx))

    with pm.Model() as model:

        # Indicies
        gene_idx_shared = pm.Data("gene_idx", gene_idx)

        # Shared data
        lfc_observed = pm.Data("lfc_observed", lfc_data)

        # Hyper-priors
        μ_α = pm.Normal("μ_α", 0, 5)
        σ_α = pm.Exponential("σ_α", 2)

        # Priors
        α_g = pm.Normal("α_g", μ_α, σ_α, shape=n_genes)

        μ = pm.Deterministic("μ", α_g[gene_idx_shared])
        σ = pm.HalfNormal("σ", 5)

        lfc = pm.Normal("lfc", μ, σ, observed=lfc_observed)

    return model


#### ---- Model 2 ---- ####


def model_2(
    sgrna_idx: np.ndarray, sgrna_to_gene_idx: np.ndarray, lfc_data: np.ndarray
) -> pm.model.Model:

    n_sgrnas = len(np.unique(sgrna_idx))
    n_genes = len(np.unique(sgrna_to_gene_idx))

    with pm.Model() as model:

        # Indicies
        sgrna_idx_shared = pm.Data("sgrna_idx", sgrna_idx)
        sgrna_to_gene_idx_shared = pm.Data("sgrna_to_gene_idx", sgrna_to_gene_idx)

        # Shared data
        lfc_observed = pm.Data("lfc_observed", lfc_data)

        # Hyper-hyper-priors
        μ_g = pm.Normal("μ_g", 0, 5)
        σ_g = pm.Exponential("σ_g", 2)

        # Hyper-priors
        μ_α = pm.Normal("μ_α", μ_g, σ_g, shape=n_genes)
        σ_α = pm.Exponential("σ_α", 2)

        # Priors
        α_g = pm.Normal("α_g", μ_α[sgrna_to_gene_idx_shared], σ_α, shape=n_sgrnas)

        # Model
        μ = pm.Deterministic("μ", α_g[sgrna_idx_shared])
        σ = pm.HalfNormal("σ", 5)

        # Likelihood
        lfc = pm.Normal("lfc", μ, σ, observed=lfc_observed)

    return model
