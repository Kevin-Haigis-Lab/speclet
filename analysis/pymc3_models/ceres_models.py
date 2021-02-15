from typing import Any, Dict

import numpy as np
import pandas as pd
import pymc3 as pm

#### ---- Setup ---- ####


#### ---- Models ---- ####


## MODEL 1


def construct_ceres_m1(
    sgrna_idx: np.ndarray,
    gene_idx: np.ndarray,
    cell_idx: np.ndarray,
    cn_data: np.ndarray,
    lfc_data: np.ndarray,
) -> pm.model.Model:
    """
    CERES Model 1

    The same model as originally described but without a spline for CN.
    (No pooling of information.)
    """

    num_sgrnas = len(np.unique(sgrna_idx))
    num_genes = len(np.unique(gene_idx))
    num_cells = len(np.unique(cell_idx))

    with pm.Model() as ceres_m1:

        # Indices
        sgrna_idx_shared = pm.Data("sgrna_idx_shared", sgrna_idx)
        gene_idx_shared = pm.Data("gene_idx_shared", gene_idx)
        cell_idx_shared = pm.Data("cell_idx_shared", cell_idx)

        # Data
        C_shared = pm.Data("C", cn_data)
        lfc_shared = pm.Data("lfc", lfc_data)

        # Priors
        q = pm.Beta("q", 3, 3, shape=num_sgrnas)
        h = pm.Normal("h", 0, 2, shape=num_genes)
        g = pm.Normal("g", 0, 2, shape=(num_genes, num_cells))
        β = pm.Normal("β", -0.2, 1, shape=num_cells)
        o = pm.Normal("o", 0, 1, shape=num_sgrnas)

        # Linear model
        μ = pm.Deterministic(
            "μ",
            q[sgrna_idx_shared]
            * (h[gene_idx_shared] + g[gene_idx_shared, cell_idx_shared])
            + β[cell_idx_shared] * C_shared
            + o[sgrna_idx_shared],
        )
        σ = pm.HalfNormal("σ", 3)

        # Likelihood
        D = pm.Normal("D", μ, σ, observed=lfc_shared)

    return ceres_m1


## MODEL 2


def construct_ceres_m2(
    sgrna_idx: np.ndarray,
    gene_idx: np.ndarray,
    cell_idx: np.ndarray,
    cn_data: np.ndarray,
    lfc_data: np.ndarray,
) -> pm.model.Model:
    """
    CERES Model 2

    Mostly the same as the original model, but now there is partial pooling
    of information for each covariate.
    """

    num_sgrnas = len(np.unique(sgrna_idx))
    num_genes = len(np.unique(gene_idx))
    num_cells = len(np.unique(cell_idx))

    with pm.Model() as ceres_m2:

        # Indices
        sgrna_idx_shared = pm.Data("sgrna_idx_shared", sgrna_idx)
        gene_idx_shared = pm.Data("gene_idx_shared", gene_idx)
        cell_idx_shared = pm.Data("cell_idx_shared", cell_idx)

        # Data
        C_shared = pm.Data("C", cn_data)
        lfc_shared = pm.Data("lfc", lfc_data)

        # Hyper-priors
        a_q = pm.Exponential("q_a", 3)
        b_q = pm.Exponential("q_b", 3)

        μ_h = pm.Normal("μ_h", -0.5, 0.5)
        σ_h = pm.HalfNormal("σ_h", 1)

        μ_g = pm.Normal("μ_g", 0, 0.5)
        σ_g = pm.HalfNormal("σ_g", 1)

        μ_β = pm.Normal("μ_β", -0.2, 1)
        σ_β = pm.HalfNormal("σ_β", 1)

        μ_o = pm.Normal("μ_o", 0, 0.5)
        σ_o = pm.HalfNormal("σ_o", 0.2)

        # Priors
        q = pm.Beta("q", a_q, b_q, shape=num_sgrnas)
        h = pm.Normal("h", μ_h, σ_h, shape=num_genes)
        g = pm.Normal("g", μ_g, σ_g, shape=(num_genes, num_cells))
        β = pm.Normal("β", μ_β, σ_β, shape=num_cells)
        o = pm.Normal("o", μ_o, σ_o, shape=num_sgrnas)

        # Linear model
        μ = pm.Deterministic(
            "μ",
            q[sgrna_idx_shared]
            * (h[gene_idx_shared] + g[gene_idx_shared, cell_idx_shared])
            + β[cell_idx_shared] * C_shared
            + o[sgrna_idx_shared],
        )
        σ = pm.HalfNormal("σ", 3)

        # Likelihood
        D = pm.Normal("D", μ, σ, observed=lfc_shared)

    return ceres_m2


#### ---- Finish ---- ####
