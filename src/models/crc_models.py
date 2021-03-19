#!/usr/bin/env python3

"""Builders for CRC PyMC3 models."""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.data_processing.common import nunique
from src.models.speclet_model import SpecletModel


class CrcModel(SpecletModel):
    """Base model for CRC modeling.

    Args:
        SpecletModel ([type]): Subclassed from a SpecletModel.
    """

    debug: bool
    data: Optional[pd.DataFrame] = None
    data_dir: Path = Path("modeling_data")

    def __init__(self, cache_dir: Optional[Path] = None, debug: bool = False):
        """Create a CrcModel object.

        Args:
            cache_dir (Optional[Path], optional): The directory for caching sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
        """
        super().__init__(cache_dir=cache_dir)
        self.debug = debug

    def get_data_path(self) -> Path:
        """Get the path for the data set to use.

        Returns:
            Path: Path to the data.
        """
        f = (
            "depmap_modeling_dataframe_subsample.csv"
            if self.debug
            else "depmap_modeling_dataframe.csv"
        )
        return self.data_dir / f

    def get_batch_size(self) -> int:
        """Decide on the minibatch size for modeling CRC data.

        Returns:
            int: Batch size.
        """
        if self.debug:
            return 1000
        else:
            return 10000

    def _load_data(self) -> pd.DataFrame:
        """Load CRC data."""
        return achelp.read_achilles_data(self.get_data_path(), low_memory=False)

    def get_data(self) -> pd.DataFrame:
        """Get the data for modeling.

        If the data is not already loaded, it is first read from disk.
        """
        if self.data is None:
            self.data = self._load_data()
        return self.data


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
