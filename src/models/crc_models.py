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
from src.data_processing import common as dphelp
from src.data_processing.common import nunique
from src.io import data_io
from src.models.protocols import SelfSufficientModel
from src.models.speclet_model import SpecletModel


class CrcModel(SpecletModel):
    """Base model for CRC modeling.

    Args:
        SpecletModel ([type]): Subclassed from a SpecletModel.
    """

    debug: bool
    data: Optional[pd.DataFrame] = None

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
        if self.debug:
            return data_io.data_path(to=data_io.DataFile.crc_subsample)
        return data_io.data_path(to=data_io.DataFile.crc_data)

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


class CrcModelOne(CrcModel, SelfSufficientModel):
    """CRC Model One."""

    shared_vars: Optional[Dict[str, TTShared]] = None
    model: Optional[pm.Model] = None

    def __init__(self, cache_dir: Optional[Path] = None, debug: bool = False):
        """Create a CrcModelOne object.

        Args:
            cache_dir (Optional[Path], optional): The directory for caching sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
        """
        super().__init__(cache_dir=cache_dir, debug=debug)

    def build_model(self) -> None:
        """Build CRC Model One."""
        data = self.get_data()

        total_size = data.shape[0]
        indices_collection = achelp.common_indices(data)

        # Shared Theano variables
        sgrna_idx_shared = theano.shared(indices_collection.sgrna_idx)
        sgrna_to_gene_idx_shared = theano.shared(indices_collection.sgrna_to_gene_idx)
        cellline_idx_shared = theano.shared(indices_collection.cellline_idx)
        batch_idx_shared = theano.shared(indices_collection.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)

        with pm.Model() as model:
            # Hyper-priors
            μ_g = pm.Normal("μ_g", np.mean(data.lfc.values), 1)
            σ_g = pm.HalfNormal("σ_g", 2)
            σ_σ_α = pm.HalfNormal("σ_σ_α", 1)

            # Prior per gene that sgRNAs are sampled from.
            μ_α = pm.Normal("μ_α", μ_g, σ_g, shape=indices_collection.n_genes)
            σ_α = pm.HalfNormal("σ_α", σ_σ_α, shape=indices_collection.n_genes)
            μ_β = pm.Normal("μ_β", 0, 0.2)
            σ_β = pm.HalfNormal("σ_β", 1)
            μ_η = pm.Normal("μ_η", 0, 0.2)
            σ_η = pm.HalfNormal("σ_η", 1)

            # Prior per sgRNA
            α_s = pm.Normal(
                "α_s",
                μ_α[sgrna_to_gene_idx_shared],
                σ_α[sgrna_to_gene_idx_shared],
                shape=indices_collection.n_sgrnas,
            )
            β_l = pm.Normal("β_l", μ_β, σ_β, shape=indices_collection.n_celllines)
            η_b = pm.Normal("η_b", μ_η, σ_η, shape=indices_collection.n_batches)

            # Main model level
            μ = pm.Deterministic(
                "μ",
                α_s[sgrna_idx_shared]
                + β_l[cellline_idx_shared]
                + η_b[batch_idx_shared],
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

        self.model = model
        self.shared_vars = shared_vars
        return None

    def sample_model(self):
        """Sample from the model."""
        print("sampling...")

    def run_simulation_based_calibration(self):
        """Run a round of simulation-based calibration."""
        print("Running SBC...")
