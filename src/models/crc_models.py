#!/usr/bin/env python3

"""Builders for CRC PyMC3 models."""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.io import data_io
from src.modeling import pymc3_sampling_api as pmapi
from src.modeling.sampling_pymc3_models import SamplingArguments
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
    advi_results: Optional[pmapi.ApproximationSamplingResults] = None
    mcmc_results: Optional[pmapi.MCMCSamplingResults] = None

    ReplacementsDict = Dict[TTShared, Union[pm.Minibatch, np.ndarray]]

    def __init__(self, cache_dir: Optional[Path] = None, debug: bool = False):
        """Create a CrcModelOne object.

        Args:
            cache_dir (Optional[Path], optional): The directory for caching sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
        """
        super().__init__(cache_dir=cache_dir, debug=debug)

    def _get_indices_collection(self, data: pd.DataFrame) -> achelp.CommonIndices:
        return achelp.common_indices(data)

    def build_model(self) -> None:
        """Build CRC Model One."""
        data = self.get_data()

        total_size = data.shape[0]
        indices_collection = self._get_indices_collection(data)

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

    def _get_replacement_parameters(self) -> ReplacementsDict:
        if self.data is None:
            raise AttributeError(
                "Cannot create replacement parameters before data has been loaded."
            )
        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        batch_size = self.get_batch_size()
        indices = self._get_indices_collection(self.data)

        sgrna_idx_batch = pm.Minibatch(indices.sgrna_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(indices.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(indices.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(self.data.lfc.values, batch_size=batch_size)

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }

    def mcmc_sample_model(
        self, sampling_args: SamplingArguments
    ) -> pmapi.MCMCSamplingResults:
        """Fit the model with MCMC.

        Args:
            sampling_args (SamplingArguments): Arguments for the sampling procedure.
        """
        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. Make sure to run `model.build_model()` first."
            )
        if self.shared_vars is None:
            raise AttributeError("Cannot sample: cannot find shared variables.")

        replacements = self._get_replacement_parameters()

        if self.mcmc_results is not None:
            return self.mcmc_results

        if (
            not sampling_args.ignore_cache
            and self.cache_dir is not None
            and self.cache_exists(method="mcmc")
        ):
            return self.read_cached_approximation()

        self.mcmc_results = pmapi.pymc3_sampling_procedure(
            model=self.model,
            cores=sampling_args.cores,
            random_seed=sampling_args.random_seed,
        )
        return self.mcmc_results

    def advi_sample_model(
        self, sampling_args: SamplingArguments
    ) -> pmapi.ApproximationSamplingResults:
        """Fit the model with ADVI.

        Args:
            sampling_args (SamplingArguments): Arguments for the sampling procedure.

        Returns:
            ApproximationSamplingResults: The results of fitting the model with ADVI.
        """
        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. Make sure to run `model.build_model()` first."
            )
        if self.shared_vars is None:
            raise AttributeError("Cannot sample: cannot find shared variables.")

        replacements = self._get_replacement_parameters()

        if self.advi_results is not None:
            return self.advi_results

        if (
            not sampling_args.ignore_cache
            and self.cache_dir is not None
            and self.cache_exists(method="advi")
        ):
            return self.read_cached_approximation()

        self.advi_results = pmapi.pymc3_advi_approximation_procedure(
            model=self.model,
            callbacks=[
                pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
            ],
            random_seed=sampling_args.random_seed,
            fit_kwargs={"more_replacements": replacements},
        )
        return self.advi_results

    def run_simulation_based_calibration(self):
        """Run a round of simulation-based calibration."""
        # TODO
        print("Running SBC...")
