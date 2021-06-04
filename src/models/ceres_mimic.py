#!/usr/bin/env python3

"""Builders for CRC CERES Mimic model."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pymc3 as pm
import theano

from src.data_processing import achilles as achelp
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel

#### ---- CERES Mimic ---- ####


class CeresMimic(SpecletModel):
    """CERES Mimic.

    This model is just the part of the CERES model that includes the sgRNA
    "activity score" (q) and the per-gene (h) and per-gene-per-cell-line (d) covariates.
    In addition, I have included a parameter for pDNA batch.
    """

    _copynumber_cov: bool
    _sgrna_intercept_cov: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
        copynumber_cov: bool = False,
        sgrna_intercept_cov: bool = False,
    ):
        """Create a CeresMimic object.

        Args:
            name (str): A unique identifier for this instance of CeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
            copynumber_cov (bool, optional): Should the gene copy number covariate be
              included in the model? Default to False.
            sgrna_intercept_cov (bool, optional): Should a varying intercept for
              `sgRNA|gene` be included in the model? Default to False.
        """
        if data_manager is None:
            data_manager = CrcDataManager(debug=debug)

        super().__init__(
            name="ceres-mimic-1_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )
        self._copynumber_cov = copynumber_cov
        self._sgrna_intercept_cov = sgrna_intercept_cov

    @property
    def copynumber_cov(self) -> bool:
        """Get the current value of `copynumber_cov` attribute.

        Returns:
            bool: Whether or not the copy number covariate is included in the model.
        """
        return self._copynumber_cov

    @copynumber_cov.setter
    def copynumber_cov(self, new_value: bool):
        """Set the value for the `copynumber_cov` attribute.

        If the value changes, then the `model` attribute and model results attributes
        `advi_results` and `mcmc_results` are all reset to None.

        Args:
            new_value (bool): Whether or not the copy number covariate should be
              included in the model.
        """
        if new_value != self._copynumber_cov:
            self._reset_model_and_results()
            self._copynumber_cov = new_value

    @property
    def sgrna_intercept_cov(self) -> bool:
        """Get the current value of `sgrna_intercept_cov` attribute.

        Returns:
            bool: Whether or not the `sgRNA|gene` varying intercept covariate is
              included in the model.
        """
        return self._sgrna_intercept_cov

    @sgrna_intercept_cov.setter
    def sgrna_intercept_cov(self, new_value: bool):
        """Set the value for the `sgrna_intercept_cov` attribute.

        If the value changes, then the `model` attribute and model results attributes
        `advi_results` and `mcmc_results` are all reset to None.

        Args:
            new_value (bool): Whether or not the `sgRNA|gene` varying intercept
              covariate is included in the model.
        """
        if new_value != self._sgrna_intercept_cov:
            self._reset_model_and_results()
            self._sgrna_intercept_cov = new_value

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build CRC CERES Mimic One.

        Returns:
            [None]: None
        """
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        indices_collection = achelp.common_indices(data)

        # Shared Theano variables
        sgrna_idx_shared = theano.shared(indices_collection.sgrna_idx)
        sgrna_to_gene_idx_shared = theano.shared(indices_collection.sgrna_to_gene_idx)
        gene_idx_shared = theano.shared(indices_collection.gene_idx)
        cellline_idx_shared = theano.shared(indices_collection.cellline_idx)
        batch_idx_shared = theano.shared(indices_collection.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)
        copynumber_shared = theano.shared(data.copy_number.values)

        with pm.Model() as model:

            # Hyper-priors
            σ_a = pm.HalfNormal("σ_a", np.array([0.1, 2]), shape=2)
            a = pm.Exponential("a", σ_a, shape=(indices_collection.n_genes, 2))

            μ_h = pm.Normal("μ_h", 0, 0.2)
            σ_h = pm.HalfNormal("σ_h", 1)

            μ_d = pm.Normal("μ_d", 0, 0.2)
            σ_d = pm.HalfNormal("σ_d", 1)

            μ_η = pm.Normal("μ_η", 0, 0.1)
            σ_η = pm.HalfNormal("σ_η", 0.1)

            # Main parameter priors
            q = pm.Beta(
                "q",
                alpha=a[sgrna_to_gene_idx_shared, 0],
                beta=a[sgrna_to_gene_idx_shared, 1],
                shape=indices_collection.n_sgrnas,
            )
            h = pm.Normal("h", μ_h, σ_h, shape=indices_collection.n_genes)
            d = pm.Normal(
                "d",
                μ_d,
                σ_d,
                shape=(indices_collection.n_genes, indices_collection.n_celllines),
            )
            η = pm.Normal("η", μ_η, σ_η, shape=indices_collection.n_batches)

            gene_comp = h[gene_idx_shared] + d[gene_idx_shared, cellline_idx_shared]

            if self.copynumber_cov:
                # Add varying slope with copy number.
                μ_β = pm.Normal("μ_β", 0, 0.1)
                σ_β = pm.HalfNormal("σ_β", 0.5)
                β = pm.Normal("β", μ_β, σ_β, shape=indices_collection.n_celllines)
                gene_comp += β[cellline_idx_shared] * copynumber_shared

            μ = pm.Deterministic(
                "μ", q[sgrna_idx_shared] * gene_comp + η[batch_idx_shared]
            )

            if self.sgrna_intercept_cov:
                # Hyper priors for sgRNA|gene varying intercept.
                μ_og = pm.Normal("μ_og", 0, 0.1)
                σ_og = pm.HalfNormal("σ_og", 0.2)
                # Priors for sgRNA|gene varying intercept.
                μ_o = pm.Normal("μ_o", μ_og, σ_og, shape=indices_collection.n_genes)
                σ_o = pm.HalfNormal("σ_o", 0.2)
                # sgRNA|gene varying intercept.
                o = pm.Normal(
                    "o",
                    μ_o[sgrna_to_gene_idx_shared],
                    σ_o,
                    shape=indices_collection.n_sgrnas,
                )
                μ = μ + o[sgrna_idx_shared]

            σ = pm.HalfNormal("σ", 2)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc", μ, σ, observed=lfc_shared, total_size=total_size
            )

        shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "lfc_shared": lfc_shared,
            "copynumber_shared": copynumber_shared,
        }

        self.shared_vars = shared_vars
        return model, "lfc"

    def get_replacement_parameters(self) -> ReplacementsDict:
        """Make a dictionary mapping the shared data variables to new data.

        Raises:
            AttributeError: Raised if there is no data manager.
            AttributeError: Raised if there are no shared variables.

        Returns:
            ReplacementsDict: A dictionary mapping new data to shared variables.
        """
        if self.data_manager is None:
            raise AttributeError(
                "Cannot create replacement parameters without a DataManager."
            )
        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        data = self.data_manager.get_data()
        batch_size = self.data_manager.get_batch_size()
        indices = achelp.common_indices(data)

        sgrna_idx_batch = pm.Minibatch(indices.sgrna_idx, batch_size=batch_size)
        gene_idx_batch = pm.Minibatch(indices.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(indices.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(indices.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)
        copynumber_data_batch = pm.Minibatch(
            data.copy_number.values, batch_size=batch_size
        )

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
            self.shared_vars["copynumber_shared"]: copynumber_data_batch,
        }
