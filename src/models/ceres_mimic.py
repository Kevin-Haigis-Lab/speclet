#!/usr/bin/env python3

"""Builders for CRC CERES Mimic model."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pymc3 as pm
import theano
from pydantic import BaseModel

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import CrisprScreenDataManager
from src.models.speclet_model import (
    ObservedVarName,
    ReplacementsDict,
    SpecletModel,
    SpecletModelDataManager,
)

#### ---- CERES Mimic ---- ####


class CeresMimicConfiguration(BaseModel):
    """Parameterizations for each covariate in CeresMimic model."""

    copynumber_cov: bool = False
    sgrna_intercept_cov: bool = False


class CeresMimic(SpecletModel):
    """CERES Mimic.

    $$
    \\begin{aligned}
    lfc &\\sim q_s (h_g + d_{g,c} + \\beta_c C) + \\eta_b + o_s \\\\
    o_s &\\sim N(\\mu_o, \\sigma_o)[\\text{gene}] \\\\
    \\end{aligned}
    $$

    where:

    - s: sgRNA
    - g: gene
    - c: cell line
    - b: batch
    - C: copy number (input data)

    This is a mimic of the CERES model with an additional parameter for pDNA batch.
    There are two optional parameters, `copynumber_cov` and `sgrna_intercept_cov`, to
    control the inclusion of \\(\\beta_c\\) and \\(o_s\\), respectively.
    """

    config: CeresMimicConfiguration

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[SpecletModelDataManager] = None,
        config: Optional[CeresMimicConfiguration] = None,
    ):
        """Create a CeresMimic object.

        Args:
            name (str): A unique identifier for this instance of CeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (SpecletSixConfiguration, optional): Model configurations.
        """
        if data_manager is None:
            data_manager = CrisprScreenDataManager(
                DataFile.DEPMAP_CRC_SUBSAMPLE,
                transformations=[achelp.set_achilles_categorical_columns],
            )

        self.config = config if config is not None else CeresMimicConfiguration()

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def set_config(self, info: dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        new_config = CeresMimicConfiguration(**info)
        if self.config is not None and self.config != new_config:
            logger.info("Setting model-specific configuration.")
            self.config = new_config
            self.model = None

    @property
    def copynumber_cov(self) -> bool:
        """Whether or not the copy number covariate is included in the model.

        Returns:
            bool: Whether or not the copy number covariate is included in the model.
        """
        return self.config.copynumber_cov

    @copynumber_cov.setter
    def copynumber_cov(self, new_value: bool) -> None:
        """Setter to control whether the copy number covariate should be in the model.

        If the value changes, then the `model` attribute and model results attributes
        `advi_results` and `mcmc_results` are all reset to None.

        Args:
            new_value (bool): Whether or not the copy number covariate should be
              included in the model.
        """
        if new_value != self.config.copynumber_cov:
            self._reset_model_and_results()
            self.config.copynumber_cov = new_value

    @property
    def sgrna_intercept_cov(self) -> bool:
        """Whether or not the `sgRNA|gene` varying intercept covariate is in the model.

        Returns:
            bool: Whether or not the `sgRNA|gene` varying intercept covariate is
              included in the model.
        """
        return self.config.sgrna_intercept_cov

    @sgrna_intercept_cov.setter
    def sgrna_intercept_cov(self, new_value: bool) -> None:
        """Control if the `sgRNA|gene` varying intercept covariate is in the model.

        If the value changes, then the `model` attribute and model results attributes
        `advi_results` and `mcmc_results` are all reset to None.

        Args:
            new_value (bool): Whether or not the `sgRNA|gene` varying intercept
              covariate is included in the model.
        """
        if new_value != self.config.sgrna_intercept_cov:
            self._reset_model_and_results()
            self.config.sgrna_intercept_cov = new_value

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Build CRC CERES Mimic One.

        Returns:
            Tuple[pm.Model, ObservedVarName]: Model and name of observed variable.
        """
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        # Shared Theano variables
        sgrna_idx_shared = theano.shared(co_idx.sgrna_idx)
        sgrna_to_gene_idx_shared = theano.shared(co_idx.sgrna_to_gene_idx)
        gene_idx_shared = theano.shared(co_idx.gene_idx)
        cellline_idx_shared = theano.shared(co_idx.cellline_idx)
        batch_idx_shared = theano.shared(b_idx.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)
        copynumber_shared = theano.shared(data.copy_number.values)

        with pm.Model() as model:

            # Hyper-priors
            σ_a = pm.HalfNormal("σ_a", np.array([0.1, 2]), shape=2)
            a = pm.Exponential("a", σ_a, shape=(co_idx.n_genes, 2))

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
                shape=co_idx.n_sgrnas,
            )
            h = pm.Normal("h", μ_h, σ_h, shape=co_idx.n_genes)
            d = pm.Normal(
                "d",
                μ_d,
                σ_d,
                shape=(co_idx.n_genes, co_idx.n_celllines),
            )
            η = pm.Normal("η", μ_η, σ_η, shape=b_idx.n_batches)

            gene_comp = h[gene_idx_shared] + d[gene_idx_shared, cellline_idx_shared]

            if self.copynumber_cov:
                # Add varying slope with copy number.
                μ_β = pm.Normal("μ_β", 0, 0.1)
                σ_β = pm.HalfNormal("σ_β", 0.5)
                β = pm.Normal("β", μ_β, σ_β, shape=co_idx.n_celllines)
                gene_comp += β[cellline_idx_shared] * copynumber_shared

            _mu = q[sgrna_idx_shared] * gene_comp + η[batch_idx_shared]

            if self.sgrna_intercept_cov:
                # Hyper priors for sgRNA|gene varying intercept.
                μ_og = pm.Normal("μ_og", 0, 0.1)
                σ_og = pm.HalfNormal("σ_og", 0.2)
                # Priors for sgRNA|gene varying intercept.
                μ_o = pm.Normal("μ_o", μ_og, σ_og, shape=co_idx.n_genes)
                σ_o = pm.HalfNormal("σ_o", 0.2)
                # sgRNA|gene varying intercept.
                o = pm.Normal(
                    "o",
                    μ_o[sgrna_to_gene_idx_shared],
                    σ_o,
                    shape=co_idx.n_sgrnas,
                )
                _mu += o[sgrna_idx_shared]

            μ = pm.Deterministic("μ", _mu)

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
            AttributeError: Raised if there are no shared variables.

        Returns:
            ReplacementsDict: A dictionary mapping new data to shared variables.
        """
        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        data = self.data_manager.get_data()
        batch_size = self._get_batch_size()
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        sgrna_idx_batch = pm.Minibatch(co_idx.sgrna_idx, batch_size=batch_size)
        gene_idx_batch = pm.Minibatch(co_idx.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(co_idx.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(b_idx.batch_idx, batch_size=batch_size)
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
