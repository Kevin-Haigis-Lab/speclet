"""Speclet Model Four."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymc3 as pm
import theano
from pydantic import BaseModel
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel
from src.project_enums import ModelParameterization as MP


class SpecletFourConfiguration(BaseModel):
    """Parameterizations for each covariate in SpecletFour model."""

    copy_number_cov: bool = False
    h: MP = MP.CENTERED
    d: MP = MP.CENTERED
    β: MP = MP.CENTERED
    η: MP = MP.CENTERED


class SpecletFour(SpecletModel):
    """SpecletFour Model.

    $$
    lfc \\sim h_g + d_{g,c} + \\beta_c C + \\eta_b
    $$

    where:

    - g: gene
    - c: cell line
    - b: batch
    - C: copy number (input data)

    A simple model with a separate consistent gene effect \\(h_g\\) and cell-line
    varying gene effect \\(d_{g,c}\\). The coefficient for copy number effect
    \\(\\beta_c\\) varies by cell line and is optional.
    """

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
        config: SpecletFourConfiguration = SpecletFourConfiguration(),
    ):
        """Instantiate a SpecletFour model.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
            config (SpecletFourConfiguration, optional): Model configuration.
        """
        if data_manager is None:
            data_manager = CrcDataManager(debug=debug)

        self.config = config

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def set_config(self, info: Dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        logger.info("Setting model-specific configuration.")
        self.config = SpecletFourConfiguration(**info)

    def _model_specification(
        self,
        co_idx: achelp.CommonIndices,
        batch_idx: achelp.DataBatchIndices,
        sgrna_idx_shared: TTShared,
        gene_idx_shared: TTShared,
        cellline_idx_shared: TTShared,
        batch_idx_shared: TTShared,
        cn_shared: TTShared,
        lfc_shared: TTShared,
        total_size: int,
    ) -> Tuple[pm.Model, str]:
        with pm.Model() as model:
            # Gene varying intercept.
            μ_h = pm.Normal("μ_h", 0, 1)
            σ_h = pm.HalfNormal("σ_h", 1)

            # [gene, cell line] varying intercept.
            μ_d = pm.Normal("μ_d", 0, 1)
            σ_d = pm.HalfNormal("σ_d", 1)

            # Batch effect varying intercept.
            μ_η = pm.Normal("μ_η", 0, 0.2)
            σ_η = pm.HalfNormal("σ_η", 0.5)

            # Copy number varying effect.
            if self.config.copy_number_cov:
                μ_β = pm.Normal("μ_β", -0.5, 1)
                σ_β = pm.Normal("σ_β", -0.5, 1)

            # Gene varying intercept.
            if self.config.h is MP.NONCENTERED:
                h_offset = pm.Normal("h_offset", 0, 1, shape=co_idx.n_genes)
                h = pm.Deterministic("h", μ_h + h_offset * σ_h)
            else:
                h = pm.Normal("h", μ_h, σ_h, shape=co_idx.n_genes)

            # [gene, cell line] varying intercept.
            if self.config.d is MP.NONCENTERED:
                d_offset = pm.Normal(
                    "d_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_celllines)
                )
                d = pm.Deterministic("d", μ_d + d_offset * σ_d)
            else:
                d = pm.Normal("d", μ_d, σ_d, shape=(co_idx.n_genes, co_idx.n_celllines))

            # Batch effect varying intercept.
            if self.config.η is MP.NONCENTERED:
                η_offset = pm.Normal("η_offset", 0, 1, shape=batch_idx.n_batches)
                η = pm.Deterministic("η", μ_η + η_offset * σ_η)
            else:
                η = pm.Normal("η", μ_η, σ_η, shape=batch_idx.n_batches)

            _μ = (
                h[gene_idx_shared]
                + d[gene_idx_shared, cellline_idx_shared]
                + η[batch_idx_shared]
            )

            # Copy number effect varying by cell line.
            if self.config.copy_number_cov:
                if self.config.β is MP.NONCENTERED:
                    β_offset = pm.Normal("β_offset", 0, 1, shape=co_idx.n_celllines)
                    β = pm.Deterministic("β", μ_β + β_offset * σ_β)
                else:
                    β = pm.Normal("β", μ_β, σ_β, shape=co_idx.n_celllines)

                _μ += β[cellline_idx_shared] * cn_shared

            μ = pm.Deterministic("μ", _μ)

            σ_σ = pm.HalfNormal("σ_σ", 0.5)
            σ = pm.HalfNormal("σ", σ_σ, shape=co_idx.n_sgrnas)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                σ[sgrna_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )

        return model, "lfc"

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletFour model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")

        assert self.data_manager is not None
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = theano.shared(co_idx.sgrna_idx)
        gene_idx_shared = theano.shared(co_idx.gene_idx)
        cellline_idx_shared = theano.shared(co_idx.cellline_idx)
        batch_idx_shared = theano.shared(b_idx.batch_idx)
        cn_shared = theano.shared(data.copy_number.values)
        lfc_shared = theano.shared(data.lfc.values)

        logger.info("Saving shared variables to dictionary.")
        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "cn_shared": cn_shared,
            "lfc_shared": lfc_shared,
        }

        logger.info("Creating PyMC3 model.")
        model, obs_var_name = self._model_specification(
            co_idx=co_idx,
            batch_idx=b_idx,
            sgrna_idx_shared=sgrna_idx_shared,
            gene_idx_shared=gene_idx_shared,
            cellline_idx_shared=cellline_idx_shared,
            batch_idx_shared=batch_idx_shared,
            cn_shared=cn_shared,
            lfc_shared=lfc_shared,
            total_size=total_size,
        )
        logger.debug("Finished building model.")
        return model, obs_var_name

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
        batch_size = self.data_manager.get_batch_size()
        co_idx = achelp.common_indices(data)
        batch_idx = achelp.data_batch_indices(data)

        gene_idx_batch = pm.Minibatch(co_idx.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(co_idx.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(batch_idx.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

        return {
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }

    def get_advi_callbacks(self) -> List[Any]:
        """Prepare a list of callbacks for ADVI fitting.

        This can be overridden by subclasses to apply custom callbacks or change the
        parameters of the CheckParametersConvergence callback.

        Returns:
            List[Any]: List of callbacks.
        """
        logger.debug("Custom ADVI callbacks.")
        return [
            pm.callbacks.CheckParametersConvergence(
                every=10, tolerance=0.01, diff="relative"
            )
        ]

    def update_mcmc_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        self.mcmc_sampling_params.draws = 4000
        self.mcmc_sampling_params.tune = 2000
        self.mcmc_sampling_params.target_accept = 0.99
        return None

    def update_advi_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        parameter_adjustment_map: Dict[bool, int] = {True: 40000, False: 100000}
        self.advi_sampling_params.n_iterations = parameter_adjustment_map[self.debug]
        return None
