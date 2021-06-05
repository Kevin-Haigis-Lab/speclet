"""Speclet Model Five."""

from pathlib import Path
from typing import Optional, Tuple

import pymc3 as pm
import theano

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


class SpecletFive(SpecletModel):
    """SpecletFive Model.

      lfc ~ i + a_g + d_c + h_[g,c] + j_b

    where
        - g: gene
        - c: cell line
        - b: batch

    The model is relatively simple. It has a single global intercept (i) and two varying
    intercepts for gene (a) and cell line (d) and for varying effects per gene per cell
    line (h). Finally, there is a coefficient for batch effect (j).
    """

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
    ) -> None:
        """Instantiate a SpecletFive model.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
        """
        logger.debug("Instantiating a SpecletFive model.")
        if data_manager is None:
            logger.debug("Creating a data manager since none was supplied.")
            data_manager = CrcDataManager(debug=debug)

        super().__init__(
            name="speclet-five_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletFour model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        gene_idx_shared = theano.shared(co_idx.gene_idx)
        cellline_idx_shared = theano.shared(co_idx.cellline_idx)
        batch_idx_shared = theano.shared(b_idx.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)

        logger.info("Creating PyMC3 model.")

        with pm.Model() as model:
            # Varying batch intercept.
            μ_j = pm.Normal("μ_j", 0, 0.2)
            σ_j = pm.HalfNormal("σ_j", 1)
            j_offset = pm.Normal("j_offset", 0, 1, shape=b_idx.n_batches)
            j = pm.Deterministic("j", μ_j + j_offset * σ_j)

            # Varying gene and cell line intercept.
            μ_h = pm.Normal("μ_h", 0, 0.2)
            σ_h = pm.HalfNormal("σ_h", 1)
            h_offset = pm.Normal(
                "h_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_celllines)
            )
            h = pm.Deterministic("h", μ_h + h_offset * σ_h)

            # Varying cell line intercept.
            μ_d = pm.Normal("μ_d", 0, 0.2)
            σ_d = pm.HalfNormal("σ_d", 1)
            d_offset = pm.Normal("d_offset", 0, 1, shape=co_idx.n_celllines)
            d = pm.Deterministic("d", μ_d + d_offset * σ_d)

            # Varying gene intercept.
            μ_a = pm.Normal("μ_a", 0, 1)
            σ_a = pm.HalfNormal("σ_a", 1)
            a_offset = pm.Normal("a_offset", 0, 1, shape=co_idx.n_genes)
            a = pm.Deterministic("a", μ_a + a_offset * σ_a)

            # Global intercept.
            i = pm.Normal("i", 0, 1)

            μ = (
                i
                + a[gene_idx_shared]
                + d[cellline_idx_shared]
                + h[gene_idx_shared, cellline_idx_shared]
                + j[batch_idx_shared]
            )

            # Standard deviation of log-fold change, varies per batch.
            σ_σ = pm.HalfNormal("σ_σ", 1)
            σ = pm.HalfNormal("σ", σ_σ, shape=b_idx.n_batches)

            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                σ[batch_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )

        logger.debug("Finished building model.")

        self.shared_vars = {
            "gene_idx_shared": gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "lfc_shared": lfc_shared,
        }

        return model, "lfc"

    def get_replacement_parameters(self) -> ReplacementsDict:
        """Make a dictionary mapping the shared data variables to new data.

        Raises:
            AttributeError: Raised if there is no data manager.
            AttributeError: Raised if there are no shared variables.

        Returns:
            ReplacementsDict: A dictionary mapping new data to shared variables.
        """
        logger.debug("Making dictionary of replacement parameters.")
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
        idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        gene_idx_batch = pm.Minibatch(idx.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(idx.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(b_idx.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

        return {
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }

    def update_mcmc_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        logger.info("Updating the MCMC sampling parameters.")
        self.mcmc_sampling_params.draws = 4000
        self.mcmc_sampling_params.tune = 2000
        self.mcmc_sampling_params.target_accept = 0.99
        return None

    def update_advi_sampling_parameters(self) -> None:
        """Adjust the ADVI parameters depending on the state of the object."""
        logger.info("Updating the ADVI fitting parameters.")
        parameter_adjustment_map = {True: 40000, False: 100000}
        self.advi_sampling_params.n_iterations = parameter_adjustment_map[self.debug]
        return None
