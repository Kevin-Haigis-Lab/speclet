"""Speclet Model Two."""

from pathlib import Path
from typing import Any, Optional

import pymc3 as pm
import theano

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import (
    CrisprScreenDataManager,
    common_crispr_screen_transformations,
)
from src.models.speclet_model import (
    ObservedVarName,
    ReplacementsDict,
    SpecletModel,
    SpecletModelDataManager,
)


class SpecletTwo(SpecletModel):
    """SpecletTwo Model.

    $$
    lfc \\sim \\alpha_{g,c} + \\eta_b
    $$

    where:

    - g: gene
    - c: cell line
    - b: batch

    This is a simple model with varying intercepts for [gene|cell line] and batch.
    """

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        data_manager: Optional[SpecletModelDataManager] = None,
    ):
        """Instantiate a SpecletTwo model.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
        """
        if data_manager is None:
            data_manager = CrisprScreenDataManager(
                DataFile.DEPMAP_CRC_SUBSAMPLE,
                transformations=common_crispr_screen_transformations,
            )

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            data_manager=data_manager,
        )

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Build SpecletTwo model.

        Returns:
            Tuple[pm.Model, ObservedVarName]: The model and name of the observed
            variable.
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
        lfc_shared = theano.shared(data.lfc.values)

        logger.info("Creating PyMC3 model.")
        with pm.Model() as model:
            # [gene, cell line] varying intercept.
            μ_α = pm.Normal("μ_α", 0, 2)
            σ_α = pm.HalfNormal("σ_α", 1)
            α = pm.Normal("α", μ_α, σ_α, shape=(co_idx.n_genes, co_idx.n_celllines))

            # Batch effect varying intercept.
            μ_η = pm.Normal("μ_η", 0, 1)
            σ_η = pm.HalfNormal("σ_η", 1)
            η = pm.Normal("η", μ_η, σ_η, shape=b_idx.n_batches)

            μ = pm.Deterministic(
                "μ", α[gene_idx_shared, cellline_idx_shared] + η[batch_idx_shared]
            )

            σ_σ = pm.HalfNormal("σ_σ", 1)
            σ = pm.HalfNormal("σ", σ_σ, shape=co_idx.n_sgrnas)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                σ[sgrna_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )

        logger.debug("Finished building model.")
        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "lfc_shared": lfc_shared,
        }
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

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }

    def get_advi_callbacks(self) -> list[Any]:
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
