"""First Model Three."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


class SpecletThree(SpecletModel):
    """SpecletThree Model.

    Model with the following covariates:
    - h: consistent gene effect
    - g: cell line-specific gene effect [gene x cell line]
    - b: batch effect
    """

    _noncentered_param: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
        noncentered_param: bool = True,
    ):
        """Instantiate a SpecletThree model.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
            noncentered_param (bool, optional): Should the model use a non-centered
              parameterization? Default to True.
        """
        if data_manager is None:
            data_manager = CrcDataManager(debug=debug)

        super().__init__(
            name="speclet-three_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )
        self._noncentered_param = noncentered_param

    def __str__(self) -> str:
        """Describe the object.

        Returns:
            str: String description of the object.
        """
        msg = super().__str__()
        return msg

    @property
    def noncentered_param(self) -> bool:
        """Value of `noncentered_param` attribute."""
        return self._noncentered_param

    @noncentered_param.setter
    def noncentered_param(self, new_value: bool) -> None:
        """Set the value of `noncentered_param` attribute.

        If the new value is different, all model and sampling results are reset.
        """
        if new_value != self._noncentered_param:
            logger.info(f"Changing `_noncentered_param` attribute to '{new_value}'.")
            self._noncentered_param = new_value
            self._reset_model_and_results()

    def _common_model_components(
        self, common_indices: achelp.CommonIndices
    ) -> pm.Model:
        with pm.Model() as model:
            # Gene varying intercept.
            μ_h = pm.Normal("μ_h", 0, 1)  # noqa: F841
            σ_h = pm.HalfNormal("σ_h", 1)  # noqa: F841

            # [gene, cell line] varying intercept.
            μ_g = pm.Normal("μ_g", 0, 0.2)  # noqa: F841
            σ_g = pm.HalfNormal("σ_g", 1)  # noqa: F841

            # Batch effect varying intercept.
            μ_b = pm.Normal("μ_b", 0, 0.2)  # noqa: F841
            σ_b = pm.HalfNormal("σ_b", 0.5)  # noqa: F841

            σ_σ = pm.HalfNormal("σ_σ", 0.5)
            σ = pm.HalfNormal("σ", σ_σ, shape=common_indices.n_sgrnas)  # noqa: F841

        return model

    def _model_centered_parameterization(
        self,
        idx: achelp.CommonIndices,
        total_size: int,
        sgrna_idx_shared: TTShared,
        gene_idx_shared: TTShared,
        cellline_idx_shared: TTShared,
        batch_idx_shared: TTShared,
        lfc_shared: TTShared,
    ) -> pm.Model:
        model = self._common_model_components(common_indices=idx)
        with model:
            # Gene varying intercept.
            h = pm.Normal("h", model["μ_h"], model["σ_h"], shape=idx.n_genes)
            # [gene, cell line] varying intercept.
            g = pm.Normal(
                "g", model["μ_g"], model["σ_g"], shape=(idx.n_genes, idx.n_celllines)
            )
            # Batch effect varying intercept.
            b = pm.Normal("b", model["μ_b"], model["σ_b"], shape=idx.n_batches)

            μ = pm.Deterministic(
                "μ",
                h[gene_idx_shared]
                + g[gene_idx_shared, cellline_idx_shared]
                + b[batch_idx_shared],
            )

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                model["σ"][sgrna_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )
        return model

    def _model_non_centered_parameterization(
        self,
        idx: achelp.CommonIndices,
        total_size: int,
        sgrna_idx_shared: TTShared,
        gene_idx_shared: TTShared,
        cellline_idx_shared: TTShared,
        batch_idx_shared: TTShared,
        lfc_shared: TTShared,
    ) -> pm.Model:
        model = self._common_model_components(common_indices=idx)
        with model:
            # Gene varying intercept.
            h_offset = pm.Normal("h_offset", 0, 1, shape=idx.n_genes)
            h = pm.Deterministic("h", model["μ_h"] + h_offset * model["σ_h"])

            # [gene, cell line] varying intercept.
            g_offset = pm.Normal("g_offset", 0, 1, shape=(idx.n_genes, idx.n_celllines))
            g = pm.Deterministic("g", model["μ_g"] + g_offset * model["σ_g"])

            # Batch effect varying intercept.
            b_offset = pm.Normal("b_offset", 0, 1, shape=idx.n_batches)
            b = pm.Deterministic("b", model["μ_b"] + b_offset * model["σ_b"])

            μ = pm.Deterministic(
                "μ",
                h[gene_idx_shared]
                + g[gene_idx_shared, cellline_idx_shared]
                + b[batch_idx_shared],
            )

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                model["σ"][sgrna_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )
        return model

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletThree model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        idx = achelp.common_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = theano.shared(idx.sgrna_idx)
        gene_idx_shared = theano.shared(idx.gene_idx)
        cellline_idx_shared = theano.shared(idx.cellline_idx)
        batch_idx_shared = theano.shared(idx.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)

        if self.noncentered_param:
            logger.info("Creating PyMC3 model (non-centered parameterization).")
            model = self._model_non_centered_parameterization(
                idx=idx,
                total_size=total_size,
                sgrna_idx_shared=sgrna_idx_shared,
                gene_idx_shared=gene_idx_shared,
                cellline_idx_shared=cellline_idx_shared,
                batch_idx_shared=batch_idx_shared,
                lfc_shared=lfc_shared,
            )
        else:
            logger.info("Creating PyMC3 model (centered parameterization).")
            model = self._model_centered_parameterization(
                idx=idx,
                total_size=total_size,
                sgrna_idx_shared=sgrna_idx_shared,
                gene_idx_shared=gene_idx_shared,
                cellline_idx_shared=cellline_idx_shared,
                batch_idx_shared=batch_idx_shared,
                lfc_shared=lfc_shared,
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
        idx = achelp.common_indices(data)

        sgrna_idx_batch = pm.Minibatch(idx.sgrna_idx, batch_size=batch_size)
        gene_idx_batch = pm.Minibatch(idx.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(idx.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(idx.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
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
        parameter_adjustment_map = {
            True: 40000,
            False: 100000,
        }
        self.advi_sampling_params.n_iterations = parameter_adjustment_map[self.debug]
        return None
