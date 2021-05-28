"""First new model for the speclet project."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


class SpecletFour(SpecletModel):
    """SpecletFour Model.

    Model with the following covariates:
    - h: consistent gene effect
    - g: cell line-specific gene effect [gene x cell line] and the mean of the prior
         varies by [gene x KRAS allele] of the cell line.
    - o: CN effect varying per cell line
    - b: batch effect

    Copy number values are divided by 2.0 to make diploid values equal 1. Therefore,
    the varying copy number effect will also act as a varying effect for the cell line.
    """

    _kras_mutation_minimum: int
    _noncentered_param: bool
    _copy_number_cov: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = CrcDataManager(),
        kras_mutation_minimum: int = 3,
        copy_number_cov: bool = False,
        noncentered_param: bool = True,
    ):
        """Instantiate a SpecletFour model.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. Defaults to None.
            kras_mutation_minimum (int, optional): The minimum number of cell lines with
              a KRAS allele for the allele to be included as a separate group. Defaults
              to 3.
            copy_number_cov (bool, optional): Shoud the covariate for gene copy number
              effect be included in the model? The covariate varies by cell line.
              Defaults to False.
            noncentered_param (bool, optional): Should the model use a non-centered
              parameterization? Default to True.
        """
        super().__init__(
            name="speclet-four_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )
        self._kras_mutation_minimum = kras_mutation_minimum
        self._noncentered_param = noncentered_param
        self._copy_number_cov = copy_number_cov

    @property
    def kras_mutation_minimum(self) -> int:
        """Value of `kras_mutation_minimum` attribute."""
        return self._kras_mutation_minimum

    @kras_mutation_minimum.setter
    def kras_mutation_minimum(self, new_value: int) -> None:
        """Set the value of `kras_mutation_minimum` attribute.

        If the new value is different, all model and sampling results are reset.
        """
        if new_value != self._kras_mutation_minimum:
            logger.info(f"Changing `kras_mutation_minimum` attribute to '{new_value}'.")
            self._kras_mutation_minimum = new_value
            self._reset_model_and_results()

    @property
    def copy_number_cov(self) -> bool:
        """Value of `copy_number_cov` attribute."""
        return self._copy_number_cov

    @copy_number_cov.setter
    def copy_number_cov(self, new_value: bool) -> None:
        """Set the value of `copy_number_cov` attribute.

        If the new value is different, all model and sampling results are reset.
        """
        if new_value != self._copy_number_cov:
            logger.info(f"Changing `copy_number_cov` attribute to '{new_value}'.")
            self._copy_number_cov = new_value
            self._reset_model_and_results()

    @property
    def noncentered_param(self) -> bool:
        """Value of `kras_cov` attribute."""
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

    def _common_model_components(self, n_sgrnas: int) -> pm.Model:
        with pm.Model() as model:
            # Gene varying intercept.
            μ_h = pm.Normal("μ_h", 0, 1)  # noqa: F841
            σ_h = pm.HalfNormal("σ_h", 1)  # noqa: F841

            # [gene, cell line] varying intercept.
            μ_μ_g = pm.Normal("μ_μ_g", 0, 0.2)  # noqa: F841
            σ_μ_g = pm.HalfNormal("σ_μ_g", 1)  # noqa: F841

            # Batch effect varying intercept.
            μ_b = pm.Normal("μ_b", 0, 0.2)  # noqa: F841
            σ_b = pm.HalfNormal("σ_b", 0.5)  # noqa: F841

            σ_σ = pm.HalfNormal("σ_σ", 0.5)
            σ = pm.HalfNormal("σ", σ_σ, shape=n_sgrnas)  # noqa: F841

        return model

    def _model_centered_parameterization(
        self,
        model: pm.Model,
        co_idx: achelp.CommonIndices,
        unco_idx: achelp.UncommonIndices,
        cellline_to_kras_idx_shared: TTShared,
    ) -> pm.Model:
        with model:
            # Gene varying intercept.
            h = pm.Normal(  # noqa: F841
                "h", model["μ_h"], model["σ_h"], shape=co_idx.n_genes
            )

            # [gene, cell line] varying intercept.
            μ_g = pm.Normal(
                "μ_g",
                model["μ_μ_g"],
                model["σ_μ_g"],
                shape=(co_idx.n_genes, unco_idx.n_kras_mutations),
            )
            σ_g = pm.HalfNormal("σ_g", 1)
            g = pm.Normal(  # noqa: F841
                "g",
                μ_g[:, cellline_to_kras_idx_shared],
                σ_g,
                shape=(co_idx.n_genes, co_idx.n_celllines),
            )

            if self.copy_number_cov:
                μ_o = pm.Normal("μ_o", -0.5, 1)
                σ_o = pm.Normal("σ_o", -0.5, 1)
                o = pm.Normal("o", μ_o, σ_o, shape=co_idx.n_celllines)  # noqa: F841

            # Batch effect varying intercept.
            b = pm.Normal(  # noqa: F841
                "b", model["μ_b"], model["σ_b"], shape=co_idx.n_batches
            )
        return model

    def _model_non_centered_parameterization(
        self,
        model: pm.Model,
        co_idx: achelp.CommonIndices,
        unco_idx: achelp.UncommonIndices,
        cellline_to_kras_idx_shared: TTShared,
    ) -> pm.Model:
        with model:
            # Gene varying intercept.
            h_offset = pm.Normal("h_offset", 0, 1, shape=co_idx.n_genes)
            h = pm.Deterministic(  # noqa: F841
                "h", model["μ_h"] + h_offset * model["σ_h"]
            )

            # [gene, cell line] varying intercept.
            μ_g_offset = pm.Normal(
                "μ_g_offset", 0, 1, shape=(co_idx.n_genes, unco_idx.n_kras_mutations)
            )
            μ_g = pm.Deterministic("μ_g", model["μ_μ_g"] + μ_g_offset * model["σ_μ_g"])
            σ_g = pm.HalfNormal("σ_g", 1)
            g_offset = pm.Normal(
                "g_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_celllines)
            )
            g = pm.Deterministic(  # noqa: F841
                "g", μ_g[:, cellline_to_kras_idx_shared] + g_offset * σ_g
            )

            if self.copy_number_cov:
                μ_o = pm.Normal("μ_o", -0.5, 1)
                σ_o = pm.Normal("σ_o", -0.5, 1)
                o_offset = pm.Normal("o_offset", 0, 1, shape=co_idx.n_celllines)
                o = pm.Deterministic("o", μ_o + o_offset * σ_o)  # noqa: F841

            # Batch effect varying intercept.
            b_offset = pm.Normal("b_offset", 0, 1, shape=co_idx.n_batches)
            b = pm.Deterministic(  # noqa: F841
                "b", model["μ_b"] + b_offset * model["σ_b"]
            )
        return model

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletFour model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data, min_kras_muts=self.kras_mutation_minimum)
        unco_idx = achelp.uncommon_indices(
            data, min_kras_muts=self.kras_mutation_minimum
        )

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = theano.shared(co_idx.sgrna_idx)
        gene_idx_shared = theano.shared(co_idx.gene_idx)
        cellline_idx_shared = theano.shared(co_idx.cellline_idx)
        cellline_to_kras_idx_shared = theano.shared(
            unco_idx.cellline_to_kras_mutation_idx
        )
        batch_idx_shared = theano.shared(co_idx.batch_idx)
        cn_shared = theano.shared(data.gene_cn.values / 2.0)
        lfc_shared = theano.shared(data.lfc.values)

        logger.info("Creating PyMC3 model.")
        model = self._common_model_components(n_sgrnas=co_idx.n_sgrnas)
        if self.noncentered_param:
            logger.info("Using non-centered parameterization.")
            model = self._model_non_centered_parameterization(
                model=model,
                co_idx=co_idx,
                unco_idx=unco_idx,
                cellline_to_kras_idx_shared=cellline_to_kras_idx_shared,
            )
        else:
            logger.info("Using centered parameterization.")
            model = self._model_centered_parameterization(
                model=model,
                co_idx=co_idx,
                unco_idx=unco_idx,
                cellline_to_kras_idx_shared=cellline_to_kras_idx_shared,
            )

        with model:
            _μ = (
                model["h"][gene_idx_shared]
                + model["g"][gene_idx_shared, cellline_idx_shared]
                + model["b"][batch_idx_shared]
            )

            if self.copy_number_cov:
                _μ += model["o"][cellline_idx_shared] * cn_shared

            μ = pm.Deterministic("μ", _μ)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                model["σ"][sgrna_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )

        logger.debug("Finished building model.")

        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "cellline_to_kras_idx_shared": cellline_to_kras_idx_shared,
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
        ic = achelp.common_indices(data, min_kras_muts=self.kras_mutation_minimum)

        gene_idx_batch = pm.Minibatch(ic.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(ic.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(ic.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

        uc = achelp.uncommon_indices(data, min_kras_muts=self.kras_mutation_minimum)
        cellline_to_kras_idx = pm.Minibatch(
            uc.cellline_to_kras_mutation_idx, batch_size=batch_size
        )

        return {
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["cellline_to_kras_idx_shared"]: cellline_to_kras_idx,
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
