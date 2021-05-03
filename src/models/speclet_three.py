"""First new model for the speclet project."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pymc3 as pm
import theano

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


class SpecletThree(SpecletModel):
    """SpecletThree Model.

    Model with the following covariates:
    - h: consistent gene effect
    - g: cell line-specific gene effect [gene x cell line]
    - a: KRAS allele-specific gene effect [gene x KRAS allele]
    - b: batch effect
    """

    _kras_cov: bool
    _kras_mutation_minimum: int

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = CrcDataManager(),
        kras_cov: bool = False,
        kras_mutation_minimum: int = 3,
    ):
        """Instantiate a SpecletThree model.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. Defaults to None.
            kras_cov (bool, optional): Should the KRAS allele covariate be included in
              the model? Default to False.
            kras_mutation_minimum (int, optional): The minimum number of cell lines with
              a KRAS allele for the allele to be included as a separate group. Defaults
              to 3.

        """
        super().__init__(
            name="speclet-three_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )
        self._kras_cov = kras_cov
        self._kras_mutation_minimum = kras_mutation_minimum

    def __str__(self) -> str:
        """Describe the object.

        Returns:
            str: String description of the object.
        """
        msg = super().__str__()
        with_kras_msg = "with" if self.kras_cov else "no"
        msg += f"\n  -> {with_kras_msg} KRAS cov."
        return msg

    @property
    def kras_cov(self) -> bool:
        """Value of `kras_cov` attribute."""
        return self._kras_cov

    @kras_cov.setter
    def kras_cov(self, new_value: bool) -> None:
        """Set the value of `kras_cov` attribute.

        If the new value is different, all model and sampling results are reset.
        """
        if new_value != self._kras_cov:
            logger.info(f"Changing `kras_cov` attribute to '{new_value}'.")
            self._kras_cov = new_value
            self._reset_model_and_results()

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

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletThree model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        ic = achelp.common_indices(data, min_kras_muts=self.kras_mutation_minimum)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = theano.shared(ic.sgrna_idx)
        gene_idx_shared = theano.shared(ic.gene_idx)
        cellline_idx_shared = theano.shared(ic.cellline_idx)
        kras_idx_shared = theano.shared(ic.kras_mutation_idx)
        batch_idx_shared = theano.shared(ic.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)

        logger.info("Creating PyMC3 model.")
        with pm.Model() as model:
            # Gene varying intercept.
            μ_h = pm.Normal("μ_h", 0, 1)
            σ_h = pm.HalfNormal("σ_h", 1)
            h = pm.Normal("h", μ_h, σ_h, shape=ic.n_genes)

            # [gene, cell line] varying intercept.
            μ_g = pm.Normal("μ_g", 0, 0.2)
            σ_g = pm.HalfNormal("σ_g", 1)
            g = pm.Normal("g", μ_g, σ_g, shape=(ic.n_genes, ic.n_celllines))

            # Batch effect varying intercept.
            μ_b = pm.Normal("μ_b", 0, 0.2)
            σ_b = pm.HalfNormal("σ_b", 0.5)
            b = pm.Normal("b", μ_b, σ_b, shape=ic.n_batches)

            _mu = (
                h[gene_idx_shared]
                + g[gene_idx_shared, cellline_idx_shared]
                + b[batch_idx_shared]
            )

            if self.kras_cov:
                # Varying effect for KRAS mutation.
                μ_a = pm.Normal("μ_a", 0, 0.2)
                σ_a = pm.HalfNormal("σ_a", 1)
                a = pm.Normal("a", μ_a, σ_a, shape=(ic.n_genes, ic.n_kras_mutations))
                _mu += a[gene_idx_shared, kras_idx_shared]

            μ = pm.Deterministic("μ", _mu)

            σ_σ = pm.HalfNormal("σ_σ", 0.5)
            σ = pm.HalfNormal("σ", σ_σ, shape=ic.n_sgrnas)

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
            "kras_idx_shared": kras_idx_shared,
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

        sgrna_idx_batch = pm.Minibatch(ic.sgrna_idx, batch_size=batch_size)
        gene_idx_batch = pm.Minibatch(ic.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(ic.cellline_idx, batch_size=batch_size)
        kras_idx_batch = pm.Minibatch(ic.kras_mutation_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(ic.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["kras_idx_shared"]: kras_idx_batch,
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
            True: {True: 40000, False: 40000},
            False: {True: 100000, False: 100000},
        }
        self.advi_sampling_params.n_iterations = parameter_adjustment_map[self.debug][
            self.kras_cov
        ]
        return None
