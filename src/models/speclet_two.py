"""First new model for the speclet project."""

from pathlib import Path
from typing import Optional, Tuple

import pymc3 as pm
import theano

from src.data_processing import achilles as achelp
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


class SpecletTwo(SpecletModel):
    """SpecletTwo Model.

    This is a simple model with varying intercepts for [gene, cell line], KRAS allele,
    and batch.
    """

    _kras_cov: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = CrcDataManager(),
        kras_cov: bool = False,
    ):
        """Instantiate a SpecletTwo model.

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
        """
        super().__init__(
            name="speclet-two_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )
        self._kras_cov = kras_cov
        self.mcmc_sampling_params.draws = 2000
        self.mcmc_sampling_params.tune = 2000

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
            self._kras_cov = new_value
            self._reset_model_and_results()

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletTwo model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        ic = achelp.common_indices(data)

        # Shared Theano variables
        sgrna_idx_shared = theano.shared(ic.sgrna_idx)
        gene_idx_shared = theano.shared(ic.gene_idx)
        cellline_idx_shared = theano.shared(ic.cellline_idx)
        kras_idx_shared = theano.shared(ic.kras_mutation_idx)
        batch_idx_shared = theano.shared(ic.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)

        with pm.Model() as model:
            # [gene, cell line] varying intercept.
            μ_ɑ = pm.Normal("μ_ɑ", 0, 5)
            σ_ɑ = pm.HalfNormal("σ_ɑ", 5)
            ɑ = pm.Normal("ɑ", μ_ɑ, σ_ɑ, shape=(ic.n_genes, ic.n_celllines))

            # Batch effect varying intercept.
            μ_η = pm.Normal("μ_η", 0, 0.1)
            σ_η = pm.HalfNormal("σ_η", 0.1)
            η = pm.Normal("η", μ_η, σ_η, shape=ic.n_batches)

            _mu = ɑ[gene_idx_shared, cellline_idx_shared] + η[batch_idx_shared]

            if self.kras_cov:
                # Varying effect for KRAS mutation.
                μ_β = pm.Normal("μ_β", 0, 1)
                σ_β = pm.HalfNormal("σ_β", 1)
                β = pm.Normal("β", μ_β, σ_β, shape=ic.n_kras_mutations)
                _mu += β[kras_idx_shared]

            μ = pm.Deterministic("μ", _mu)

            σ_σ = pm.HalfNormal("σ_σ", 1)
            σ = pm.HalfNormal("σ", σ_σ, shape=ic.n_sgrnas)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                σ[sgrna_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )

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
        ic = achelp.common_indices(data)

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
