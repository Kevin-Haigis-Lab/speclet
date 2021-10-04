"""Speclet Model One."""

from pathlib import Path
from typing import Optional

import pymc3 as pm
import theano

from src.data_processing import achilles as achelp
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ObservedVarName, ReplacementsDict, SpecletModel


class SpecletOne(SpecletModel):
    """SpecletOne Model.

    $$
    \\begin{aligned}
    lfc &\\sim h_s + d_{s,c} + \\beta_c C + \\eta_b \\\\
    h_s &\\sim N(\\mu_h, \\sigma_h)[\\text{gene}] \\\\
    d_{s,c} &\\sim N(\\mu_d, \\sigma_d)[\\text{gene}|\\text{cell line}] \\\\
    \\end{aligned}
    $$

    where:

    - s: sgRNA
    - g: gene
    - c: cell line
    - b: batch
    - C: copy number (input data)

    This model is based on the CERES model, but removes the multiplicative sgRNA
    "activity" score due to issues of non-identifiability. Also, the consistent gene
    effect parameter has been replaced with a coefficient for sgRNA with a pooling prior
    per gene. Similarly, the cell-line specific gene effect coefficient \\(d_{s,c}\\)
    has been extended with a similar structure.
    """

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
    ):
        """Instantiate a SpecletOne model.

        Args:
            name (str): A unique identifier for this instance of SpecletOne. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
        """
        if data_manager is None:
            data_manager = CrcDataManager(debug=debug)

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Build SpecletOne model.

        Returns:
            Tuple[pm.Model, ObservedVarName]: The model and name of the observed
            variable.
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
            # sgRNA|gene varying intercept.
            μ_μ_h = pm.Normal("μ_μ_h", 0, 5)
            σ_μ_h = pm.HalfNormal("σ_μ_h", 5)
            μ_h = pm.Normal("μ_h", μ_μ_h, σ_μ_h, shape=co_idx.n_genes)
            σ_σ_h = pm.HalfNormal("σ_σ_h", 5)
            σ_h = pm.HalfNormal("σ_h", σ_σ_h, shape=co_idx.n_genes)
            h = pm.Normal(
                "h",
                μ_h[co_idx.sgrna_to_gene_idx],
                σ_h[co_idx.sgrna_to_gene_idx],
                shape=co_idx.n_sgrnas,
            )

            # [sgRNA|gene, cell line] varying intercept.
            μ_μ_d = pm.Normal("μ_μ_d", 0, 1)
            σ_μ_d = pm.Normal("σ_μ_d", 1)
            μ_d = pm.Normal(
                "μ_d", μ_μ_d, σ_μ_d, shape=(co_idx.n_genes, co_idx.n_celllines)
            )
            σ_σ_d = pm.HalfNormal("σ_σ_d", 0.2)
            σ_d = pm.HalfNormal(
                "σ_d", σ_σ_d, shape=(co_idx.n_genes, co_idx.n_celllines)
            )
            d = pm.Normal(
                "d",
                μ_d[sgrna_to_gene_idx_shared, :],
                σ_d[sgrna_to_gene_idx_shared, :],
                shape=(co_idx.n_sgrnas, co_idx.n_celllines),
            )

            # Varying slope per cell line for CN.
            μ_β = pm.Normal("μ_β", -1, 2)
            σ_β = pm.HalfNormal("σ_β", 1)
            β = pm.Normal("β", μ_β, σ_β, shape=co_idx.n_celllines)

            # Batch effect varying intercept.
            μ_η = pm.Normal("μ_η", 0, 0.1)
            σ_η = pm.HalfNormal("σ_η", 0.1)
            η = pm.Normal("η", μ_η, σ_η, shape=b_idx.n_batches)

            μ = pm.Deterministic(
                "μ",
                h[sgrna_idx_shared]
                + d[sgrna_idx_shared, cellline_idx_shared]
                + β[cellline_idx_shared] * copynumber_shared
                + η[batch_idx_shared],
            )

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
        batch_size = self.data_manager.get_batch_size()
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
