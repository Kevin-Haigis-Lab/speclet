"""Speclet Model Six."""

from pathlib import Path
from typing import Optional, Tuple

import pymc3 as pm
from theano import shared as ts

from src.data_processing import achilles as achelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


class SpecletSix(SpecletModel):
    """SpecletSix Model.

    $$
    \\begin{aligned}
    lfc &\\sim i + a_s + d_c + h_{g,c} + j_b +
    k_c C^{(c)} + n_g C^{(g)} + q_{g,l} R^{(g,l)} + m_{g,l} M \\\\
    a_s &\\sim N(μ_a, σ_a)[\\text{gene}] \\\\
    d_c &\\sim N(μ_d, σ_d)[\\text{lineage}] \\\\
    j_b &\\sim N(μ_j, σ_j)[\\text{source}] \\text{(if more than one source)}
    \\end{aligned}
    $$

    where:

    - s: sgRNA
    - g: gene
    - c: cell line
    - l: cell line lineage
    - b: batch
    - o: data source (Broad or Sanger)

    Below is a description of each parameter in the model:

    - a_s : sgRNA effect with hierarchical level for gene (g)
    - d_c: cell line effect with hierarchical level for lineage (l; if more than
                one is found)
    - j_b: data source (o; if more than one is found)
    - k_c: cell line effect of copy number (Cc: z-scaled per cell line)
    - n_g: gene effect of copy number (Cg: z-scaled per gene)
    - q_{g,l} : RNA effect varying per gene and cell line lineage (R[g,l]: z-scaled
                within each gene and lineage)
    - m_{g,l} : mutation effect varying per gene and cell line lineage (M: {0, 1})
    """

    _cell_line_cna_cov: bool
    _gene_cna_cov: bool
    _rna_cov: bool
    _mutation_cov: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
        cell_line_cna_cov: bool = False,
        gene_cna_cov: bool = False,
        rna_cov: bool = False,
        mutation_cov: bool = False,
    ) -> None:
        """Instantiate a SpecletSix model.

        Args:
            name (str): A unique identifier for this instance of SpecletSix. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
            cell_line_cna_cov (bool, optional): Include the cell line copy number
              covariate? Defaults to False.
            gene_cna_cov (bool, optional): Include the gene-specific copy number
              covariate? Defaults to False.
            rna_cov (bool, optional): Include the RNA expression covariate? Defaults to
              False.
            mutation_cov (bool, optional): Include the mutation covariate? Defaults to
              False.
        """
        logger.debug("Instantiating a SpecletSix model.")
        if data_manager is None:
            logger.debug("Creating a data manager since none was supplied.")
            data_manager = CrcDataManager(debug=debug)
        super().__init__(
            name="speclet-six_" + name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )
        self._cell_line_cna_cov = cell_line_cna_cov
        self._gene_cna_cov = gene_cna_cov
        self._rna_cov = rna_cov
        self._mutation_cov = mutation_cov

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletSix model.

        Returns:
            Tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        idx = achelp.common_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = ts(idx.sgrna_idx)
        gene_idx_shared = ts(idx.gene_idx)
        sgrna_to_gene_idx_shared = ts(idx.sgrna_to_gene_idx)
        cellline_idx_shared = ts(idx.cellline_idx)
        # lineage_idx_shared = ts(idx.lineage_idx)
        cellline_to_lineage_idx_shared = ts(idx.cellline_to_lineage_idx)
        batch_idx_shared = ts(idx.batch_idx)
        lfc_shared = ts(data.lfc.values)

        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            # "lineage_idx_shared": lineage_idx_shared,
            "cellline_to_lineage_idx_shared": cellline_to_lineage_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "lfc_shared": lfc_shared,
        }

        logger.info("Creating PyMC3 model.")
        logger.warning(
            "Still need to implement varying effect for source on batch effect."
        )

        with pm.Model() as model:
            # Varying batch intercept.
            μ_j = pm.Normal("μ_j", 0, 0.2)
            σ_j = pm.HalfNormal("σ_j", 1)
            j_offset = pm.Normal("j_offset", 0, 1, shape=idx.n_celllines)
            j = pm.Deterministic("j", μ_j + j_offset * σ_j)

            # Varying gene and cell line intercept.
            μ_h = pm.Normal("μ_h", 0, 0.2)
            σ_h = pm.HalfNormal("σ_h", 1)
            h_offset = pm.Normal("h_offset", 0, 1, shape=(idx.n_genes, idx.n_celllines))
            h = pm.Deterministic("h", μ_h + h_offset * σ_h)

            # Varying cell line intercept.
            if idx.n_lineages == 1:
                logger.info("Only 1 cell line lineage found.")
                μ_d = pm.Normal("μ_d", 0, 0.2)
                σ_d = pm.HalfNormal("σ_d", 1)
                d_offset = pm.Normal("d_offset", 0, 1, shape=idx.n_celllines)
                d = pm.Deterministic("d", μ_d + d_offset * σ_d)
            else:
                logger.info(f"Found {idx.n_lineages} cell line lineages.")
                μ_μ_d = pm.Normal("μ_μ_d", 0, 0.2)
                σ_μ_d = pm.HalfNormal("σ_μ_d", 1)
                μ_d_offset = pm.Normal("μ_d_offset", 0, 1, shape=idx.n_lineages)
                μ_d = pm.Deterministic("μ_d", μ_μ_d + μ_d_offset * σ_μ_d)
                σ_σ_d = pm.HalfNormal("σ_σ_d", 1)
                σ_d = pm.HalfNormal("σ_d", σ_σ_d, shape=idx.n_lineages)
                d_offset = pm.Normal("d_offset", 0, 1, shape=idx.n_celllines)
                d = pm.Deterministic(
                    "d",
                    μ_d[cellline_to_lineage_idx_shared]
                    + d_offset * σ_d[cellline_to_lineage_idx_shared],
                )

            # Varying gene intercept.
            μ_μ_a = pm.Normal("μ_μ_a", 0, 1)
            σ_μ_a = pm.HalfNormal("σ_μ_a", 1)
            μ_a_offset = pm.Normal("μ_a_offset", 0, 1, shape=idx.n_sgrnas)
            μ_a = pm.Deterministic("μ_a", μ_μ_a + μ_a_offset * σ_μ_a)
            σ_σ_a = pm.HalfNormal("σ_σ_a", 1)
            σ_a = pm.HalfNormal("σ_a", σ_σ_a, shape=idx.n_genes)
            a_offset = pm.Normal("a_offset", 0, 1, shape=idx.n_sgrnas)
            a = pm.Deterministic(
                "a",
                μ_a[sgrna_to_gene_idx_shared]
                + a_offset * σ_a[sgrna_to_gene_idx_shared],
            )

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
            σ = pm.HalfNormal("σ", σ_σ, shape=idx.n_batches)

            lfc = pm.Normal(  # noqa: F841
                "lfc",
                μ,
                σ[batch_idx_shared],
                observed=lfc_shared,
                total_size=total_size,
            )

        logger.debug("Finished building model.")

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

        gene_idx_batch = pm.Minibatch(idx.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(idx.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(idx.batch_idx, batch_size=batch_size)
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
