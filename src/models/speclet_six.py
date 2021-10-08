"""Speclet Model Six."""

from pathlib import Path
from typing import Any, Optional

import pymc3 as pm
from pydantic import BaseModel
from theano import shared as ts

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import CrisprScreenDataManager, DataFrameTransformation
from src.modeling import feature_engineering as feng
from src.models.speclet_model import (
    ObservedVarName,
    ReplacementsDict,
    SpecletModel,
    SpecletModelDataManager,
)
from src.project_enums import ModelParameterization as MP


class SpecletSixConfiguration(BaseModel):
    """Parameterizations for each covariate in SpecletSix model."""

    cell_line_cna_cov: bool = False
    gene_cna_cov: bool = False
    rna_cov: bool = False
    mutation_cov: bool = False
    a: MP = MP.CENTERED
    d: MP = MP.CENTERED
    h: MP = MP.CENTERED
    j: MP = MP.CENTERED
    k: MP = MP.CENTERED
    n: MP = MP.CENTERED
    q: MP = MP.CENTERED
    m: MP = MP.CENTERED


class SpecletSix(SpecletModel):
    """SpecletSix Model.

    $$
    \\begin{aligned}
    lfc &\\sim i + a_s + d_c + h_{g,c} + j_b +
    k_c C^{(c)} + n_g C^{(g)} + q_{g,l} R^{(g,l)} + m_{g,l} M \\\\
    a_s &\\sim N(μ_a, σ_a)[\\text{gene}] \\\\
    d_c &\\sim N(μ_d, σ_d)[\\text{lineage}] \\text{ (if more than one lineage)} \\\\
    j_b &\\sim N(μ_j, σ_j)[\\text{source}] \\text{ (if more than one source)}
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

    - \\(a_s\\): sgRNA effect with hierarchical level for gene (g)
    - \\(d_c\\): cell line effect with hierarchical level for lineage (l; if more than
      one is found)
    - \\(j_b\\): data source (o; if more than one is found)
    - \\(k_c\\): cell line effect of copy number ( \\(C^{(c)}\\): z-scaled per cell
      line)
    - \\(n_g\\): gene effect of copy number ( \\(C^{(g)}\\): z-scaled per gene)
    - \\(q_{g,l}\\): RNA effect varying per gene and cell line lineage
      ( \\(R^{(g,l)}\\): z-scaled within each gene and lineage)
    - \\(m_{g,l}\\): mutation effect varying per gene and cell line lineage
      ( \\(M \\in {0, 1}\\))
    """

    config: SpecletSixConfiguration

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[SpecletModelDataManager] = None,
        config: Optional[SpecletSixConfiguration] = None,
    ) -> None:
        """Instantiate a SpecletSix model.

        Args:
            name (str): A unique identifier for this instance of SpecletSix. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (SpecletSixConfiguration, optional): Model configurations.
        """
        logger.debug("Instantiating a SpecletSix model.")
        if data_manager is None:
            logger.debug("Creating a data manager since none was supplied.")
            data_manager = CrisprScreenDataManager(DataFile.DEPMAP_CRC_SUBSAMPLE)

        transformations: list[DataFrameTransformation] = [
            feng.centered_copynumber_by_cellline,
            feng.centered_copynumber_by_gene,
            feng.zscale_rna_expression_by_gene_and_lineage,
            feng.convert_is_mutated_to_numeric,
        ]
        data_manager.add_transformation(transformations)

        self.config = config if config is not None else SpecletSixConfiguration()

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def set_config(self, info: dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        new_config = SpecletSixConfiguration(**info)
        if self.config is not None and self.config != new_config:
            logger.info("Setting model-specific configuration.")
            self.config = new_config
            self.model = None

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Build SpecletSix model.

        Returns:
            Tuple[pm.Model, ObservedVarName]: The model and name of the observed
            variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = ts(co_idx.sgrna_idx)
        gene_idx_shared = ts(co_idx.gene_idx)
        sgrna_to_gene_idx_shared = ts(co_idx.sgrna_to_gene_idx)
        cellline_idx_shared = ts(co_idx.cellline_idx)
        lineage_idx_shared = ts(co_idx.lineage_idx)
        cellline_to_lineage_idx_shared = ts(co_idx.cellline_to_lineage_idx)
        batch_idx_shared = ts(b_idx.batch_idx)
        batch_to_screen_idx_shared = ts(b_idx.batch_to_screen_idx)
        lfc_shared = ts(data.lfc.values)

        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "lineage_idx_shared": lineage_idx_shared,
            "cellline_to_lineage_idx_shared": cellline_to_lineage_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "batch_to_screen_idx_shared": batch_to_screen_idx_shared,
            "lfc_shared": lfc_shared,
        }

        if self.config.cell_line_cna_cov:
            cellline_cna_shared = ts(data["copy_number_cellline"].values)
            self.shared_vars["cellline_cna_shared"] = cellline_cna_shared

        if self.config.gene_cna_cov:
            gene_cna_shared = ts(data["copy_number_gene"].values)
            self.shared_vars["gene_cna_shared"] = gene_cna_shared

        if self.config.rna_cov:
            rna_expr_shared = ts(data["rna_expr_gene_lineage"].values)
            self.shared_vars["rna_expr_shared"] = rna_expr_shared

        if self.config.mutation_cov:
            mutation_shared = ts(data["is_mutated"].values)
            self.shared_vars["mutation_shared"] = mutation_shared

        logger.info("Creating PyMC3 SpecletSix model.")

        single_screen = b_idx.n_screens == 1
        single_cell_line_lineage = co_idx.n_lineages == 1

        with pm.Model() as model:
            # Varying batch intercept.
            if single_screen:
                μ_j = pm.Normal("μ_j", 0, 1)
                σ_j = pm.HalfNormal("σ_j", 1)
            else:
                μ_μ_j = pm.Normal("μ_μ_j", 0, 0.5)
                σ_μ_j = pm.HalfNormal("σ_μ_j", 1)
                σ_σ_j = pm.HalfNormal("σ_σ_j", 1)
                μ_j = pm.Normal("μ_j", μ_μ_j, σ_μ_j, shape=b_idx.n_screens)
                σ_j = pm.HalfNormal("σ_j", σ_σ_j, shape=b_idx.n_screens)

            if self.config.j is MP.NONCENTERED:
                j_offset = pm.Normal("j_offset", 0, 0.5, shape=b_idx.n_batches)
                if single_screen:
                    j = pm.Deterministic("j", μ_j + j_offset * σ_j)
                else:
                    j = pm.Deterministic(
                        "j",
                        μ_j[batch_to_screen_idx_shared]
                        + j_offset * σ_j[batch_to_screen_idx_shared],
                    )
            else:
                if single_screen:
                    j = pm.Normal("j", μ_j, σ_j, shape=b_idx.n_batches)
                else:
                    j = pm.Normal(
                        "j",
                        μ_j[batch_to_screen_idx_shared],
                        σ_j[batch_to_screen_idx_shared],
                        shape=b_idx.n_batches,
                    )

            # Varying gene and cell line intercept.
            μ_h = pm.Normal("μ_h", 0, 0.5)
            σ_h = pm.HalfNormal("σ_h", 1)
            if self.config.h is MP.NONCENTERED:
                h_offset = pm.Normal(
                    "h_offset", 0, 0.5, shape=(co_idx.n_genes, co_idx.n_celllines)
                )
                h = pm.Deterministic("h", μ_h + h_offset * σ_h)
            else:
                h = pm.Normal("h", μ_h, σ_h, shape=(co_idx.n_genes, co_idx.n_celllines))

            # Varying cell line intercept.
            if single_cell_line_lineage:
                logger.info("Only 1 cell line lineage found.")
                μ_d = pm.Normal("μ_d", 0, 0.5)
                σ_d = pm.HalfNormal("σ_d", 2)
            else:
                logger.info(f"Found {co_idx.n_lineages} cell line lineages.")
                μ_μ_d = pm.Normal("μ_μ_d", 0, 0.5)
                σ_μ_d = pm.HalfNormal("σ_μ_d", 1)
                μ_d_offset = pm.Normal("μ_d_offset", 0, 0.5, shape=co_idx.n_lineages)
                μ_d = pm.Deterministic("μ_d", μ_μ_d + μ_d_offset * σ_μ_d)
                σ_σ_d = pm.HalfNormal("σ_σ_d", 1)
                σ_d = pm.HalfNormal("σ_d", σ_σ_d, shape=co_idx.n_lineages)

            if self.config.d is MP.NONCENTERED:
                d_offset = pm.Normal("d_offset", 0, 0.5, shape=co_idx.n_celllines)
                if single_cell_line_lineage:
                    d = pm.Deterministic("d", μ_d + d_offset * σ_d)
                else:
                    d = pm.Deterministic(
                        "d",
                        μ_d[cellline_to_lineage_idx_shared]
                        + d_offset * σ_d[cellline_to_lineage_idx_shared],
                    )
            else:
                if single_cell_line_lineage:
                    d = pm.Normal("d", μ_d, σ_d, shape=co_idx.n_celllines)
                else:
                    d = pm.Normal(
                        "d",
                        μ_d[cellline_to_lineage_idx_shared],
                        σ_d[cellline_to_lineage_idx_shared],
                        shape=co_idx.n_celllines,
                    )

            # Varying gene intercept.
            μ_μ_a = pm.Normal("μ_μ_a", 0, 0.5)
            σ_μ_a = pm.HalfNormal("σ_μ_a", 1)
            μ_a_offset = pm.Normal("μ_a_offset", 0, 0.5, shape=co_idx.n_genes)
            μ_a = pm.Deterministic("μ_a", μ_μ_a + μ_a_offset * σ_μ_a)
            σ_σ_a = pm.HalfNormal("σ_σ_a", 1)
            σ_a = pm.HalfNormal("σ_a", σ_σ_a, shape=co_idx.n_genes)

            if self.config.a is MP.NONCENTERED:
                a_offset = pm.Normal("a_offset", 0, 0.5, shape=co_idx.n_sgrnas)
                a = pm.Deterministic(
                    "a",
                    μ_a[sgrna_to_gene_idx_shared]
                    + a_offset * σ_a[sgrna_to_gene_idx_shared],
                )
            else:
                a = pm.Normal(
                    "a",
                    μ_a[sgrna_to_gene_idx_shared],
                    σ_a[sgrna_to_gene_idx_shared],
                    shape=co_idx.n_sgrnas,
                )

            # Global intercept.
            i = pm.Normal("i", 0, 1)

            _μ = (
                i
                + a[gene_idx_shared]
                + d[cellline_idx_shared]
                + h[gene_idx_shared, cellline_idx_shared]
                + j[batch_idx_shared]
            )

            if self.config.cell_line_cna_cov:
                μ_k = pm.Normal("μ_k", -0.5, 2)
                σ_k = pm.HalfNormal("σ_k", 1)
                if self.config.k is MP.NONCENTERED:
                    k_offset = pm.Normal("k_offset", 0, 1, shape=co_idx.n_celllines)
                    k = pm.Deterministic("k", (μ_k + k_offset * σ_k))
                else:
                    k = pm.Normal("k", μ_k, σ_k, shape=co_idx.n_celllines)

                _μ += k[cellline_idx_shared] * cellline_cna_shared

            if self.config.gene_cna_cov:
                μ_n = pm.Normal("μ_n", 0, 2)
                σ_n = pm.HalfNormal("σ_n", 1)
                if self.config.n is MP.NONCENTERED:
                    n_offset = pm.Normal("n_offset", 0, 1, shape=co_idx.n_genes)
                    n = pm.Deterministic("n", (μ_n + n_offset * σ_n))
                else:
                    n = pm.Normal("n", μ_n, σ_n, shape=co_idx.n_genes)

                _μ += n[gene_idx_shared] * gene_cna_shared

            if self.config.rna_cov:
                μ_q = pm.Normal("μ_q", 0, 2)
                σ_q = pm.HalfNormal("σ_q", 1)
                if self.config.q is MP.NONCENTERED:
                    q_offset = pm.Normal(
                        "q_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_lineages)
                    )
                    q = pm.Deterministic("q", (μ_q + q_offset * σ_q))
                else:
                    q = pm.Normal(
                        "q", μ_q, σ_q, shape=(co_idx.n_genes, co_idx.n_lineages)
                    )

                _μ += q[gene_idx_shared, lineage_idx_shared] * rna_expr_shared

            if self.config.mutation_cov:
                μ_m = pm.Normal("μ_m", 0, 2)
                σ_m = pm.HalfNormal("σ_m", 1)
                if self.config.m is MP.NONCENTERED:
                    m_offset = pm.Normal(
                        "m_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_lineages)
                    )
                    m = pm.Deterministic("m", (μ_m + m_offset * σ_m))
                else:
                    m = pm.Normal(
                        "m", μ_m, σ_m, shape=(co_idx.n_genes, co_idx.n_lineages)
                    )

                _μ += m[gene_idx_shared, lineage_idx_shared] * mutation_shared

            μ = pm.Deterministic("μ", _μ)

            # Standard deviation of log-fold change, varies per batch.
            σ_σ = pm.HalfNormal("σ_σ", 0.5)
            σ = pm.HalfNormal("σ", σ_σ, shape=b_idx.n_batches)

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
            AttributeError: Raised if there are no shared variables.

        Returns:
            ReplacementsDict: A dictionary mapping new data to shared variables.
        """
        logger.debug("Making dictionary of replacement parameters.")

        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        data = self.data_manager.get_data()
        mb_size = self._get_batch_size()
        co_idx = achelp.common_indices(data)
        batch_idx = achelp.data_batch_indices(data)

        sgrna_idx_batch = pm.Minibatch(co_idx.sgrna_idx, batch_size=mb_size)
        gene_idx_batch = pm.Minibatch(co_idx.gene_idx, batch_size=mb_size)
        cellline_idx_batch = pm.Minibatch(co_idx.cellline_idx, batch_size=mb_size)
        lineage_idx_batch = pm.Minibatch(co_idx.lineage_idx, batch_size=mb_size)
        batch_idx_batch = pm.Minibatch(batch_idx.batch_idx, batch_size=mb_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=mb_size)

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["lineage_idx_shared"]: lineage_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }
