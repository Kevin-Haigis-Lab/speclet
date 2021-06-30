"""Speclet Model Six."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pymc3 as pm
from theano import shared as ts

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.models.speclet_model import ReplacementsDict, SpecletModel


def centered_copynumber_by_cellline(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column of centered copy number values by cell line.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with a new column `"copy_number_cellline"`.
    """
    logger.info("Adding 'copy_number_cellline' column.")
    return dphelp.center_column_grouped_dataframe(
        df,
        grp_col="depmap_id",
        val_col="copy_number",
        new_col_name="copy_number_cellline",
    )


def centered_copynumber_by_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column of centered copy number values by gene.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with a new column `"copy_number_gene"`.
    """
    logger.info("Adding 'copy_number_gene' column.")
    return dphelp.center_column_grouped_dataframe(
        df,
        grp_col="hugo_symbol",
        val_col="copy_number",
        new_col_name="copy_number_gene",
    )


def zscale_rna_expression_by_gene_and_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """Z-scale the RNA expression per gene in each lineage.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with a new column `"rna_expr_gene_lineage"`.
    """
    logger.info("Adding 'rna_expr_gene_lineage' column.")
    return achelp.zscale_rna_expression_by_gene_lineage(
        df,
        rna_col="rna_expr",
        new_col="rna_expr_gene_lineage",
        lower_bound=-5.0,
        upper_bound=5.0,
    )


def convert_is_mutated_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the boolean column for gene mutation status to type integer.

    Args:
        df (pd.DataFrame): Achilles data frame.

    Returns:
        pd.DataFrame: Same data frame with the column `"is_mutated"` as type integer.
    """
    logger.info("Converting 'is_mutated' column to 'int'.")
    df["is_mutated"] = df["is_mutated"].astype(int)
    return df


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

    Attributes:
        cell_line_cna_cov (bool): Include the covariate for copy number effect per cell
          line \\(k_c\\).
        gene_cna_cov (bool): Include the covariate for copy number effect per gene
          \\(n_g\\).
        rna_cov (bool): Include the covariate for RNA expression per gene/lineage
          \\(q_{g,l}\\).
        mutation_cov (bool): Include the covariate for mutation effect per
          gene/lineage \\(m_{g,l}\\).
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

        data_manager.add_transformations(
            [
                centered_copynumber_by_cellline,
                centered_copynumber_by_gene,
                zscale_rna_expression_by_gene_and_lineage,
                convert_is_mutated_to_numeric,
            ]
        )

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

    @property
    def cell_line_cna_cov(self) -> bool:
        """Should the covariate for cell line-specific CN be included?

        Returns:
            bool: Whether the covariate is included or not.
        """
        return self._cell_line_cna_cov

    @cell_line_cna_cov.setter
    def cell_line_cna_cov(self, new_value: bool) -> None:
        """Decide if the cell line-specific CN covariate should be included.

        Args:
            new_value (bool): Whether the covariate is included or not.
        """
        if self._cell_line_cna_cov != new_value:
            logger.info("Changing `cell_line_cna_cov` to `{new_value}`.")
            self._cell_line_cna_cov = new_value
            self._reset_model_and_results()

    @property
    def gene_cna_cov(self) -> bool:
        """Should the covariate for gene-specific CN be included?

        Returns:
            bool: Whether the covariate is included or not.
        """
        return self._gene_cna_cov

    @gene_cna_cov.setter
    def gene_cna_cov(self, new_value: bool) -> None:
        """Decide if the gene-specific CN covariate should be included.

        Args:
            new_value (bool): Whether the covariate is included or not.
        """
        if self._gene_cna_cov != new_value:
            logger.info("Changing `gene_cna_cov` to `{new_value}`.")
            self._gene_cna_cov = new_value
            self._reset_model_and_results()

    @property
    def rna_cov(self) -> bool:
        """Should the covariate for gene- and lineage-specific RNA be included?

        Returns:
            bool: Whether the covariate is included or not.
        """
        return self._rna_cov

    @rna_cov.setter
    def rna_cov(self, new_value: bool) -> None:
        """Decide if the gene- and lineage-specific RNA covariate should be included.

        Args:
            new_value (bool): Whether the covariate is included or not.
        """
        if self._rna_cov != new_value:
            logger.info("Changing `rna_cov` to `{new_value}`.")
            self._rna_cov = new_value
            self._reset_model_and_results()

    @property
    def mutation_cov(self) -> bool:
        """Should the covariate for gene- and lineage-specific mutation be included?

        Returns:
            bool: Whether the covariate is included or not.
        """
        return self._mutation_cov

    @mutation_cov.setter
    def mutation_cov(self, new_value: bool) -> None:
        """Decide if the gene- and lineage-specific mutation covariate should be included.

        Args:
            new_value (bool): Whether the covariate is included or not.
        """
        if self._mutation_cov != new_value:
            logger.info("Changing `mutation_cov` to `{new_value}`.")
            self._mutation_cov = new_value
            self._reset_model_and_results()

    def model_specification(self) -> Tuple[pm.Model, str]:
        """Build SpecletSix model.

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

        if self.cell_line_cna_cov:
            cellline_cna_shared = ts(data["copy_number_cellline"].values)
            self.shared_vars["cellline_cna_shared"] = cellline_cna_shared

        if self.gene_cna_cov:
            gene_cna_shared = ts(data["copy_number_gene"].values)
            self.shared_vars["gene_cna_shared"] = gene_cna_shared

        if self.rna_cov:
            rna_expr_shared = ts(data["rna_expr_gene_lineage"].values)
            self.shared_vars["rna_expr_shared"] = rna_expr_shared

        if self.mutation_cov:
            mutation_shared = ts(data["is_mutated"].values)
            self.shared_vars["mutation_shared"] = mutation_shared

        logger.info("Creating PyMC3 SpecletSix model.")

        with pm.Model() as model:
            # Varying batch intercept.
            if b_idx.n_screens == 1:
                μ_j = pm.Normal("μ_j", 0, 1)
                σ_j = pm.HalfNormal("σ_j", 1)
                j_offset = pm.Normal("j_offset", 0, 0.5, shape=b_idx.n_batches)
                j = pm.Deterministic("j", μ_j + j_offset * σ_j)
            else:
                μ_μ_j = pm.Normal("μ_μ_j", 0, 0.5)
                σ_μ_j = pm.HalfNormal("σ_μ_j", 1)
                σ_σ_j = pm.HalfNormal("σ_σ_j", 1)
                μ_j_offset = pm.Normal("μ_j_offset", 0, 0.5, shape=b_idx.n_screens)
                μ_j = pm.Deterministic("μ_j", μ_μ_j + μ_j_offset * σ_μ_j)
                σ_j = pm.HalfNormal("σ_j", σ_σ_j, shape=b_idx.n_screens)
                j_offset = pm.Normal("j_offset", 0, 0.5, shape=b_idx.n_batches)
                j = pm.Deterministic(
                    "j",
                    μ_j[batch_to_screen_idx_shared]
                    + j_offset * σ_j[batch_to_screen_idx_shared],
                )

            # Varying gene and cell line intercept.
            μ_h = pm.Normal("μ_h", 0, 0.5)
            σ_h = pm.HalfNormal("σ_h", 1)
            h_offset = pm.Normal(
                "h_offset", 0, 0.5, shape=(co_idx.n_genes, co_idx.n_celllines)
            )
            h = pm.Deterministic("h", μ_h + h_offset * σ_h)

            # Varying cell line intercept.
            if co_idx.n_lineages == 1:
                logger.info("Only 1 cell line lineage found.")
                μ_d = pm.Normal("μ_d", 0, 0.5)
                σ_d = pm.HalfNormal("σ_d", 2)
                d_offset = pm.Normal("d_offset", 0, 0.5, shape=co_idx.n_celllines)
                d = pm.Deterministic("d", μ_d + d_offset * σ_d)
            else:
                logger.info(f"Found {co_idx.n_lineages} cell line lineages.")
                μ_μ_d = pm.Normal("μ_μ_d", 0, 0.5)
                σ_μ_d = pm.HalfNormal("σ_μ_d", 1)
                μ_d_offset = pm.Normal("μ_d_offset", 0, 0.5, shape=co_idx.n_lineages)
                μ_d = pm.Deterministic("μ_d", μ_μ_d + μ_d_offset * σ_μ_d)
                σ_σ_d = pm.HalfNormal("σ_σ_d", 1)
                σ_d = pm.HalfNormal("σ_d", σ_σ_d, shape=co_idx.n_lineages)
                d_offset = pm.Normal("d_offset", 0, 0.5, shape=co_idx.n_celllines)
                d = pm.Deterministic(
                    "d",
                    μ_d[cellline_to_lineage_idx_shared]
                    + d_offset * σ_d[cellline_to_lineage_idx_shared],
                )

            # Varying gene intercept.
            μ_μ_a = pm.Normal("μ_μ_a", 0, 0.5)
            σ_μ_a = pm.HalfNormal("σ_μ_a", 1)
            μ_a_offset = pm.Normal("μ_a_offset", 0, 0.5, shape=co_idx.n_genes)
            μ_a = pm.Deterministic("μ_a", μ_μ_a + μ_a_offset * σ_μ_a)
            σ_σ_a = pm.HalfNormal("σ_σ_a", 1)
            σ_a = pm.HalfNormal("σ_a", σ_σ_a, shape=co_idx.n_genes)
            a_offset = pm.Normal("a_offset", 0, 0.5, shape=co_idx.n_sgrnas)
            a = pm.Deterministic(
                "a",
                μ_a[sgrna_to_gene_idx_shared]
                + a_offset * σ_a[sgrna_to_gene_idx_shared],
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

            if self.cell_line_cna_cov:
                μ_k = pm.Normal("μ_k", -0.5, 2)
                σ_k = pm.HalfNormal("σ_k", 1)
                k_offset = pm.Normal("k_offset", 0, 1, shape=co_idx.n_celllines)
                k = pm.Deterministic("k", (μ_k + k_offset * σ_k))
                _μ += k[cellline_idx_shared] * cellline_cna_shared

            if self.gene_cna_cov:
                μ_q = pm.Normal("μ_n", 0, 2)
                σ_q = pm.HalfNormal("σ_n", 1)
                n_offset = pm.Normal("n_offset", 0, 1, shape=co_idx.n_genes)
                n = pm.Deterministic("n", (μ_q + n_offset * σ_q))
                _μ += n[gene_idx_shared] * gene_cna_shared

            if self.rna_cov:
                μ_q = pm.Normal("μ_q", 0, 2)
                σ_q = pm.HalfNormal("σ_q", 1)
                q_offset = pm.Normal(
                    "q_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_lineages)
                )
                q = pm.Deterministic("q", (μ_q + q_offset * σ_q))
                _μ += q[gene_idx_shared, lineage_idx_shared] * rna_expr_shared

            if self.mutation_cov:
                μ_m = pm.Normal("μ_m", 0, 2)
                σ_m = pm.HalfNormal("σ_m", 1)
                m_offset = pm.Normal(
                    "m_offset", 0, 1, shape=(co_idx.n_genes, co_idx.n_lineages)
                )
                m = pm.Deterministic("m", (μ_m + m_offset * σ_m))
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
        mb_size = self.data_manager.get_batch_size()
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
