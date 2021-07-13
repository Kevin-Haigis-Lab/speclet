"""Speclet Model Seven."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel
from theano import shared as ts
from theano import tensor
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.exceptions import ShapeError
from src.loggers import logger
from src.managers.model_data_managers import CrcDataManager, DataManager
from src.modeling import feature_engineering as feng
from src.models.speclet_model import ReplacementsDict, SpecletModel

# from src.project_enums import ModelParameterization as MP


def _assert_shapes(
    a: Union[int, tuple[int, ...]], b: Union[int, tuple[int, ...]]
) -> None:
    if not a == b:
        raise ShapeError(a, b)


class SpecletSevenConfiguration(BaseModel):
    """Parameterizations for each covariate in SpecletSeven model."""

    cell_line_cna_cov: bool = False
    gene_cna_cov: bool = False
    rna_cov: bool = False
    mutation_cov: bool = False
    batch_cov: bool = False


class SpecletSeven(SpecletModel):
    """SpecletSeven Model.

    $$
    \\begin{aligned}
    lfc &\\sim N(\\mu, \\sigma) \\\\
    \\mu &= a_{s,c} \\quad \\sigma \\sim HN(1) \\\\
    a_{s,c} &\\sim N(\\mu_a, \\sigma_a)_{s,c} \\\\
    \\mu_a &\\sim N(\\mu_{\\mu_a}, \\sigma_{\\mu_a})_{g,c}
      \\quad \\sigma_a \\sim HN(\\sigma_{\\sigma_a})_{s}
      \\quad \\sigma_{\\sigma_a} \\sim HN(1) \\\\
    \\mu_{\\mu_a} &\\sim N(\\mu_{\\mu_{\\mu_a}}, \\sigma_{\\mu_{\\mu_a}})_{g,l}
      \\quad \\sigma_{\\mu_a} \\sim HN(\\sigma_{\\sigma_{\\mu_a}})_{c}
      \\quad \\sigma_{\\sigma_{\\mu_a}} \\sim HN(1)_{c} \\\\
    \\mu_{\\mu_{\\mu_a}} &\\sim N(0, 1)
      \\quad \\sigma_{\\mu_{\\mu_a}} \\sim HN(1)
    \\end{aligned}
    $$

    where:

    - s: sgRNA
    - g: gene
    - c: cell line
    - l: cell line lineage

    A very deep hierarchical model that is, in part, meant to be a proof-of-concept for
    constructing, fitting, and interpreting such a "tall" hierarchical model.
    """

    config: SpecletSevenConfiguration

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        data_manager: Optional[DataManager] = None,
        config: Optional[SpecletSevenConfiguration] = None,
    ) -> None:
        """Instantiate a SpecletSeven model.

        Args:
            name (str): A unique identifier for this instance of SpecletSeven. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[DataManager], optional): Object that will manage the
              data. If None (default), a `CrcDataManager` is created automatically.
            config (SpecletSevenConfiguration, optional): Model configuration.
        """
        logger.debug("Instantiating a SpecletSeven model.")
        if data_manager is None:
            logger.debug("Creating a data manager since none was supplied.")
            data_manager = CrcDataManager(debug=debug)

        data_manager.add_transformations(
            [
                feng.centered_copynumber_by_cellline,
                feng.centered_copynumber_by_gene,
                feng.zscale_rna_expression_by_gene_and_lineage,
                feng.convert_is_mutated_to_numeric,
            ]
        )

        self.config = config if config is not None else SpecletSevenConfiguration()

        super().__init__(
            name=name,
            root_cache_dir=root_cache_dir,
            debug=debug,
            data_manager=data_manager,
        )

    def set_config(self, info: dict[Any, Any]) -> None:
        """Set model-specific configuration."""
        new_config = SpecletSevenConfiguration(**info)
        if self.config is not None and self.config != new_config:
            logger.info("Setting model-specific configuration.")
            self.config = new_config
            self.model = None

    def _add_cell_line_copy_number_covariate(
        self,
        model: pm.Model,
        co_idx: achelp.CommonIndices,
        cellline_to_lineage_idx_shared: TTShared,
    ) -> tuple[int, int]:
        multiple_lineages = co_idx.n_lineages > 1
        k_shape = (1, co_idx.n_celllines)
        with model:
            if multiple_lineages:
                μ_μ_k = pm.Normal("μ_μ_k", 0, 1)
                σ_μ_k = pm.HalfNormal("σ_μ_k", 1)
                μ_k = pm.Normal("μ_k", μ_μ_k, σ_μ_k, shape=co_idx.n_lineages)
            else:
                μ_k = pm.Normal("μ_k", 0, 1)
            σ_σ_k = pm.HalfNormal("σ_σ_k", 1)
            σ_k = pm.HalfNormal("σ_k", σ_σ_k, shape=co_idx.n_lineages)
            k = pm.Normal(  # noqa: F841
                "k",
                μ_k[cellline_to_lineage_idx_shared],
                σ_k[cellline_to_lineage_idx_shared],
                shape=k_shape,
            )
        return k_shape

    def _add_gene_copy_number_covariate(
        self, model: pm.Model, co_idx: achelp.CommonIndices
    ) -> tuple[int, int]:
        n_shape = (co_idx.n_genes, 1)
        with model:
            μ_n = pm.Normal("μ_n", 0, 1)
            σ_n = pm.HalfNormal("σ_n", 1)
            n = pm.Normal("n", μ_n, σ_n, shape=n_shape)  # noqa: F841
        return n_shape

    def _assert_shape_using_categories_in_df(
        self, ary: np.ndarray, df: pd.DataFrame, cols: tuple[str, ...]
    ) -> None:
        expected_shape = tuple(dphelp.nunique(df[c].values) for c in cols)
        if not ary.shape == expected_shape:
            raise ShapeError(expected_shape, ary.shape)

    def _get_gene_cellline_rna_expression_matrix(self, df: pd.DataFrame) -> np.ndarray:
        rna_df = (
            df[["hugo_symbol", "depmap_id", "lineage", "rna_expr"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .pipe(feng.zscale_rna_expression_by_gene_and_lineage)
        )
        rna_ary = dphelp.dataframe_to_matrix(
            rna_df, rows="hugo_symbol", cols="depmap_id", values="rna_expr_gene_lineage"
        )
        self._assert_shape_using_categories_in_df(
            rna_ary, df, ("hugo_symbol", "depmap_id")
        )
        return rna_ary

    def _get_gene_scaled_copynumber_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cn_df = (
            df[["hugo_symbol", "depmap_id", "copy_number"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .pipe(feng.centered_copynumber_by_gene)
        )
        cn_ary = dphelp.dataframe_to_matrix(
            cn_df, rows="hugo_symbol", cols="depmap_id", values="copy_number_gene"
        )
        self._assert_shape_using_categories_in_df(
            cn_ary, df, ("hugo_symbol", "depmap_id")
        )
        return cn_ary

    def _get_cellline_scaled_copynumber_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cn_df = (
            df[["hugo_symbol", "depmap_id", "copy_number"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .pipe(feng.centered_copynumber_by_cellline)
        )
        cn_ary = dphelp.dataframe_to_matrix(
            cn_df, rows="hugo_symbol", cols="depmap_id", values="copy_number_cellline"
        )
        self._assert_shape_using_categories_in_df(
            cn_ary, df, ("hugo_symbol", "depmap_id")
        )
        return cn_ary

    def _add_gene_expression_covariate(
        self, model: pm.Model, co_idx: achelp.CommonIndices
    ) -> tuple[int, int]:
        q_shape = (co_idx.n_genes, co_idx.n_lineages)
        with model:
            μ_q = pm.Normal("μ_q", 0, 5)
            σ_q = pm.HalfNormal("σ_q", 5)
            q = pm.Normal("q", μ_q, σ_q, shape=q_shape)  # noqa: F841
        return q_shape

    def _add_batch_covariate(
        self,
        model: pm.Model,
        b_idx: achelp.DataBatchIndices,
    ) -> None:
        with model:
            μ_j = pm.Normal("μ_j", 0, 0.5)
            σ_j = pm.HalfNormal("σ_j", 1)
            j = pm.Normal("j", μ_j, σ_j, shape=b_idx.n_batches)  # noqa: F841
        return None

    def _add_varying_gene_cell_line_intercept_covariate(
        self,
        model: pm.Model,
        co_idx: achelp.CommonIndices,
        cellline_to_lineage_idx_shared: TTShared,
    ) -> tuple[int, int]:
        mu_h_shape = (co_idx.n_genes, co_idx.n_lineages)
        h_shape = (co_idx.n_genes, co_idx.n_celllines)
        with model:
            μ_μ_h = pm.Normal("μ_μ_h", 0, 2)
            σ_μ_h = pm.HalfNormal("σ_μ_h", 1)
            μ_h = pm.Normal("μ_h", μ_μ_h, σ_μ_h, shape=mu_h_shape)
            σ_σ_h = pm.HalfNormal("σ_σ_h", 1)
            σ_h = pm.HalfNormal("σ_h", σ_σ_h, shape=co_idx.n_celllines)
            h = pm.Normal(  # noqa: F841
                "h",
                μ_h[:, cellline_to_lineage_idx_shared],
                tensor.ones(shape=h_shape) * σ_h,
                shape=h_shape,
            )
        return h_shape

    def model_specification(self) -> tuple[pm.Model, str]:
        """Build SpecletSeven model.

        Returns:
            tuple[pm.Model, str]: The model and name of the observed variable.
        """
        logger.info("Beginning PyMC3 model specification.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        # Shared Theano variables
        logger.info("Getting Theano shared variables.")
        sgrna_idx_shared = ts(co_idx.sgrna_idx)
        sgrna_to_gene_idx_shared = ts(co_idx.sgrna_to_gene_idx)
        cellline_idx_shared = ts(co_idx.cellline_idx)
        cellline_to_lineage_idx_shared = ts(co_idx.cellline_to_lineage_idx)
        lfc_shared = ts(data.lfc.values)

        self.shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "cellline_to_lineage_idx_shared": cellline_to_lineage_idx_shared,
            "lfc_shared": lfc_shared,
        }

        logger.info("Creating PyMC3 model for SpecletSeven.")

        _a_shape = (co_idx.n_sgrnas, co_idx.n_celllines)

        multiple_lineages = co_idx.n_lineages > 1
        if multiple_lineages:
            logger.info("Multiple cell line lineages in data.")
        else:
            logger.info("Only a single cell line lineage in the data.")

        model = pm.Model()

        # Introduce covariate `h`.
        h_shape = self._add_varying_gene_cell_line_intercept_covariate(
            model,
            co_idx=co_idx,
            cellline_to_lineage_idx_shared=cellline_to_lineage_idx_shared,
        )

        # Create intermediate for `μ_a` and start with `h`.
        with model:
            _μ_a = model["h"]

        # If config, introduce covariate `k` and multiply against cell line-scaled CN.
        if self.config.cell_line_cna_cov:
            k_shape = self._add_cell_line_copy_number_covariate(
                model,
                co_idx=co_idx,
                cellline_to_lineage_idx_shared=cellline_to_lineage_idx_shared,
            )
            cellline_cna_matrix = self._get_cellline_scaled_copynumber_matrix(data)
            _assert_shapes(h_shape[1], k_shape[1])
            _assert_shapes(h_shape, cellline_cna_matrix.shape)
            cellline_cna_shared = ts(cellline_cna_matrix)
            self.shared_vars["cellline_cna_shared"] = cellline_cna_shared
            # Add to the intermediate for `μ_a`.
            with model:
                _μ_a += model["k"] * cellline_cna_shared

        # If config, introduce covariate `n` and multiply against gene-scaled CN.
        if self.config.gene_cna_cov:
            n_shape = self._add_gene_copy_number_covariate(model, co_idx=co_idx)
            gene_cna_matrix = self._get_gene_scaled_copynumber_matrix(data)
            _assert_shapes(n_shape[0], gene_cna_matrix.shape[0])
            _assert_shapes(h_shape, gene_cna_matrix.shape)
            gene_cna_shared = ts(gene_cna_matrix)
            self.shared_vars["gene_cna_shared"] = gene_cna_shared
            # Add to the intermediate for `μ_a`.
            with model:
                _μ_a += model["n"] * gene_cna_shared

        # If config, introduce covariate `q` and multiply against gene- and
        # lineage-scaled RNA.
        if self.config.rna_cov:
            q_shape = self._add_gene_expression_covariate(model, co_idx=co_idx)
            rna_expr_matrix = self._get_gene_cellline_rna_expression_matrix(data)
            _assert_shapes(q_shape[0], rna_expr_matrix.shape[0])
            _assert_shapes(h_shape, rna_expr_matrix.shape)
            rna_expr_shared = ts(rna_expr_matrix)
            self.shared_vars["rna_expr_shared"] = rna_expr_shared
            # Add to the intermediate for `μ_a`.
            with model:
                _μ_a += model["q"][:, cellline_to_lineage_idx_shared] * rna_expr_shared

        ########################################
        # NOTE: Add other `μ_a` covariates here!
        # >
        ########################################

        # With `μ_a` complete, finalize covariate `a`.
        # Create intermediate for `μ` and start it with covariate `a`
        with model:
            μ_a = pm.Deterministic("μ_a", _μ_a)
            σ_σ_a = pm.HalfNormal("σ_σ_a", 1)
            σ_a = pm.HalfNormal("σ_a", σ_σ_a, shape=(co_idx.n_sgrnas, 1))
            a = pm.Normal("a", μ_a[sgrna_to_gene_idx_shared, :], σ_a, shape=_a_shape)
            _μ = a[sgrna_idx_shared, cellline_idx_shared]

        # If config, introduce covariate `j` and add to the intermediate for `μ`.
        if self.config.batch_cov:
            batch_idx_shared = ts(b_idx.batch_idx)
            self.shared_vars["batch_idx_shared"] = batch_idx_shared
            self._add_batch_covariate(model, b_idx=b_idx)
            with model:
                _μ += model["j"][batch_idx_shared]

        # With `μ` complete, finalize the model.
        with model:
            μ = pm.Deterministic("μ", _μ)
            σ = pm.HalfNormal("σ", 1)
            lfc = pm.Normal(  # noqa: F841
                "lfc", μ, σ, observed=lfc_shared, total_size=total_size
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
        mb_size = self.data_manager.get_batch_size()
        co_idx = achelp.common_indices(data)
        b_idx = achelp.data_batch_indices(data)

        sgrna_idx_batch = pm.Minibatch(co_idx.sgrna_idx, batch_size=mb_size)
        cellline_idx_batch = pm.Minibatch(co_idx.cellline_idx, batch_size=mb_size)
        lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=mb_size)

        replacement_params: ReplacementsDict = {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
        }

        if self.config.batch_cov:
            batch_idx_batch = pm.Minibatch(b_idx.batch_idx, batch_size=mb_size)
            replacement_params[self.shared_vars["batch_idx_shared"]] = batch_idx_batch

        return replacement_params
