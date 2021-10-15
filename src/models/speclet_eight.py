"""SpecletEight model."""


from pathlib import Path
from typing import Final, Optional, Union

import janitor  # noqa: F401
import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel
from pymc3.model import FreeRV as pmFreeRV
from theano import shared as ts
from theano import tensor as tt

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import (
    CrisprScreenDataManager,
    Data,
    common_crispr_screen_transformations,
)
from src.modeling import feature_engineering as feng
from src.models.speclet_model import (
    ObservedVarName,
    SpecletModel,
    SpecletModelDataManager,
)


class SpecletEightConfiguration(BaseModel):
    """Configuration for SpecletEight."""

    broad_only: bool = True


def _append_total_read_counts(df: Data) -> Data:
    return achelp.append_total_read_counts(df)


def _add_useful_read_count_columns(df: Data) -> Data:
    return achelp.add_useful_read_count_columns(df)


def _thin_data_columns(df: Data) -> Data:
    keep_cols: Final[list[str]] = [
        "sgrna",
        "hugo_symbol",
        "depmap_id",
        "lineage",
        "counts_initial_adj",
        "counts_final",
        "rna_expr_gene_lineage",
        "p_dna_batch",
        "screen",
    ]
    return df[keep_cols]


class SpecletEight(SpecletModel):
    """## SpecletEight.

    A negative binomial model of the read counts from the CRISPR screen data.
    """

    _config: SpecletEightConfiguration

    def __init__(
        self,
        name: str,
        data_manager: Optional[SpecletModelDataManager] = None,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        config: Optional[SpecletEightConfiguration] = None,
    ) -> None:
        """Instantiate a SpecletEight model.

        Args:
            name (str): A unique identifier for this instance of SpecletEight. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (SpecletEightConfiguration, optional): Model configuration.
        """
        logger.info("Creating a new SpecletEight object.")
        self._config = SpecletEightConfiguration() if config is None else config

        if data_manager is None:
            data_manager = CrisprScreenDataManager(
                data_source=DataFile.DEPMAP_CRC_BONE_SUBSAMPLE
            )

        data_transformations = common_crispr_screen_transformations.copy()
        data_transformations += [
            feng.zscale_rna_expression_by_gene_and_lineage,
            _append_total_read_counts,
            _add_useful_read_count_columns,
        ]
        if self._config.broad_only:
            data_transformations.append(achelp.filter_for_broad_source_only)

        data_manager.add_transformation(data_transformations)
        data_manager.add_transformation(_thin_data_columns)
        # Need to add at end because `p_dna_batch` becomes non-categorical.
        data_manager.add_transformation(achelp.set_achilles_categorical_columns)

        super().__init__(name, data_manager, root_cache_dir=root_cache_dir, debug=debug)

    def model_specification(self) -> tuple[pm.Model, ObservedVarName]:
        """Define the PyMC3 model.

        Returns:
            pm.Model: The PyMC3 model.
            ObservedVarName: Name of the target variable in the model.
        """
        logger.info("Creating SpecletEight model.")
        data = self.data_manager.get_data()

        total_size = data.shape[0]
        logger.info(f"Number of data points: {total_size}")
        co_idx = achelp.common_indices(data)
        logger.info(f"Number of sgRNA: {co_idx.n_sgrnas}")
        logger.info(f"Number of genes: {co_idx.n_genes}")
        logger.info(f"Number of cell lines: {co_idx.n_celllines}")
        logger.info(f"Number of lineages: {co_idx.n_lineages}")
        # b_idx = achelp.data_batch_indices(data)

        logger.info("Creating shared variables.")
        x_initial = ts(data.counts_initial_adj.values)

        coords = {
            "sgrna": data.sgrna.cat.categories,
            "gene": data.hugo_symbol.cat.categories,
            "cell_line": data.depmap_id.cat.categories,
            "lineage": data.lineage.cat.categories,
            "batch": data.p_dna_batch.cat.categories,
        }

        rna_expr_matrix = _rna_expr_gene_lineage_matrix(data, "rna_expr_gene_lineage")
        _sgrna_by_cell_line = (co_idx.n_sgrnas, co_idx.n_celllines)

        with pm.Model(coords=coords) as model:
            s = pm.Data("sgrna_idx", co_idx.sgrna_idx)
            g_s = pm.Data("sgrna_to_gene_idx", co_idx.sgrna_to_gene_idx)
            c = pm.Data("cell_line_idx", co_idx.cellline_idx)
            l_c = pm.Data("cell_line_to_lineage_idx", co_idx.cellline_to_lineage_idx)
            # b = pm.Data("batch_idx", b_idx.batch_idx)
            rna_expr = pm.Data("rna_expression_gl", rna_expr_matrix)
            x_initial = pm.Data("x_initial", data.counts_initial_adj.values)
            counts_final = pm.Data("counts_final", data.counts_final.values)

            h = _gene_by_cell_line_hierarchical_structure(
                "h", co_idx=co_idx, l_c_idx=l_c
            )
            q = _gene_by_cell_line_hierarchical_structure(
                "q", co_idx=co_idx, l_c_idx=l_c
            )

            mu_beta = pm.Deterministic(
                "mu_beta", h + q * rna_expr
            )  # shape: [gene x cell line]

            sigma_sigma_beta = pm.HalfNormal("sigma_sigma_beta", 1)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma_sigma_beta, dims="gene")

            beta_sgrna = pm.Normal(
                "beta_sgrna",
                mu_beta[g_s, :],
                tt.ones(shape=_sgrna_by_cell_line)
                * sigma_beta[g_s].reshape((co_idx.n_sgrnas, 1)),
                dims=("sgrna", "cell_line"),
            )

            eta = pm.Deterministic("eta", beta_sgrna[s, c])
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            alpha = pm.HalfCauchy("alpha", 5)
            y = pm.NegativeBinomial(  # noqa: F841
                "y", x_initial * mu, alpha, observed=counts_final
            )

        return model, "y"


def _gene_by_cell_line_hierarchical_structure(
    name: str, co_idx: achelp.CommonIndices, l_c_idx: Union[np.ndarray, pm.Data]
) -> pmFreeRV:
    _gene_by_cell_line = (co_idx.n_genes, co_idx.n_celllines)

    mu_mu_beta = pm.Normal(f"mu_mu_{name}", 0, 1)
    sigma_mu_beta = pm.HalfNormal(f"sigma_mu_{name}", 1)
    mu_beta = pm.Normal(
        f"mu_{name}", mu_mu_beta, sigma_mu_beta, dims=("gene", "lineage")
    )

    sigma_sigma_beta = pm.HalfNormal(f"sigma_sigma_{name}", 1)
    sigma_beta = pm.HalfNormal(f"sigma_{name}", sigma_sigma_beta, dims="lineage")

    beta = pm.Normal(
        name,
        mu_beta[:, l_c_idx],
        tt.ones(shape=_gene_by_cell_line) * sigma_beta[l_c_idx],
        dims=("gene", "cell_line"),
    )
    return beta


def _rna_expr_gene_lineage_matrix(df: pd.DataFrame, rna_col: str) -> np.ndarray:
    """Pivot a long data frame into a matrix of RNA expression data.

    Output is a matrix of shape (gene x cell line).

    """
    pivot_df = (
        df.copy()[["hugo_symbol", "depmap_id", rna_col]]
        .drop_duplicates()
        .sort_values("depmap_id")
        .pivot_wider(index="hugo_symbol", names_from="depmap_id", values_from=rna_col)
        .sort_values("hugo_symbol", ascending=True)
    )

    assert all(pivot_df.hugo_symbol.values == df.hugo_symbol.cat.categories)
    assert all(pivot_df.columns[1:] == df.depmap_id.cat.categories)
    return pivot_df.drop(columns="hugo_symbol").values
