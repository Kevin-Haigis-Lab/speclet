"""SpecletEight model."""


from pathlib import Path
from typing import Final, Optional, Union

import janitor  # noqa: F401
import numpy as np
import pandas as pd
import pymc3 as pm
from pydantic import BaseModel
from pymc3.model import FreeRV as pmFreeRV

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile
from src.loggers import logger
from src.managers.data_managers import (
    CrisprScreenDataManager,
    Data,
    DataFrameTransformation,
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


def _reduce_num_genes_for_dev(df: Data) -> Data:
    logger.warn("Reducing number of genes for development.")
    _genes = ["KRAS", "TP53", "NLRP8", "KLF5"]
    return df[df.hugo_symbol.isin(_genes)]


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


def make_count_model_data_manager(
    data_source: DataFile,
    other_transforms: Optional[list[DataFrameTransformation]] = None,
) -> CrisprScreenDataManager:
    """Make a data manager good for models of count data.

    Default list of data transformation:

    1. z-scale RNA expression by gene and lineage (`src.modeling.feature_engineering`)
    2. append total read counts (`src.data_processing.achilles`)
    3. add useful read count columns (`src.data_processing.achilles`)
    4. (optional additional transformations)
    5. set Achilles categorical columns (`src.data_processing.achilles`)

    Args:
        data_source (DataFile): Data source.
        other_transforms (Optional[list[DataFrameTransformation]], optional): Additional
          transformation to include in the data manager's defaults. Defaults to None.

    Returns:
        CrisprScreenDataManager: Data manager for count modeling.
    """
    data_manager = CrisprScreenDataManager(data_source=data_source)

    data_transformations = common_crispr_screen_transformations.copy()
    data_transformations += [
        feng.zscale_rna_expression_by_gene_and_lineage,
        _append_total_read_counts,
        _add_useful_read_count_columns,
    ]

    if other_transforms is not None:
        data_transformations += other_transforms

    data_manager.add_transformation(data_transformations)
    # Need to add at end because `p_dna_batch` becomes non-categorical.
    data_manager.add_transformation(achelp.set_achilles_categorical_columns)
    return data_manager


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
        config: Optional[SpecletEightConfiguration] = None,
    ) -> None:
        """Instantiate a SpecletEight model.

        Args:
            name (str): A unique identifier for this instance of SpecletEight. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            data_manager (Optional[SpecletModelDataManager], optional): Object that will
              manage the data. If None (default), a `CrisprScreenDataManager` is
              created automatically.
            config (SpecletEightConfiguration, optional): Model configuration.
        """
        logger.info("Creating a new SpecletEight object.")
        self._config = SpecletEightConfiguration() if config is None else config

        if data_manager is None:
            _other_transforms: list[DataFrameTransformation] = []
            _other_transforms.append(_thin_data_columns)
            if self._config.broad_only:
                _other_transforms.append(achelp.filter_for_broad_source_only)

            data_manager = make_count_model_data_manager(
                DataFile.DEPMAP_CRC_BONE_SUBSAMPLE, other_transforms=_other_transforms
            )

        super().__init__(name, data_manager, root_cache_dir=root_cache_dir)

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

        logger.info("Creating coordinates dictionary.")
        coords = {
            "one": ["dim_one"],
            "sgrna": data.sgrna.cat.categories,
            "gene": data.hugo_symbol.cat.categories,
            "cell_line": data.depmap_id.cat.categories,
            "lineage": data.lineage.cat.categories,
            "batch": data.p_dna_batch.cat.categories,
        }

        logger.info("Creating RNA expression matrix.")
        rna_expr_matrix = _rna_expr_gene_lineage_matrix(data, "rna_expr_gene_lineage")
        # _sgrna_by_cell_line = (co_idx.n_sgrnas, co_idx.n_celllines)

        logger.info("Building PyMC3 model.")
        with pm.Model(coords=coords) as model:
            s = pm.Data("sgrna_idx", co_idx.sgrna_idx)
            g_s = pm.Data("sgrna_to_gene_idx", co_idx.sgrna_to_gene_idx)
            c = pm.Data("cell_line_idx", co_idx.cellline_idx)
            l_c = pm.Data("cell_line_to_lineage_idx", co_idx.cellline_to_lineage_idx)
            # b = pm.Data("batch_idx", b_idx.batch_idx)
            rna_expr = pm.Data("rna_expression_gl", rna_expr_matrix)
            ct_initial = pm.Data("ct_initial", data.counts_initial_adj.values)
            ct_final = pm.Data("ct_final", data.counts_final.values)

            h = _gene_by_cell_line_hierarchical_structure(
                "h", l_c_idx=l_c, centered=False
            )
            q = _gene_by_cell_line_hierarchical_structure(
                "q", l_c_idx=l_c, centered=False
            )

            # shape: [gene x cell line]
            mu_beta = pm.Deterministic("mu_beta", h + q * rna_expr)

            sigma_sigma_beta = pm.Exponential("sigma_sigma_beta", 0.2)
            sigma_beta = pm.Exponential(
                "sigma_beta", sigma_sigma_beta, dims=("gene", "one")
            )

            beta = pm.Normal(
                "beta",
                mu_beta[g_s, :],
                sigma_beta[g_s, :],
                dims=("sgrna", "cell_line"),
            )

            eta = pm.Deterministic("eta", beta[s, c])
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            alpha = pm.Exponential("alpha", 1 / 10)
            y = pm.NegativeBinomial(  # noqa: F841
                "y", ct_initial * mu, alpha, observed=ct_final
            )

        return model, "y"


def _gene_by_cell_line_hierarchical_structure(
    name: str,
    l_c_idx: Union[np.ndarray, pm.Data],
    centered: bool = True,
) -> pmFreeRV:
    mu_mu_x = pm.Normal(f"mu_mu_{name}", 0, 1)
    sigma_mu_x = pm.HalfNormal(f"sigma_mu_{name}", 1)

    if centered:
        logger.info(f"Centered parameterization for var 'mu_{name}'.")
        mu_x = pm.Normal(f"mu_{name}", mu_mu_x, sigma_mu_x, dims=("gene", "lineage"))
    else:
        logger.info(f"Non-centered parameterization for var 'mu_{name}'.")
        mu_x_delta = pm.Normal(f"mu_{name}_Δ", 0, 1, dims=("gene", "lineage"))
        mu_x = pm.Deterministic(f"mu_{name}", mu_mu_x + mu_x_delta * sigma_mu_x)

    sigma_sigma_x = pm.HalfNormal(f"sigma_sigma_{name}", 1)
    sigma_x = pm.HalfNormal(f"sigma_{name}", sigma_sigma_x, dims=("one", "lineage"))

    if centered:
        logger.info(f"Centered parameterization for var '{name}'.")
        x = pm.Normal(
            name,
            mu_x[:, l_c_idx],
            sigma_x[:, l_c_idx],
            dims=("gene", "cell_line"),
        )
    else:
        logger.info(f"Non-centered parameterization for var '{name}'.")
        x_delta = pm.Normal(f"{name}_Δ", 0.0, 1.0, dims=("gene", "cell_line"))
        x = pm.Deterministic(
            name,
            mu_x[:, l_c_idx] + x_delta * sigma_x[:, l_c_idx],
        )

    return x


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
