"""A hierarchical negative binomial generalized linear model for a single lineage."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import aesara.tensor as at
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.math as pmmath
from pandera import Check, Column, DataFrameSchema
from pydantic import BaseModel

from speclet.data_processing.common import get_cats, get_indices
from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
    common_indices,
    data_batch_indices,
    set_achilles_categorical_columns,
    zscale_cna_by_group,
    zscale_rna_expression_by_gene,
)
from speclet.data_processing.validation import (
    check_finite,
    check_nonnegative,
    check_single_unique_value,
    check_unique_groups,
)
from speclet.data_processing.vectors import squish_array
from speclet.loggers import logger
from speclet.managers.data_managers import CancerGeneDataManager as CancerGeneDM
from speclet.project_enums import ModelFitMethod


class TooFewGenes(BaseException):
    """Too few genes."""

    ...


class TooFewCellLines(BaseException):
    """Too few cell lines."""

    ...


@dataclass
class LineageHierNegBinomModelData:
    """Data for `LineageHierNegBinomModel`."""

    N: int  # total number of data points
    S: int  # number of sgRNAs
    G: int  # number of genes
    C: int  # number of cell lines
    SC: int  # number of screens
    CG: int  # number of cancer genes
    ct_initial: npt.NDArray[np.float32]
    ct_final: npt.NDArray[np.int32]
    sgrna_idx: npt.NDArray[np.int32]
    gene_idx: npt.NDArray[np.int32]
    sgrna_to_gene_idx: npt.NDArray[np.int32]
    cellline_idx: npt.NDArray[np.int32]
    screen_idx: npt.NDArray[np.int32]
    copy_number: npt.NDArray[np.float32]
    copy_number_z_gene: npt.NDArray[np.float32]
    copy_number_z_cell: npt.NDArray[np.float32]
    log_rna_expr: npt.NDArray[np.float32]
    z_log_rna_gene: npt.NDArray[np.float32]
    m_log_rna_gene: npt.NDArray[np.float32]
    is_mutated: npt.NDArray[np.int32]
    comutation_matrix: npt.NDArray[np.int32]
    coords: dict[str, list[str]]


class LineageHierNegBinomModelConfig(BaseModel):
    """Single-lineage hierarchical negative binominal model configuration."""

    lineage: str
    lfc_limits: tuple[float, float] = (-5.0, 5.0)
    min_n_cancer_genes: int = 2
    min_frac_cancer_genes: float = 0.0


class LineageHierNegBinomModel:
    """A hierarchical negative binomial generalized linear model fora single lineage."""

    def __init__(self, **kwargs: Any) -> None:
        """Single-lineage hierarchical negative binominal model.

        All keyword-arguments are passed into a configuration pydantic model for
        data validation.

        Args:
            lineage (str): Lineage to model.
            lfc_limits (tuple[float, float]): Data points with log-fold change values
            outside of this range are dropped. Defaults to (-5, 5).
        """
        self._config = LineageHierNegBinomModelConfig(**kwargs)
        self.lineage = self._config.lineage
        self.lfc_limits = self._config.lfc_limits
        self.min_n_cancer_genes = self._config.min_n_cancer_genes
        self.min_frac_cancer_genes = self._config.min_frac_cancer_genes
        if self.min_n_cancer_genes < 2:
            logger.warning(
                "Setting `min_n_cancer_genes` less than "
                + "2 may result is some non-identifiability."
            )
        return None

    @property
    def data_schema(self) -> DataFrameSchema:
        """Expected data schema for this model."""
        return DataFrameSchema(
            {
                "counts_initial_adj": Column(
                    float, checks=[check_nonnegative(), check_finite()], nullable=False
                ),
                "counts_final": Column(
                    int,
                    checks=[check_nonnegative(), check_finite()],
                    nullable=False,
                    coerce=True,
                ),
                "lfc": Column(
                    float,
                    checks=[
                        Check.in_range(
                            min_value=self.lfc_limits[0],
                            max_value=self.lfc_limits[1],
                            include_min=True,
                            include_max=True,
                        )
                    ],
                    nullable=False,
                ),
                "sgrna": Column("category"),
                "hugo_symbol": Column(
                    "category",
                    checks=[
                        # A sgRNA maps to a single gene ("hugo_symbol").
                        Check(check_unique_groups, groupby="sgrna"),
                    ],
                ),
                "depmap_id": Column("category"),
                "lineage": Column(
                    "category", checks=[check_single_unique_value(self.lineage)]
                ),
                "copy_number": Column(
                    float, checks=[check_nonnegative(), check_finite()]
                ),
                "rna_expr": Column(float, nullable=False),
                "z_rna_gene": Column(float, checks=[check_finite()]),
                "z_cn_gene": Column(float, checks=[check_finite()]),
                "z_cn_cell_line": Column(float, checks=[check_finite()]),
                "is_mutated": Column(bool, nullable=False),
                "screen": Column("category", nullable=False),
            }
        )

    def vars_regex(self, fit_method: ModelFitMethod | None = None) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        _vars = [
            "~^mu$",
            "~^eta$",
            "~^delta_.*",
            "~.*effect$",
            "~^celllines_chol_cov.*$",
            "~^.*celllines$",
            "~^genes_chol_cov.*$",
            "~^.*genes$",
        ]
        return _vars

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data according to this model's requirements.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Validated data.
        """
        return self.data_schema.validate(data)

    def _model_coords(
        self,
        valid_data: pd.DataFrame,
        cancer_genes: list[str],
    ) -> dict[str, list[str]]:
        return {
            "sgrna": get_cats(valid_data, "sgrna"),
            "gene": get_cats(valid_data, "hugo_symbol"),
            "cell_line": get_cats(valid_data, "depmap_id"),
            "screen": get_cats(valid_data, "screen"),
            "cancer_gene": cancer_genes,
            "one": ["1"],
        }

    def make_data_structure(self, data: pd.DataFrame) -> LineageHierNegBinomModelData:
        """Create the model data structure."""
        # Indices
        indices = common_indices(data)
        batch_indices = data_batch_indices(data)

        # Cancer genes and mutations.
        lineage_cancer_genes = CancerGeneDM().cosmic_cancer_genes()[self.lineage]
        cancer_gene_set: set[str] = set()

        for genes in lineage_cancer_genes.values():
            cancer_gene_set = cancer_gene_set.union(genes)

        # Only keep cancer genes that are included in the CRISPR screen data.
        cancer_gene_set = cancer_gene_set.intersection(data["hugo_symbol"])
        if len(cancer_gene_set) == 0:
            logger.warning("No cancer genes are in the data set.")

        cancer_genes = list(cancer_gene_set)
        cancer_genes.sort()
        coords = self._model_coords(data, cancer_genes)

        is_mutated = target_gene_is_mutated_vector(data)
        comutation_matrix, cancer_genes = make_cancer_gene_comutation_matrix(
            data,
            cancer_genes,
            min_n=self.min_n_cancer_genes,
            min_freq=self.min_frac_cancer_genes,
        )
        coords["cancer_gene"] = cancer_genes

        return LineageHierNegBinomModelData(
            N=data.shape[0],
            S=indices.n_sgrnas,
            G=indices.n_genes,
            C=indices.n_celllines,
            SC=batch_indices.n_screens,
            CG=len(cancer_genes),
            ct_initial=data.counts_initial_adj.values.astype(np.float32),
            ct_final=data.counts_final.values.astype(np.int32),
            sgrna_idx=indices.sgrna_idx.astype(np.int32),
            gene_idx=indices.gene_idx.astype(np.int32),
            sgrna_to_gene_idx=indices.sgrna_to_gene_idx.astype(np.int32),
            cellline_idx=indices.cellline_idx.astype(np.int32),
            screen_idx=batch_indices.screen_idx.astype(np.int32),
            copy_number=data.copy_number.values.astype(np.float32),
            copy_number_z_gene=data.z_cn_gene.values.astype(np.float32),
            copy_number_z_cell=data.z_cn_cell_line.values.astype(np.float32),
            log_rna_expr=data.rna_expr.values.astype(np.float32),
            z_log_rna_gene=data.z_rna_gene.values.astype(np.float32),
            m_log_rna_gene=data.m_rna_gene.values.astype(np.float32),
            is_mutated=is_mutated.astype(np.int32),
            comutation_matrix=comutation_matrix.astype(np.int32),
            coords=coords,
        )

    def data_processing_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data processing pipeline.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Processed and validated modeling data.
        """
        logger.info("Processing data for modeling.")
        logger.info(f"LFC limits: {self.lfc_limits}")
        min_lfc, max_lfc = self.lfc_limits
        initial_size = data.shape[0]
        data = (
            data.dropna(
                axis=0,
                how="any",
                subset=["counts_final", "counts_initial", "copy_number"],
            )
            .query(f"{min_lfc} <= lfc <= {max_lfc}")
            .reset_index(drop=True)
            .pipe(
                zscale_rna_expression_by_gene,
                new_col="z_rna_gene",
                lower_bound=-5,
                upper_bound=5,
                center_metric="mean",
            )
            .pipe(
                zscale_rna_expression_by_gene,
                new_col="m_rna_gene",
                lower_bound=-5,
                upper_bound=5,
                center_metric="median",
            )
            .pipe(
                zscale_cna_by_group,
                groupby_cols=["hugo_symbol"],
                new_col="z_cn_gene",
                cn_max=7,
                center=None,
            )
            .pipe(
                zscale_cna_by_group,
                groupby_cols=["depmap_id"],
                new_col="z_cn_cell_line",
                cn_max=7,
                center=None,
            )
            .assign(
                z_cn_gene=lambda d: squish_array(d["z_cn_gene"], lower=-5.0, upper=5.0),
                z_cn_cell_line=lambda d: squish_array(
                    d["z_cn_cell_line"], lower=-5.0, upper=5.0
                ),
            )
            .pipe(append_total_read_counts)
            .pipe(add_useful_read_count_columns)
            .pipe(set_achilles_categorical_columns, sort_cats=True, skip_if_cat=False)
            .pipe(self.validate_data)
        )
        final_size = data.shape[0]
        logger.warning(f"number of data points dropped: {initial_size - final_size}")
        return data

    def _pre_model_messages(self, model_data: LineageHierNegBinomModelData) -> None:
        logger.info(f"Lineage: {self.lineage}")
        logger.info(f"Number of genes: {model_data.G}")
        logger.info(f"Number of sgRNA: {model_data.S}")
        logger.info(f"Number of cell lines: {model_data.C}")
        logger.info(f"Number of cancer genes: {model_data.CG}")
        logger.info(f"Number of screens: {model_data.SC}")
        logger.info(f"Number of data points: {model_data.N}")

        if model_data.G < 2:
            raise TooFewGenes(model_data.G)
        if model_data.C < 2:
            raise TooFewCellLines(model_data.C)
        return None

    def pymc_model(
        self,
        data: pd.DataFrame,
        skip_data_processing: bool = False,
    ) -> pm.Model:
        """Hierarchical negative binomial model in PyMC.

        Args:
            data (pd.DataFrame): Data to model.
            skip_data_processing (bool, optional). Skip data pre-processing step?
            Defaults to `False`.

        Returns:
            pm.Model: PyMC model.
        """
        if not skip_data_processing:
            data = self.data_processing_pipeline(data)
        model_data = self.make_data_structure(data)
        self._pre_model_messages(model_data)

        # Multi-dimensional parameter coordinates (labels).
        coords = model_data.coords
        # Data.
        rna = model_data.m_log_rna_gene
        cn_gene = model_data.copy_number_z_gene
        mut = model_data.is_mutated
        cg_mut = model_data.comutation_matrix
        logger.debug(f"shape of cancer gene matrix: {cg_mut.shape}")
        # Indexing arrays.
        s_to_g = model_data.sgrna_to_gene_idx
        s = model_data.sgrna_idx
        g = model_data.gene_idx
        # Sizes.
        n_G = model_data.G
        n_CG = model_data.CG
        n_gene_vars = 4 + n_CG

        with pm.Model(coords=coords) as model:
            # Gene varying effects covariance matrix.
            g_chol, _, g_sigmas = pm.LKJCholeskyCov(
                "genes_chol_cov",
                eta=2,
                n=n_gene_vars,
                sd_dist=pm.Exponential.dist(2, shape=n_gene_vars),
                compute_corr=True,
            )
            for i, var_name in enumerate(["mu_a", "b", "d", "f"]):
                pm.Deterministic(f"sigma_{var_name}", g_sigmas[i])
            pm.Deterministic("sigma_h", g_sigmas[4:])

            # Gene varying effects.
            mu_mu_a = pm.Normal("mu_mu_a", 0, 0.5)
            mu_b = pm.Normal("mu_b", -0.5, 0.5)
            mu_d = pm.Normal("mu_d", 0, 0.2)
            mu_f = pm.Normal("mu_f", 0, 0.2)
            mu_h = pm.Normal("mu_h", 0, 0.1, dims="cancer_gene")
            _mu_genes = [mu_mu_a, mu_b, mu_d, mu_f] + [mu_h[i] for i in range(n_CG)]
            mu_genes = at.stack(_mu_genes, axis=0)
            delta_genes = pm.Normal("delta_genes", 0, 1, shape=(n_gene_vars, n_G))
            genes = mu_genes + at.dot(g_chol, delta_genes).T
            mu_a = pm.Deterministic("mu_a", genes[:, 0], dims="gene")
            b = pm.Deterministic("b", genes[:, 1], dims="gene")
            d = pm.Deterministic("d", genes[:, 2], dims="gene")
            f = pm.Deterministic("f", genes[:, 3], dims="gene")
            h = pm.Deterministic("h", genes[:, 4:], dims=("gene", "cancer_gene"))

            sigma_a = pm.Exponential("sigma_a", 2)
            delta_a = pm.Normal("delta_a", 0, 1, dims="sgrna")
            a = pm.Deterministic("a", mu_a[s_to_g] + delta_a * sigma_a, dims="sgrna")

            gene_effect = (
                a[s]
                + b[g] * rna
                + d[g] * cn_gene
                + f[g] * mut
                + at.sum(h[g, :] * cg_mut, axis=1)
            )
            eta = pm.Deterministic("eta", gene_effect + np.log(model_data.ct_initial))
            # eta = gene_effect + np.log(model_data.ct_initial)
            mu = pmmath.exp(eta)

            alpha = pm.Gamma("alpha", 10, 1)
            pm.NegativeBinomial(
                "ct_final",
                mu,
                alpha,
                observed=model_data.ct_final,
            )
        return model


def target_gene_is_mutated_vector(data: pd.DataFrame) -> npt.NDArray[np.int32]:
    """Create a target gene mutation vector.

    Accounts for an issue that can lead to non-identifiability where the target gene is
    mutated in all cell lines. This would be co-linear with the varying intercept.

    Args:
        data (pd.DataFrame): CRISPR data.

    Returns:
        npt.NDArray[np.int32]: Binary vector for if the target gene is mutated.
    """
    always_mutated_genes = (
        data.copy()[["depmap_id", "hugo_symbol", "is_mutated"]]
        .drop_duplicates()
        .groupby(["hugo_symbol"])["is_mutated"]
        .mean()
        .reset_index(drop=False)
        .query("is_mutated >= 0.99")["hugo_symbol"]
        .tolist()
    )
    logger.info(
        f"number of genes mutated in all cells lines: {len(always_mutated_genes)}"
    )
    logger.debug(f"Genes always mutated: {', '.join(always_mutated_genes)}")
    mut_ary = data["is_mutated"].values.astype(np.int32)
    idx = np.asarray([g in always_mutated_genes for g in data["hugo_symbol"]])
    mut_ary[idx] = 0
    return mut_ary


def _cell_line_by_cancer_gene_mutation_matrix(
    data: pd.DataFrame, cancer_genes: list[str]
) -> npt.NDArray[np.int32]:
    """Create a binary matrix of [cancer gene x cell line].

    I did this verbosely with a numpy matrix and iteration to make sure I didn't drop
    any cell lines or cancer genes never mutated and to ensure the order of each group.
    """
    cells = get_cats(data, "depmap_id")
    mat = np.zeros((len(cells), len(cancer_genes)))
    mutations = (
        data.copy()[["depmap_id", "hugo_symbol", "is_mutated"]]
        .drop_duplicates()
        .filter_column_isin("hugo_symbol", cancer_genes)
        .reset_index(drop=True)
        .astype({"is_mutated": np.int32})
    )
    for (i, cell), (j, gene) in product(enumerate(cells), enumerate(cancer_genes)):
        _query = f"depmap_id == '{cell}' and hugo_symbol == '{gene}'"
        is_mut = mutations.query(_query)["is_mutated"]
        assert len(is_mut) == 1
        mat[i, j] = is_mut.values[0]
    return mat.astype(np.int32)


def _trim_cancer_genes(
    cg_mut_matrix: npt.NDArray[np.int32],
    cancer_genes: list[str],
    min_n: int = 1,
    min_freq: float = 0.0,
) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Trim cancer genes and mutation matrix to avoid colinearities.

    Corrects for:
        1. remove cancer genes never mutated (or above a threshold)
        2. remove cancer genes always mutated
        3. merge cancer genes that are perfectly co-mutated
    """
    # Identifying cancer genes to remove.
    all_mut = np.all(cg_mut_matrix, axis=0)
    low_n_mut = np.sum(cg_mut_matrix, axis=0) < min_n
    low_freq_mut = np.mean(cg_mut_matrix, axis=0) < min_freq
    drop_idx = all_mut + low_n_mut + low_freq_mut

    # Logging.
    _dropped_cancer_genes = list(np.asarray(cancer_genes)[drop_idx])
    logger.info(f"Dropping {len(_dropped_cancer_genes)} cancer genes.")
    logger.debug(f"Dropped cancer genes: {_dropped_cancer_genes}")

    # Execute changes.
    cg_mut_matrix = cg_mut_matrix[:, ~drop_idx]
    cancer_genes = list(np.asarray(cancer_genes)[~drop_idx])
    assert len(cancer_genes) == cg_mut_matrix.shape[1], "Shape mis-match."
    return cg_mut_matrix, cancer_genes


def _get_colinear_columns(mat: np.ndarray, col: int) -> list[int]:
    col_vals = mat[:, col]
    to_merge: list[int] = [col]
    for i in range(mat.shape[1]):
        if i == col:
            continue
        if np.all(col_vals == mat[:, i]):
            to_merge.append(i)
    return to_merge


def _merge_colinear_columns(
    mat: np.ndarray, colinear_cols: list[int], keep_pos: int
) -> np.ndarray:
    drop_cols = [c for c in colinear_cols if c != keep_pos]
    keep_idx = np.array([i not in drop_cols for i in range(mat.shape[1])])
    return mat[:, keep_idx]


def _merge_cancer_genes(
    genes: list[str], colinear_cols: list[int], keep_pos: int
) -> list[str]:
    merge_genes = [genes[i] for i in colinear_cols]
    new_gene = "|".join(merge_genes)
    genes[keep_pos] = new_gene
    return [g for g in genes if g not in merge_genes]


def _merge_colinear_cancer_genes(
    cg_mut_matrix: npt.NDArray[np.int32], cancer_genes: list[str]
) -> tuple[npt.NDArray[np.int32], list[str]]:
    n_cg = len(cancer_genes) + 1
    while n_cg != len(cancer_genes) and len(cancer_genes) > 0:
        n_cg = len(cancer_genes)
        for i in range(n_cg):
            colinear_cols = _get_colinear_columns(cg_mut_matrix, i)
            if len(colinear_cols) > 1:
                cg_mut_matrix = _merge_colinear_columns(
                    cg_mut_matrix, colinear_cols, keep_pos=i
                )
                cancer_genes = _merge_cancer_genes(
                    cancer_genes, colinear_cols, keep_pos=i
                )
                break
    return cg_mut_matrix, cancer_genes


def _set_cancer_gene_rows_to_zero(
    data: pd.DataFrame,
    cancer_gene_mut_mat: npt.NDArray[np.int32],
    cancer_genes: list[str],
) -> npt.NDArray[np.int32]:
    """For cancer genes, set their co-mutation value to 0.

    This avoids colinearities between the comutation variables and the target gene
    mutation variable.

    Note that merged genes are accounted for by setting both genes to 0.
    """
    cancer_gene_mut_mat = cancer_gene_mut_mat.copy()
    for j, merged_genes in enumerate(cancer_genes):
        for gene in merged_genes.split("|"):
            gene_idx = data["hugo_symbol"] == gene
            cancer_gene_mut_mat[gene_idx, j] = 0
    return cancer_gene_mut_mat


def make_cancer_gene_comutation_matrix(
    data: pd.DataFrame, cancer_genes: list[str], min_n: int = 1, min_freq: float = 0.0
) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Generate a cancer gene comutation matrix.

    Args:
        data (pd.DataFrame): CRISPR data frame.
        cancer_genes (list[str]): List of cancer genes (in desired order).
        min_n (int, optional): Minimum number of cell lines with the cancer gene mutated
        in order to use the cancer gene. Defaults to 1.
        min_freq (float, optional): Minimum fraction of cell lines (mutation frequency)
        with the cancer gene mutated in order to use the cancer gene. Defaults to 0.0.

    Returns:
        tuple[npt.NDArray[np.int32], list[str]]: Cancer gene comutation matrix of shape
        [num. data points x num. cancer gene] and the updated list of cancer genes.
    """
    c_by_g_matrix = _cell_line_by_cancer_gene_mutation_matrix(data, cancer_genes)

    c_by_g_matrix, new_cg = _trim_cancer_genes(
        c_by_g_matrix, cancer_genes, min_n=min_n, min_freq=min_freq
    )
    if len(new_cg) == 0:
        return np.zeros((len(data), 1), dtype=np.int32), new_cg

    c_by_g_matrix, new_merged_cg = _merge_colinear_cancer_genes(c_by_g_matrix, new_cg)
    if len(new_merged_cg) == 0:
        return np.zeros((len(data), 1), dtype=np.int32), new_merged_cg

    cell_line_idx = get_indices(data, "depmap_id")
    cg_mut_mat = c_by_g_matrix[cell_line_idx, :]
    cg_mut_mat = _set_cancer_gene_rows_to_zero(data, cg_mut_mat, new_merged_cg)

    assert cg_mut_mat.shape[0] == len(data)
    assert cg_mut_mat.shape[1] == len(new_merged_cg)
    return cg_mut_mat, new_merged_cg
