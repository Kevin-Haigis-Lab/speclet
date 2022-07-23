"""A hierarchical negative binomial generalized linear model for a single lineage."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.math as pmmath
from pandera import Check, Column, DataFrameSchema
from pydantic import BaseModel

from speclet.data_processing.common import get_cats
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
from speclet.managers.data_managers import LineageGeneMap
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

        comutation_matrix = make_cancer_gene_mutation_matrix(
            data, cancer_genes=cancer_genes, cell_lines=coords["cell_line"]
        )
        comutation_matrix, cancer_genes = remove_collinearity_in_comutation_matrix(
            comutation_matrix, cancer_genes
        )
        coords = self._model_coords(data, cancer_genes)
        data = remove_cancer_gene_mutations_from_mutation_data(
            data,
            cancer_genes=set(cancer_genes),
            mut_col="is_mutated",
            new_col="is_mutated_adj",
        )
        data = remove_mutations_for_fully_mutated_genes(
            data, mut_col="is_mutated_adj", new_col="is_mutated_adj"
        )
        mut = data["is_mutated_adj"].values

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
            is_mutated=mut.astype(np.int32),
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
        # Data
        rna = model_data.m_log_rna_gene
        # Indexing arrays.
        s_to_g = model_data.sgrna_to_gene_idx
        s = model_data.sgrna_idx
        g = model_data.gene_idx

        with pm.Model(coords=coords) as model:

            mu_b = pm.Normal("mu_b", -0.5, 1)
            sigma_b = pm.Exponential("sigma_b", 2)
            delta_b = pm.Normal("delta_b", 0, 1, dims="gene")
            b = pm.Normal("b", mu_b + delta_b * sigma_b, dims="gene")

            mu_mu_a = pm.Normal("mu_mu_a", 0, 1)
            sigma_mu_a = pm.Exponential("sigma_mu_a", 2)
            delta_mu_a = pm.Normal("delta_mu_a", 0, 1, dims="gene")
            mu_a = pm.Normal("mu_a", mu_mu_a + delta_mu_a * sigma_mu_a, dims="gene")
            sigma_a = pm.Exponential("sigma_a", 2)
            delta_a = pm.Normal("delta_a", 0, 1, dims="sgrna")
            a = pm.Normal("a", mu_a[s_to_g] + delta_a * sigma_a, dims="sgrna")

            gene_effect = a[s] + b[g] * rna
            eta = pm.Deterministic("eta", gene_effect + np.log(model_data.ct_initial))

            mu = pmmath.exp(eta)

            alpha = pm.Gamma("alpha", 10, 1)
            pm.NegativeBinomial(
                "ct_final",
                mu,
                alpha,
                observed=model_data.ct_final,
            )
        return model


def _collect_mutations_per_cell_line(data: pd.DataFrame) -> dict[str, set[str]]:
    mut_data = (
        data[["depmap_id", "hugo_symbol", "is_mutated"]]
        .drop_duplicates()
        .query("is_mutated")
        .reset_index(drop=True)
    )
    mutations: LineageGeneMap = {}
    for cl in data.depmap_id.unique():
        mutations[cl] = set(mut_data.query(f"depmap_id == '{cl}'").hugo_symbol.unique())
    return mutations


def make_cancer_gene_mutation_matrix(
    data: pd.DataFrame, cancer_genes: list[str], cell_lines: list[str]
) -> npt.NDArray[np.int_]:
    """Make a cancer gene x cell line mutation matrix.

    Args:
        data (pd.DataFrame): DepMap data frame.
        cancer_genes (list[str]): Sorted list of cancer genes.
        cell_lines (list[str]): Sorted list of cell lines.

    Returns:
        npt.NDArray[np.int_]: Binary mutation matrix.
    """
    cell_mutations = _collect_mutations_per_cell_line(data)
    mut_mat = np.zeros(shape=(len(cancer_genes), len(cell_lines)), dtype=int)
    for (i, cg), (j, cell) in product(enumerate(cancer_genes), enumerate(cell_lines)):

        if cg in cell_mutations[cell]:
            mut_mat[i, j] = 1
    return mut_mat


def remove_cancer_gene_mutations_from_mutation_data(
    data: pd.DataFrame, cancer_genes: set[str], mut_col: str, new_col: str
) -> pd.DataFrame:
    """Augment mutation data to account for cancer gene co-mutations.

    If the gene is in the collection of cancer genes, then set its mutation status in
    the original data to 0. This will result in the mutation effect for cancer genes
    being estimated in the co-mutation variable instead of the normal mutation effect
    variable.

    Args:
        data (pd.DataFrame): DepMap data frame.
        cancer_genes (set[str]): Collection of cancer genes.
        mut_col (str): Mutation column name.
        new_col (str): New column name.

    Returns:
        pd.DataFrame: Augmented mutation data.
    """
    mut = data[mut_col].values.astype(int)
    for i, gene in enumerate(data["hugo_symbol"]):
        if gene in cancer_genes:
            mut[i] = 0
    data[new_col] = mut
    return data


def remove_mutations_for_fully_mutated_genes(
    data: pd.DataFrame, mut_col: str, new_col: str
) -> pd.DataFrame:
    """Remove mutations for genes mutated in every cell line.

    If a gene is mutated in every cell line, the varying mutation effect for that gene
    in `m` will be perfectly co-linear with the varying intercept for the gene.

    Args:
        data (pd.DataFrame): CRISPR data.
        mut_col (str): Mutation column.
        new_col (str): New mutation column.

    Returns:
        pd.DataFrame: Adjusted data frame.
    """
    mut_freq = data.copy().groupby(["hugo_symbol"])[mut_col].mean()
    all_mut_genes = mut_freq[mut_freq == 1].index.tolist()
    adj_muts = data[mut_col].copy()
    adj_muts[data["hugo_symbol"].isin(all_mut_genes)] = 0
    data[new_col] = adj_muts
    return data


def _remove_cancer_genes_never_mutated(
    comut_mat: npt.NDArray[np.int32], cancer_genes: list[str]
) -> tuple[npt.NDArray[np.int32], list[str]]:
    any_muts = np.any(comut_mat, axis=1)
    new_comut_mat = comut_mat.copy()[any_muts, :]
    new_cancer_genes = [g for g, i in zip(cancer_genes, any_muts) if i]
    return new_comut_mat, new_cancer_genes


def _remove_cancer_genes_always_mutated(
    comut_mat: npt.NDArray[np.int32], cancer_genes: list[str]
) -> tuple[npt.NDArray[np.int32], list[str]]:
    all_muts = np.all(comut_mat, axis=1)
    new_comut_mat = comut_mat.copy()[~all_muts, :]
    new_cancer_genes = [g for g, i in zip(cancer_genes, all_muts) if not i]
    return new_comut_mat, new_cancer_genes


def _set_mutations_false_if_all_mutated_in_cell(
    comut_mat: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    all_mut = np.all(comut_mat, axis=0)
    new_comut_mat = comut_mat.copy()
    new_comut_mat[:, all_mut] = 0
    return new_comut_mat


def remove_collinearity_in_comutation_matrix(
    comutation_matrix: npt.NDArray[np.int32], cancer_genes: list[str]
) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Remove patterns that produce colinearity in the comutation matrix.

    Args:
        comutation_matrix (npt.NDArray[np.int32]): Comutation matrix.
        cancer_genes (list[str]): Cancer genes.

    Returns:
        tuple[npt.NDArray[np.int32], list[str]]: Update comutation matrix and list of
        cancer genes.
    """
    n_cancer_genes = len(cancer_genes) + 1
    prev_comut_mat = np.zeros_like(comutation_matrix)
    while (
        n_cancer_genes != len(cancer_genes)
        and not np.all(comutation_matrix == prev_comut_mat)
        and n_cancer_genes > 0
    ):
        n_cancer_genes = len(cancer_genes)
        prev_comut_mat = comutation_matrix.copy()
        comutation_matrix = _set_mutations_false_if_all_mutated_in_cell(
            comutation_matrix
        )
        comutation_matrix, cancer_genes = _remove_cancer_genes_never_mutated(
            comutation_matrix, cancer_genes
        )
        comutation_matrix, cancer_genes = _remove_cancer_genes_always_mutated(
            comutation_matrix, cancer_genes
        )
    return comutation_matrix, cancer_genes
