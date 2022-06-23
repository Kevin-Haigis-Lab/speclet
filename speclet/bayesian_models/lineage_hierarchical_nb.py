"""A hierarchical negative binomial generalized linear model for a single lineage."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.math as pmmath
from aesara import tensor as at
from aesara.tensor.random.op import RandomVariable
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
    ct_initial: npt.NDArray[np.int32]
    ct_final: npt.NDArray[np.float32]
    sgrna_idx: npt.NDArray[np.int32]
    gene_idx: npt.NDArray[np.int32]
    cellline_idx: npt.NDArray[np.int32]
    screen_idx: npt.NDArray[np.int32]
    copy_number: npt.NDArray[np.float32]
    copy_number_z_gene: npt.NDArray[np.float32]
    copy_number_z_cell: npt.NDArray[np.float32]
    log_rna_expr: npt.NDArray[np.float32]
    is_mutated: npt.NDArray[np.int32]
    comutation_matrix: npt.NDArray[np.int32]
    coords: dict[str, list[str]]


class LineageHierNegBinomModelConfig(BaseModel):
    """Single-lineage hierarchical negative binominal model configuration."""

    lineage: str


class LineageHierNegBinomModel:
    """A hierarchical negative binomial generalized linear model fora single lineage."""

    def __init__(self, **kwargs: Any) -> None:
        """Single-lineage hierarchical negative binominal model.

        All keyword-arguments are passed into a configuration pydantic model for
        data validation.

        Args:
            lineage (str): Lineage to model.
        """
        self._config = LineageHierNegBinomModelConfig(**kwargs)
        self.lineage = self._config.lineage
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
                "log_rna_expr": Column(float, checks=[check_finite()]),
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

        n_cancer_genes = len(cancer_genes)
        while n_cancer_genes == len(cancer_genes) and n_cancer_genes > 0:
            n_cancer_genes = len(cancer_genes)
            comutation_matrix, cancer_genes = drop_cancer_genes_without_wt_and_mutation(
                comutation_matrix, cancer_genes
            )

        coords = self._model_coords(data, cancer_genes)
        mut = augmented_mutation_data(data, cancer_genes=set(cancer_genes))

        return LineageHierNegBinomModelData(
            N=data.shape[0],
            S=indices.n_sgrnas,
            G=indices.n_genes,
            C=indices.n_celllines,
            SC=batch_indices.n_screens,
            CG=len(cancer_genes),
            ct_initial=data.counts_initial_adj.values.astype(np.int32),
            ct_final=data.counts_final.values.astype(np.int32),
            sgrna_idx=indices.sgrna_idx.astype(np.int32),
            gene_idx=indices.gene_idx.astype(np.int32),
            cellline_idx=indices.cellline_idx.astype(np.int32),
            screen_idx=batch_indices.screen_idx.astype(np.int32),
            copy_number=data.copy_number.values.astype(np.float32),
            copy_number_z_gene=data.z_cn_gene.values.astype(np.float32),
            copy_number_z_cell=data.z_cn_cell_line.values.astype(np.float32),
            log_rna_expr=data.log_rna_expr.values.astype(np.float32),
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
        return (
            data.dropna(
                axis=0,
                how="any",
                subset=["counts_final", "counts_initial", "copy_number"],
            )
            .pipe(zscale_rna_expression_by_gene, new_col="z_rna_gene")
            .pipe(
                zscale_cna_by_group,
                groupby_cols=["hugo_symbol"],
                new_col="z_cn_gene",
                cn_max=7,
                center=1,
            )
            .pipe(
                zscale_cna_by_group,
                groupby_cols=["depmap_id"],
                new_col="z_cn_cell_line",
                cn_max=7,
                center=1,
            )
            .assign(
                z_cn_gene=lambda d: squish_array(d.z_cn_gene, lower=-5.0, upper=5.0),
                z_cn_cell_line=lambda d: squish_array(
                    d.z_cn_cell_line, lower=-5.0, upper=5.0
                ),
                log_rna_expr=lambda d: np.log(d.rna_expr + 1.0),
            )
            .pipe(append_total_read_counts)
            .pipe(add_useful_read_count_columns)
            .pipe(set_achilles_categorical_columns, sort_cats=True, skip_if_cat=False)
            .pipe(self.validate_data)
        )

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

        # Multi-dimensional parameter coordinates (labels).
        coords = model_data.coords
        # Indexing arrays.
        s = model_data.sgrna_idx
        c = model_data.cellline_idx
        g = model_data.gene_idx
        s = model_data.screen_idx
        # Data.
        cn_gene = model_data.copy_number_z_gene
        cn_cell = model_data.copy_number_z_cell
        rna = model_data.log_rna_expr
        mut = model_data.is_mutated
        M = model_data.comutation_matrix

        n_C = model_data.C
        n_G = model_data.G
        n_CG = model_data.CG
        n_gene_vars = 4 + n_CG

        if n_G < 2:
            raise TooFewGenes(n_G)
        if n_C < 2:
            raise TooFewCellLines(n_C)

        def _sigma_dist(name: str) -> RandomVariable:
            return pm.HalfNormal(name, 1)

        with pm.Model(coords=coords) as model:
            z = pm.Normal("z", 0, 2.5, initval=0)

            sigma_a = _sigma_dist("sigma_a")
            delta_a = pm.Normal("delta_a", 0, 1, dims=("sgrna"))
            a = pm.Deterministic("a", delta_a * sigma_a, dims=("sgrna"))

            cl_chol, _, cl_sigmas = pm.LKJCholeskyCov(
                "celllines_chol_cov", eta=2, n=2, sd_dist=pm.HalfNormal.dist(1)
            )
            pm.Deterministic("sigma_b", cl_sigmas[0])
            pm.Deterministic("sigma_f", cl_sigmas[1])

            mu_b = 0
            mu_f = pm.Normal("mu_f", -0.5, 0.5)
            mu_celllines = at.stack([mu_b, mu_f])
            delta_celllines = pm.Normal("delta_celllines", 0, 1, shape=(n_C, 2))
            celllines = pm.Deterministic(
                "celllines", mu_celllines + at.dot(cl_chol, delta_celllines.T).T
            )
            b = pm.Deterministic("b", celllines[:, 0], dims="cell_line")
            f = pm.Deterministic("f", celllines[:, 1], dims="cell_line")

            g_chol, _, g_sigmas = pm.LKJCholeskyCov(
                "genes_chol_cov", eta=2, n=n_gene_vars, sd_dist=pm.HalfNormal.dist(1)
            )
            for i, var_name in enumerate(["d", "h", "k", "m"]):
                pm.Deterministic(f"sigma_{var_name}", g_sigmas[i])

            if n_CG > 0:
                pm.Deterministic("sigma_w", g_sigmas[4:])

            mu_d = 0
            mu_h = 0
            mu_k = pm.Normal("mu_k", -0.5, 0.5)
            mu_m = 0
            mu_w = [0] * n_CG
            mu_genes = at.stack([mu_d, mu_h, mu_k, mu_m] + mu_w)
            delta_genes = pm.Normal("delta_genes", 0, 1, shape=(n_G, n_gene_vars))
            genes = pm.Deterministic(
                "genes", mu_genes + at.dot(g_chol, delta_genes.T).T
            )
            d = pm.Deterministic("d", genes[:, 0], dims="gene")
            h = pm.Deterministic("h", genes[:, 1], dims="gene")
            k = pm.Deterministic("k", genes[:, 2], dims="gene")
            m = pm.Deterministic("m", genes[:, 3], dims="gene")

            w: RandomVariable | np.ndarray
            if n_CG > 0:
                w = pm.Deterministic("w", genes[:, 4:], dims=("gene", "cancer_gene"))
            else:
                w = np.zeros((n_G, n_CG))

            p: RandomVariable | np.ndarray
            if model_data.SC > 1:
                # Multiple screens.
                sigma_p = _sigma_dist("sigma_p")
                delta_p = pm.Normal("delta_p", 0, 1, dims=("gene", "screen"))
                p = pm.Deterministic("p", delta_p * sigma_p, dims=("gene", "screen"))
            else:
                # Single screen.
                logger.warning("Only 1 screen detected - ignoring variable `p`.")
                p = np.zeros(shape=(model_data.G, 1))

            gene_effect = pm.Deterministic(
                "gene_effect",
                d[g] + cn_gene * h[g] + rna * k[g] + mut * m[g] + at.dot(w, M)[g, c],
            )
            cell_effect = pm.Deterministic("cell_line_effect", b[c] + cn_cell * f[c])
            eta = pm.Deterministic(
                "eta", z + a[s] + gene_effect + cell_effect + p[g, s]
            )
            mu = pm.Deterministic("mu", pmmath.exp(eta))

            alpha = pm.Gamma("alpha", 2.0, 0.5)
            pm.NegativeBinomial(
                "ct_final",
                mu * model_data.ct_initial,
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


def augmented_mutation_data(
    data: pd.DataFrame, cancer_genes: set[str]
) -> npt.NDArray[np.int_]:
    """Augment mutation data to account for cancer gene co-mutations.

    If the gene is in the collection of cancer genes, then set its mutation status in
    the original data to 0. This will result in the mutation effect for cancer genes
    being estimated in the co-mutation variable instead of the normal mutation effect
    variable.

    Args:
        data (pd.DataFrame): DepMap data frame.
        cancer_genes (set[str]): Collection of cancer genes.

    Returns:
        np.ndarray: Augmented mutation data.
    """
    mut = data["is_mutated"].values.astype(int)
    for i, gene in enumerate(data["hugo_symbol"]):
        if gene in cancer_genes:
            mut[i] = 0
    return mut


def drop_cancer_genes_without_wt_and_mutation(
    comutation_matrix: npt.NDArray[np.int32], cancer_genes: list[str]
) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Drop cancer genes without any mutations.

    Args:
        comutation_matrix (npt.NDArray[np.int32]): Co-mutation matrix.
        cancer_genes (list[str]): Ordered list of cancer genes.

    Returns:
        tuple[npt.NDArray[np.int32], list[str]]: Modified co-mutation matrix and list of
        cancer genes.
    """
    any_muts = np.any(comutation_matrix, axis=1)
    all_muts = np.all(comutation_matrix, axis=1)
    assert any_muts.ndim == all_muts.ndim == 1
    assert any_muts.shape[0] == all_muts.shape[0] == len(cancer_genes)
    keep_idx = any_muts * ~all_muts
    new_comut_mat = comutation_matrix.copy()[keep_idx, :]
    new_cancer_genes = [g for g, i in zip(cancer_genes, keep_idx) if i]
    return new_comut_mat, new_cancer_genes


def genes_with_mutations_switch(
    genes: npt.NDArray[np.str_], muts: npt.NDArray[np.int32]
) -> npt.NDArray[np.bool_]:
    """Make a switch array for if a gene has any mutations or not.

    Args:
        genes (npt.NDArray[np.str_]): Gene array from the data frame.
        muts (npt.NDArray[np.int32]): Mutation data.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where `True` marks genes with at least one
        mutation and `False` for genes without any mutations.
    """
    assert genes.ndim == 1
    assert muts.ndim == 1
    assert len(genes) == len(muts)
    switch_ary = np.ones_like(genes, dtype=np.bool_)
    for gene in np.unique(genes):
        g_idx = genes == gene
        num_muts = np.sum(muts[g_idx] > 0)
        if num_muts == 0:
            switch_ary[g_idx] = False
    return switch_ary
