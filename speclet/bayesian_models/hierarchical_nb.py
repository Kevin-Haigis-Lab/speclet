"""A hierarchical negative binomial generialzed linear model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pymc as pm
import pymc.math as pmmath
from aesara import tensor as at
from pandera import Check, Column, DataFrameSchema

from speclet.data_processing.common import get_cats
from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
    common_indices,
    data_batch_indices,
    set_achilles_categorical_columns,
    zscale_cna_by_group,
    zscale_rna_expression_by_gene_lineage,
)
from speclet.data_processing.validation import (
    check_finite,
    check_nonnegative,
    check_unique_groups,
)
from speclet.data_processing.vectors import squish_array
from speclet.loggers import logger
from speclet.managers.data_managers import CancerGeneDataManager, LineageGeneMap
from speclet.project_enums import ModelFitMethod


@dataclass
class HierarchcalNegBinomModelData:
    """Data for `HierarchcalNegativeBinomialModel`."""

    N: int  # total number of data points
    S: int  # number of sgRNAs
    G: int  # number of genes
    C: int  # number of cell lines
    L: int  # number of lineages
    SC: int  # number of screens
    CG: int  # number of cancer genes
    ct_initial: np.ndarray
    ct_final: np.ndarray
    sgrna_idx: np.ndarray
    gene_idx: np.ndarray
    cellline_idx: np.ndarray
    lineage_idx: np.ndarray
    cellline_to_lineage_idx: np.ndarray
    screen_idx: np.ndarray
    copy_number: np.ndarray
    copy_number_z_gene: np.ndarray
    copy_number_z_cell: np.ndarray
    log_rna_expr: np.ndarray
    is_mutated: np.ndarray
    cancer_genes: LineageGeneMap
    comutation_matrix: np.ndarray
    coords: dict[str, list[str]]


@dataclass
class HierarchicalNegativeBinomialConfig:
    """Configuration for `HierarchcalNegativeBinomialModel`."""

    num_knots: int = 5


class HierarchcalNegativeBinomialModel:
    """A hierarchical negative binomial generialzed linear model."""

    def __init__(self) -> None:
        """Create a negative binomial Bayesian model object."""
        self._config = HierarchicalNegativeBinomialConfig()
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
                    "category",
                    checks=[
                        # A lineage maps to a single cell line.
                        Check(check_unique_groups, groupby="depmap_id"),
                    ],
                ),
                "copy_number": Column(
                    float, checks=[check_nonnegative(), check_finite()]
                ),
                "rna_expr": Column(float, nullable=False),
                "z_rna_gene_lineage": Column(float, checks=[check_finite()]),
                "log_rna_expr": Column(float, checks=[check_finite()]),
                "z_cn_gene": Column(float, checks=[check_finite()]),
                "z_cn_cell_line": Column(float, checks=[check_finite()]),
                "is_mutated": Column(bool, nullable=False),
                "screen": Column("category", nullable=False),
            }
        )

    def vars_regex(self, fit_method: ModelFitMethod) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        _vars = ["~^mu$", "~^eta$", "~^delta_.*", "~.*effect$"]
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
        cancer_genes: Optional[LineageGeneMap] = None,
    ) -> dict[str, list[str]]:
        coords = {
            "sgrna": get_cats(valid_data, "sgrna"),
            "gene": get_cats(valid_data, "hugo_symbol"),
            "cell_line": get_cats(valid_data, "depmap_id"),
            "lineage": get_cats(valid_data, "lineage"),
            "screen": get_cats(valid_data, "screen"),
            "one": ["1"],
        }
        if cancer_genes is not None:
            coords["cancer_gene"] = _collect_all_cancer_genes(cancer_genes)

        return coords

    def _make_data_structure(self, data: pd.DataFrame) -> HierarchcalNegBinomModelData:
        indices = common_indices(data)
        batch_indices = data_batch_indices(data)

        cancer_gene_manager = CancerGeneDataManager()
        cancer_genes = cancer_gene_manager.reduce_to_lineage(
            cancer_gene_manager.cosmic_cancer_genes()
        )
        cancer_genes = {
            line: cancer_genes[line] for line in data.lineage.cat.categories
        }

        coords = self._model_coords(data, cancer_genes)

        mut = _augmented_mutation_data(data, cancer_genes=cancer_genes)
        comutation_matrix = _make_cancer_gene_mutation_matrix(
            data,
            cancer_genes,
            cell_lines=coords["cell_line"],
            genes=coords["cancer_gene"],
        )
        # Add a 3rd dimension of length 1.
        comutation_matrix = comutation_matrix[None, :, :]

        return HierarchcalNegBinomModelData(
            N=data.shape[0],
            S=indices.n_sgrnas,
            G=indices.n_genes,
            C=indices.n_celllines,
            L=indices.n_lineages,
            SC=batch_indices.n_screens,
            CG=len(_collect_all_cancer_genes(cancer_genes)),
            ct_initial=data.counts_initial_adj.values.astype(float),
            ct_final=data.counts_final.values.astype(int),
            sgrna_idx=indices.sgrna_idx,
            gene_idx=indices.gene_idx,
            cellline_idx=indices.cellline_idx,
            cellline_to_lineage_idx=indices.cellline_to_lineage_idx,
            lineage_idx=indices.lineage_idx,
            screen_idx=batch_indices.screen_idx,
            copy_number=data.copy_number.values,
            copy_number_z_gene=data.z_cn_gene.values,
            copy_number_z_cell=data.z_cn_cell_line.values,
            log_rna_expr=data.log_rna_expr.values,
            is_mutated=mut,
            cancer_genes=cancer_genes,
            comutation_matrix=comutation_matrix,
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
            .pipe(zscale_rna_expression_by_gene_lineage, new_col="z_rna_gene_lineage")
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
        seed: Optional[int] = None,
        skip_data_processing: bool = False,
    ) -> pm.Model:
        """Hierarchical negative binomial model in PyMC.

        Args:
            data (pd.DataFrame): Data to model.
            seed (Optional[seed], optional): Random seed. Defaults to `None`.
            skip_data_processing (bool, optional). Skip data pre-processing step?
            Defaults to `False`.

        Returns:
            pm.Model: PyMC model.
        """
        if not skip_data_processing:
            data = self.data_processing_pipeline(data)
        model_data = self._make_data_structure(data)

        # Multi-dimensional parameter coordinates (labels).
        coords = model_data.coords
        # Indexing arrays.
        s = model_data.sgrna_idx
        c = model_data.cellline_idx
        g = model_data.gene_idx
        ll = model_data.lineage_idx
        s = model_data.screen_idx
        c_to_l = model_data.cellline_to_lineage_idx
        # Data.
        cn_gene = model_data.copy_number_z_gene
        cn_cell = model_data.copy_number_z_cell
        rna = model_data.log_rna_expr
        mut = model_data.is_mutated
        M = model_data.comutation_matrix

        with pm.Model(coords=coords, rng_seeder=seed) as model:
            z = pm.Normal("z", 0, 5, initval=0)

            sigma_a = pm.Gamma("sigma_a", 3, 1)
            delta_a = pm.Normal("delta_a", 0, 1, dims=("sgrna"))
            a = pm.Deterministic("a", delta_a * sigma_a, dims=("sgrna"))

            sigma_b = pm.Gamma("sigma_b", 3, 1)
            delta_b = pm.Normal("delta_b", 0, 1, dims=("cell_line"))
            b = pm.Deterministic("b", delta_b * sigma_b, dims=("cell_line"))

            sigma_d = pm.Gamma("sigma_d", 3, 1)
            delta_d = pm.Normal("delta_d", 0, 1, dims=("gene", "lineage"))
            d = pm.Deterministic("d", delta_d * sigma_d, dims=("gene", "lineage"))

            sigma_f = pm.Gamma("sigma_f", 3, 1)
            delta_f = pm.Normal("delta_f", 0, 1, dims=("cell_line"))
            f = pm.Deterministic("f", delta_f * sigma_f, dims=("cell_line"))

            sigma_h = pm.Gamma("sigma_h", 3, 1)
            delta_h = pm.Normal("delta_h", 0, 1, dims=("gene", "lineage"))
            h = pm.Deterministic("h", delta_h * sigma_h, dims=("gene", "lineage"))

            sigma_k = pm.Gamma("sigma_k", 3, 1)
            delta_k = pm.Normal("delta_k", 0, 1, dims=("gene", "lineage"))
            k = pm.Deterministic("k", delta_k * sigma_k, dims=("gene", "lineage"))

            sigma_m = pm.Gamma("sigma_m", 3, 1)
            delta_m = pm.Normal("delta_m", 0, 1, dims=("gene", "lineage"))
            m = pm.Deterministic("m", delta_m * sigma_m, dims=("gene", "lineage"))

            sigma_w = pm.Gamma("sigma_w", 3, 1)
            delta_w = pm.Normal(
                "delta_w", 0, 1, dims=("gene", "cancer_gene", "lineage")
            )
            w = pm.Deterministic(
                "w", delta_w * sigma_w, dims=("gene", "cancer_gene", "lineage")
            )

            sigma_p = pm.Gamma("sigma_p", 3, 1)
            p = pm.Normal("p", 0, sigma_p, dims=("gene", "screen"))

            # Note: This is a hack to get around some weird error with sampling with
            # the Numpyro JAX backend.
            _w_pieces = []
            for cell_i, line_i in enumerate(c_to_l):
                _m_slice = M[:, :, cell_i]
                _w_slice = w[:, :, line_i]
                _w_piece = (_w_slice * _m_slice).sum(axis=1)[:, None]
                _w_pieces.append(_w_piece)
            _w = pm.Deterministic(
                "_w", at.horizontal_stack(*_w_pieces), dims=("gene", "cell_line")
            )

            gene_effect = pm.Deterministic(
                "gene_effect",
                d[g, ll]
                + cn_gene * h[g, ll]
                + rna * k[g, ll]
                + mut * m[g, ll]
                + _w[g, c],
            )
            cell_effect = pm.Deterministic("cell_line_effect", b[c] + cn_cell * f[c])
            eta = pm.Deterministic(
                "eta", z + a[s] + p[g, s] + gene_effect + cell_effect
            )
            mu = pm.Deterministic("mu", pmmath.exp(eta))

            alpha = pm.Gamma("alpha", 2.0, 0.5)
            y = pm.NegativeBinomial(  # noqa: F841
                "ct_final",
                mu * model_data.ct_initial,
                alpha,
                observed=model_data.ct_final,
            )
        return model


def _collect_all_cancer_genes(cancer_genes: LineageGeneMap) -> list[str]:
    gene_set: set[str] = set()
    for genes in cancer_genes.values():
        gene_set = gene_set.union(genes)
    gene_list = list(gene_set)
    gene_list.sort()
    return gene_list


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


def _make_cancer_gene_mutation_matrix(
    data: pd.DataFrame,
    cancer_genes: LineageGeneMap,
    cell_lines: list[str],
    genes: list[str],
) -> np.ndarray:
    lineages = (
        data[["depmap_id", "lineage"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .sort_values("depmap_id")
        .lineage.values
    )
    assert len(cell_lines) == len(lineages)
    cell_mutations = _collect_mutations_per_cell_line(data)
    mut_mat = np.zeros(shape=(len(genes), len(cell_lines)), dtype=int)
    for j, (cl, lineage) in enumerate(zip(cell_lines, lineages)):
        if lineage not in cancer_genes:
            logger.warn(f"Lineage {lineage} not found in cancer genes.")
            continue

        cell_muts = cell_mutations[cl].intersection(cancer_genes[lineage])
        if len(cell_muts) == 0:
            continue

        mut_ary = np.array([g in cell_muts for g in genes])
        mut_mat[:, j] = mut_ary

    return mut_mat


def _augmented_mutation_data(
    data: pd.DataFrame, cancer_genes: LineageGeneMap
) -> np.ndarray:
    mut = data["is_mutated"].values.astype(int)
    _empty_set: set[str] = set()
    for i, (gene, lineage) in enumerate(zip(data["hugo_symbol"], data["lineage"])):
        if gene in cancer_genes.get(lineage, _empty_set):
            mut[i] = 0
    return mut
