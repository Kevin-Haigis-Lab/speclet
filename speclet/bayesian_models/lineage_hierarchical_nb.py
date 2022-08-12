"""A hierarchical negative binomial generalized linear model for a single lineage."""

from dataclasses import dataclass
from typing import Any

import aesara.tensor as at
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.math as pmmath
from pandera import Check, Column, DataFrameSchema
from pydantic import BaseModel

import speclet.modeling.posterior_checks as post_checks
from speclet.data_processing.common import get_cats
from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
    chromosome_indices,
    common_indices,
    data_batch_indices,
    grouped_copy_number_transform,
    make_chromosome_cellline_column,
    set_achilles_categorical_columns,
    set_chromosome_categories,
    zscale_rna_expression_by_gene,
)
from speclet.data_processing.validation import (
    check_chromosome_category_order,
    check_finite,
    check_nonnegative,
    check_single_unique_value,
    check_unique_groups,
)
from speclet.data_processing.vectors import squish_array
from speclet.loggers import logger
from speclet.managers.data_managers import CancerGeneDataManager as CancerGeneDM
from speclet.modeling.cancer_gene_mutation_matrix import (
    extract_mutation_matrix_and_cancer_genes,
    make_cancer_gene_mutation_matrix,
)
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
    CHROM: int  # number of chromosomes
    ct_initial: npt.NDArray[np.float64]
    ct_final: npt.NDArray[np.int64]
    sgrna_idx: npt.NDArray[np.int64]
    gene_idx: npt.NDArray[np.int64]
    sgrna_to_gene_idx: npt.NDArray[np.int64]
    sgrna_to_gene_map: pd.DataFrame
    cellline_chromosome_idx: npt.NDArray[np.int64]
    cellline_idx: npt.NDArray[np.int64]
    chromosome_to_cellline_idx: npt.NDArray[np.int64]
    chromosome_to_cellline_map: pd.DataFrame
    screen_idx: npt.NDArray[np.int64]
    copy_number: npt.NDArray[np.float64]
    copy_number_gene: npt.NDArray[np.float64]
    copy_number_cell: npt.NDArray[np.float64]
    log_rna_expr: npt.NDArray[np.float64]
    z_log_rna_gene: npt.NDArray[np.float64]
    m_log_rna_gene: npt.NDArray[np.float64]
    is_mutated: npt.NDArray[np.int64]
    comutation_matrix: npt.NDArray[np.int64]
    coords: dict[str, list[str]]


class LineageHierNegBinomModelConfig(BaseModel):
    """Single-lineage hierarchical negative binominal model configuration."""

    lineage: str
    lfc_limits: tuple[float, float] = (-5.0, 5.0)
    min_n_cancer_genes: int = 2
    min_frac_cancer_genes: float = 0.0
    top_n_cancer_genes: int | None = None
    reduce_deterministic_vars: bool = True


class LineageHierNegBinomModel:
    """A hierarchical negative binomial generalized linear model for one lineage."""

    version: str = "0.1.3"

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
        self.top_n_cancer_genes = self._config.top_n_cancer_genes
        if self.min_n_cancer_genes < 2:
            logger.warning(
                "Setting `min_n_cancer_genes` less than "
                + "2 may result is some non-identifiability."
            )
        self.reduce_deterministic_vars = self._config.reduce_deterministic_vars
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
                "depmap_id": Column(
                    "category",
                    checks=[
                        # A "cell line-chromosome" maps to a single cell line.
                        Check(check_unique_groups, groupby="cell_chrom"),
                    ],
                ),
                "lineage": Column(
                    "category", checks=[check_single_unique_value(self.lineage)]
                ),
                "copy_number": Column(
                    float, checks=[check_nonnegative(), check_finite()]
                ),
                "rna_expr": Column(
                    float, nullable=False, checks=[check_nonnegative(), check_finite()]
                ),
                "z_rna_gene": Column(float, checks=[check_finite()]),
                "cn_gene": Column(float, checks=[check_finite()]),
                "cn_cell_line": Column(float, checks=[check_finite()]),
                "is_mutated": Column(bool, nullable=False),
                "screen": Column("category", nullable=False),
                "sgrna_target_chr": Column(
                    "category",
                    nullable=False,
                    checks=[check_chromosome_category_order()],
                ),
                "cell_chrom": Column("category", nullable=False),
            }
        )

    def vars_regex(self, fit_method: ModelFitMethod | None = None) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        _vars = [
            "~^delta_.*",
            "~^.*cells.*$",
            "~^.*genes.*$",
        ]
        if not self.reduce_deterministic_vars:
            _vars += [
                "~^mu$",
                "~^eta$",
                "~.*effect$",
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
            "cell_chrom": get_cats(valid_data, "cell_chrom"),
            "one": ["1"],
        }

    def make_data_structure(self, data: pd.DataFrame) -> LineageHierNegBinomModelData:
        """Create the model data structure."""
        # Indices
        indices = common_indices(data)
        batch_indices = data_batch_indices(data)
        chrom_indices = chromosome_indices(
            data, cell_chrom_col="cell_chrom", cell_line_col="depmap_id"
        )

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
        cg_mutation_matrix = make_cancer_gene_mutation_matrix(
            data,
            cancer_genes,
            min_n=self.min_n_cancer_genes,
            min_freq=self.min_frac_cancer_genes,
            top_n_cg=self.top_n_cancer_genes,
        )
        if cg_mutation_matrix is None:
            comutation_matrix = np.array([0])
            cancer_genes = []
        else:
            comutation_matrix, cancer_genes = extract_mutation_matrix_and_cancer_genes(
                cg_mutation_matrix
            )
        coords["cancer_gene"] = cancer_genes

        genes_params = ["mu_a", "b", "d", "f"]
        genes_params += [f"h[{cg}]" for cg in cancer_genes]
        coords["genes_params"] = genes_params
        coords["genes_params_"] = genes_params.copy()

        cells_params = ["mu_k", "mu_m"]
        coords["cells_params"] = cells_params
        coords["cells_params_"] = cells_params.copy()

        return LineageHierNegBinomModelData(
            N=data.shape[0],
            S=indices.n_sgrnas,
            G=indices.n_genes,
            C=chrom_indices.n_celllines,
            SC=batch_indices.n_screens,
            CG=len(cancer_genes),
            CHROM=chrom_indices.n_chromosome_cell,
            ct_initial=data.counts_initial_adj.values.copy().astype(np.float64),
            ct_final=data.counts_final.values.astype(np.int64),
            sgrna_idx=indices.sgrna_idx.astype(np.int64),
            gene_idx=indices.gene_idx.astype(np.int64),
            sgrna_to_gene_idx=indices.sgrna_to_gene_idx.astype(np.int64),
            sgrna_to_gene_map=indices.sgrna_to_gene_map,
            cellline_idx=indices.cellline_idx.astype(np.int64),
            cellline_chromosome_idx=chrom_indices.cell_chrom_idx,
            chromosome_to_cellline_idx=chrom_indices.chrom_to_cell_idx,
            chromosome_to_cellline_map=chrom_indices.chrom_to_cell_map,
            screen_idx=batch_indices.screen_idx.astype(np.int64),
            copy_number=data.copy_number.values.astype(np.float64),
            copy_number_gene=data.cn_gene.values.astype(np.float64),
            copy_number_cell=data.cn_cell_line.values.astype(np.float64),
            log_rna_expr=data.rna_expr.values.astype(np.float64),
            z_log_rna_gene=data.z_rna_gene.values.astype(np.float64),
            m_log_rna_gene=data.m_rna_gene.values.astype(np.float64),
            is_mutated=is_mutated.astype(np.int64),
            comutation_matrix=comutation_matrix.astype(np.int64),
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
                grouped_copy_number_transform,
                group="hugo_symbol",
                cn_col="copy_number",
                new_col="cn_gene",
                max_cn=3,
            )
            .pipe(
                grouped_copy_number_transform,
                group="depmap_id",
                cn_col="copy_number",
                new_col="cn_cell_line",
                max_cn=3,
            )
            .assign(
                cn_gene=lambda d: squish_array(d["cn_gene"], lower=-5.0, upper=5.0),
                cn_cell_line=lambda d: squish_array(
                    d["cn_cell_line"], lower=-5.0, upper=5.0
                ),
            )
            .pipe(append_total_read_counts)
            .pipe(add_useful_read_count_columns)
            .pipe(set_achilles_categorical_columns, sort_cats=True, skip_if_cat=False)
            .pipe(set_chromosome_categories, col="sgrna_target_chr")
            .pipe(
                make_chromosome_cellline_column,
                chrom_col="sgrna_target_chr",
                cell_line_col="depmap_id",
                new_col="cell_chrom",
            )
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
        logger.info(f"Number of chromosomes: {model_data.CHROM}")
        logger.info(f"Number of cancer genes: {model_data.CG}")
        logger.info(f"Number of screens: {model_data.SC}")
        logger.info(f"Number of data points: {model_data.N}")

        if self.reduce_deterministic_vars:
            logger.info("Configured to reduce the number of deterministic variables.")
        else:
            logger.info("Including all non-essential deterministic variables.")

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
        cn_gene = model_data.copy_number_gene
        cn_cell = model_data.copy_number_cell
        mut = model_data.is_mutated
        cg_mut = model_data.comutation_matrix
        logger.debug(f"shape of cancer gene matrix: {cg_mut.shape}")
        # Indexing arrays.
        s_to_g = model_data.sgrna_to_gene_idx
        s = model_data.sgrna_idx
        g = model_data.gene_idx
        chrom_to_cell = model_data.chromosome_to_cellline_idx
        chrom = model_data.cellline_chromosome_idx
        # Sizes.
        n_G = model_data.G
        n_CG = model_data.CG
        n_C = model_data.C
        # Number of varying effects variables.
        n_gene_vars = 4 + n_CG
        g_sd_dists = [0.2, 0.1, 0.5, 0.2] + ([0.2] * n_CG)
        assert (
            len(g_sd_dists) == n_gene_vars
        ), "Varying gene effects SDs not correct length."
        n_cell_vars = 2
        c_sd_dists = [0.1, 0.2]
        assert (
            len(c_sd_dists) == n_cell_vars
        ), "Varying cell effects SDs not correct length."

        mu_mu_a_loc = np.log(
            np.mean(model_data.ct_final) / (np.mean(model_data.ct_initial))
        )
        logger.debug(f"location for `mu_mu_a`: {mu_mu_a_loc:0.4f}")

        with pm.Model(coords=coords) as model:
            # Gene varying effects covariance matrix.

            g_chol, _, g_sigmas = pm.LKJCholeskyCov(
                "genes_chol_cov",
                eta=2,
                n=n_gene_vars,
                sd_dist=pm.HalfNormal.dist(g_sd_dists, shape=n_gene_vars),
                compute_corr=True,
            )
            for i, var_name in enumerate(["mu_a", "b", "d", "f"]):
                pm.Deterministic(f"sigma_{var_name}", g_sigmas[i])
            pm.Deterministic("sigma_h", g_sigmas[4:], dims="cancer_gene")

            # Gene varying effects.
            mu_mu_a = pm.Normal("mu_mu_a", mu_mu_a_loc, 0.2)
            mu_b = pm.Normal("mu_b", 0, 0.1)
            mu_d = 0  # Must be zero if CN for cell lines is a variable.
            mu_f = 0
            mu_h = [0] * n_CG
            _mu_genes = [mu_mu_a, mu_b, mu_d, mu_f] + [mu_h[i] for i in range(n_CG)]
            mu_genes = at.stack(_mu_genes, axis=0)
            delta_genes = pm.Normal("delta_genes", 0, 1, shape=(n_gene_vars, n_G))
            genes = mu_genes + at.dot(g_chol, delta_genes).T
            mu_a = pm.Deterministic("mu_a", genes[:, 0], dims="gene")
            b = pm.Deterministic("b", genes[:, 1], dims="gene")
            d = pm.Deterministic("d", genes[:, 2], dims="gene")
            f = pm.Deterministic("f", genes[:, 3], dims="gene")

            sigma_a = pm.HalfNormal("sigma_a", 0.5)
            delta_a = pm.Normal("delta_a", 0, 1, dims="sgrna")
            a = pm.Deterministic("a", mu_a[s_to_g] + delta_a * sigma_a, dims="sgrna")

            _gene_effect = a[s] + b[g] * rna + d[g] * cn_gene + f[g] * mut

            if n_CG > 0:
                h = pm.Deterministic("h", genes[:, 4:], dims=("gene", "cancer_gene"))
                _gene_effect = _gene_effect + at.sum(h[g, :] * cg_mut, axis=1)
            else:
                logger.warning("No cancer genes -> no variable `h` in model.")

            if self.reduce_deterministic_vars:
                gene_effect = _gene_effect
            else:
                gene_effect = pm.Deterministic("gene_effect", _gene_effect)

            # Cell line varying effects covariance matrix.
            cl_chol, _, cl_sigmas = pm.LKJCholeskyCov(
                "cells_chol_cov",
                eta=2,
                n=n_cell_vars,
                sd_dist=pm.HalfNormal.dist(c_sd_dists, shape=n_cell_vars),
                compute_corr=True,
            )
            for i, var_name in enumerate(["mu_k", "mu_m"]):
                pm.Deterministic(f"sigma_{var_name}", cl_sigmas[i])

            mu_mu_k = 0
            mu_mu_m = pm.Normal("mu_mu_m", -0.2, 0.1)
            _mu_cells = [mu_mu_k, mu_mu_m]
            mu_cells = at.stack(_mu_cells, axis=0)
            delta_cells = pm.Normal("delta_cells", 0, 1, shape=(n_cell_vars, n_C))
            cells = mu_cells + at.dot(cl_chol, delta_cells).T
            mu_k = pm.Deterministic("mu_k", cells[:, 0], dims="cell_line")
            mu_m = pm.Deterministic("mu_m", cells[:, 1], dims="cell_line")

            sigma_k = pm.HalfNormal("sigma_k", 0.1)
            delta_k = pm.Normal("delta_k", 0, 1, dims="cell_chrom")
            k = pm.Deterministic(
                "k", mu_k[chrom_to_cell] + delta_k * sigma_k, dims="cell_chrom"
            )

            sigma_m = pm.HalfNormal("sigma_m", 0.1)
            delta_m = pm.Normal("delta_m", 0, 1, dims="cell_chrom")
            m = pm.Deterministic(
                "m", mu_m[chrom_to_cell] + delta_m * sigma_m, dims="cell_chrom"
            )

            _cell_effect = k[chrom] + m[chrom] * cn_cell
            if self.reduce_deterministic_vars:
                cell_effect = _cell_effect
            else:
                cell_effect = pm.Deterministic("cell_effect", _cell_effect)

            _eta = gene_effect + cell_effect + np.log(model_data.ct_initial)
            if self.reduce_deterministic_vars:
                eta = _eta
            else:
                eta = pm.Deterministic("eta", _eta)

            _mu = pmmath.exp(eta)
            if self.reduce_deterministic_vars:
                mu = _mu
            else:
                mu = pm.Deterministic("mu", _mu)

            alpha = pm.Exponential("alpha", 0.5)
            pm.NegativeBinomial(
                "ct_final",
                mu=mu,
                alpha=alpha,
                observed=model_data.ct_final,
            )
        return model

    def posterior_sample_checks(self) -> list[post_checks.PosteriorCheck]:
        """Default posterior checks."""
        checks: list[post_checks.PosteriorCheck] = []
        marginal_checks = [
            "sigma_mu_a",
            "sigma_b",
            "sigma_d",
            "sigma_f",
            "sigma_k",
            "sigma_mu_k",
            "sigma_mu_m",
        ]
        for var_name in marginal_checks:
            checks.append(
                post_checks.CheckMarginalPosterior(
                    var_name=var_name,
                    min_avg=0.0001,
                    max_avg=np.inf,
                    skip_if_missing=True,
                )
            )

        ess_checks = ["sigma_mu_a", "mu_mu_a"]
        for var_name in ess_checks:
            checks.append(
                post_checks.CheckEffectiveSampleSize(
                    var_name=var_name, min_frac_ess=0.05
                )
            )

        return checks

    def additional_variable_dims(self) -> dict[str, list[str]]:
        """Additional dimensions for variables not included in the model spec."""
        return {
            "cells_chol_cov_stds": ["cells_params"],
            "cells_chol_cov_corr": ["cells_params", "cells_params_"],
            "genes_chol_cov_stds": ["genes_params"],
            "genes_chol_cov_corr": ["genes_params", "genes_params_"],
        }


def target_gene_is_mutated_vector(data: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Create a target gene mutation vector.

    Accounts for an issue that can lead to non-identifiability where the target gene is
    mutated in all cell lines. This would be co-linear with the varying intercept.

    Args:
        data (pd.DataFrame): CRISPR data.

    Returns:
        npt.NDArray[np.int64]: Binary vector for if the target gene is mutated.
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
    mut_ary = data["is_mutated"].values.astype(np.int64)
    idx = np.asarray([g in always_mutated_genes for g in data["hugo_symbol"]])
    mut_ary[idx] = 0
    return mut_ary
