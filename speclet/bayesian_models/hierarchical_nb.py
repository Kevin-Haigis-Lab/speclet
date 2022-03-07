"""A hierarchical negative binomial generialzed linear model."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
import pymc as pm
import pymc.math as pmmath
import stan
from pandera import Check, Column, DataFrameSchema
from stan.model import Model as StanModel

from speclet.data_processing.common import get_cats
from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
    common_indices,
    set_achilles_categorical_columns,
    zscale_cna_by_group,
    zscale_rna_expression_by_gene_lineage,
)
from speclet.data_processing.validation import (
    check_finite,
    check_nonnegative,
    check_unique_groups,
)
from speclet.io import stan_models_dir
from speclet.modeling.stan_helpers import read_code_file
from speclet.project_enums import ModelFitMethod


@dataclass
class NegativeBinomialModelData:
    """Data for `NegativeBinomialModel`."""

    N: int  # total number of data points
    S: int  # number of sgRNAs
    G: int  # number of genes
    C: int  # number of cell lines
    L: int  # number of lineages
    ct_initial: np.ndarray
    ct_final: np.ndarray
    sgrna_idx: np.ndarray
    gene_idx: np.ndarray
    cellline_idx: np.ndarray
    lineage_idx: np.ndarray
    copy_number: np.ndarray
    copy_number_z_gene: np.ndarray
    copy_number_z_cell: np.ndarray
    rna_z_gene_lineage: np.ndarray
    is_mutated: np.ndarray


@dataclass
class HierarchicalNegativeBinomialConfig:
    """Configuration for `HierarchcalNegativeBinomialModel`."""

    num_knots: int = 5


class HierarchcalNegativeBinomialModel:
    """A hierarchical negative binomial generialzed linear model."""

    _config: HierarchicalNegativeBinomialConfig
    _stan_code_file: Final[Path] = stan_models_dir() / "hierarchical_nb.stan"

    def __init__(self) -> None:
        """Create a negative binomial Bayesian model object."""
        self._config = HierarchicalNegativeBinomialConfig()
        assert self._stan_code_file.exists(), "Cannot find Stan code."
        assert self._stan_code_file.is_file(), "Path to Stan code is not a file."
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
                "z_rna_gene_lineage": Column(float, checks=[check_finite()]),
                "z_cn_gene": Column(float, checks=[check_finite()]),
                "z_cn_cell_line": Column(float, checks=[check_finite()]),
                "is_mutated": Column(bool, nullable=False),
            }
        )

    def vars_regex(self, fit_method: ModelFitMethod) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        _vars = ["~^mu$", "~^eta$", "~^delta_.*"]
        if fit_method is ModelFitMethod.STAN_MCMC:
            _vars += ["~^log_lik$", "~^y_hat$"]
        else:
            _vars += []
        return _vars

    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data according to this model's requirements.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Validated data.
        """
        return self.data_schema.validate(data)

    def _make_data_structure(self, data: pd.DataFrame) -> NegativeBinomialModelData:
        indices = common_indices(data)
        return NegativeBinomialModelData(
            N=data.shape[0],
            S=indices.n_sgrnas,
            G=indices.n_genes,
            C=indices.n_celllines,
            L=indices.n_lineages,
            ct_initial=data.counts_initial_adj.values.astype(float),
            ct_final=data.counts_final.values.astype(int),
            sgrna_idx=indices.sgrna_idx,
            gene_idx=indices.gene_idx,
            cellline_idx=indices.cellline_idx,
            lineage_idx=indices.lineage_idx,
            copy_number=data.copy_number.values,
            copy_number_z_gene=data.z_cn_gene.values,
            copy_number_z_cell=data.z_cn_cell_line.values,
            rna_z_gene_lineage=data.z_rna_gene_lineage.values,
            is_mutated=data.is_mutated.values,
        )

    def data_processing_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data processing pipeline.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            NegativeBinomialModelData: Processed and validated modeling data.
        """
        return (
            data.dropna(
                axis=0,
                how="any",
                subset=["counts_final", "counts_initial", "copy_number"],
            )
            .pipe(zscale_rna_expression_by_gene_lineage, new_col="z_rna_gene_lineage")
            .pipe(
                zscale_cna_by_group, groupby_cols=["hugo_symbol"], new_col="z_cn_gene"
            )
            .pipe(
                zscale_cna_by_group,
                groupby_cols=["depmap_id"],
                new_col="z_cn_cell_line",
            )
            .pipe(append_total_read_counts)
            .pipe(add_useful_read_count_columns)
            .pipe(set_achilles_categorical_columns)
            .pipe(self.validate_data)
        )

    @property
    def stan_code(self) -> str:
        """Stan code for the Negative Binomial model."""
        return read_code_file(self._stan_code_file)

    def stan_model(
        self, data: pd.DataFrame, random_seed: Optional[int] = None
    ) -> StanModel:
        """Stan model for a simple negative binomial model.

        Args:
            data (pd.DataFrame): Data to model.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            StanModel: Stan model.
        """
        model_data = self.data_processing_pipeline(data).pipe(self._make_data_structure)
        model_data.sgrna_idx = model_data.sgrna_idx + 1
        model_data.gene_idx = model_data.gene_idx + 1
        model_data.lineage_idx = model_data.lineage_idx + 1
        model_data.cellline_idx = model_data.cellline_idx + 1
        return stan.build(
            self.stan_code, data=asdict(model_data), random_seed=random_seed
        )

    def _model_coords(self, valid_data: pd.DataFrame) -> dict[str, list[str]]:
        return {
            "sgrna": get_cats(valid_data, "sgrna"),
            "gene": get_cats(valid_data, "hugo_symbol"),
            "cell_line": get_cats(valid_data, "depmap_id"),
            "lineage": get_cats(valid_data, "lineage"),
        }

    def stan_idata_addons(self, data: pd.DataFrame) -> dict[str, Any]:
        """Information to add to the InferenceData posterior object."""
        valid_data = self.data_processing_pipeline(data)
        return {
            "posterior_predictive": ["y_hat"],
            "observed_data": ["ct_final"],
            "log_likelihood": {"ct_final": "log_lik"},
            "constant_data": ["ct_initial"],
            "coords": self._model_coords(valid_data),
            "dims": {
                "a": ["sgrna"],
                "b": ["cell_line"],
                "d": ["gene", "lineage"],
                "alpha": ["gene"],
            },
        }

    def pymc_model(self, data: pd.DataFrame) -> pm.Model:
        """Hierarchical negative binomial model in PyMC.

        Args:
            data (pd.DataFrame): Data to model.

        Returns:
            pm.Model: PyMC model.
        """
        valid_data = self.data_processing_pipeline(data)
        model_data = self._make_data_structure(valid_data)
        coords = self._model_coords(valid_data)

        s = model_data.sgrna_idx
        c = model_data.cellline_idx
        g = model_data.gene_idx
        ll = model_data.lineage_idx

        cn_gene = model_data.copy_number_z_gene
        cn_cell = model_data.copy_number_z_cell
        rna = model_data.rna_z_gene_lineage
        mut = model_data.is_mutated

        with pm.Model(coords=coords) as model:
            z = pm.Normal("z", 0, 5)

            sigma_a = pm.HalfNormal("sigma_a", 2.5)
            a = pm.Normal("a", 0, sigma_a, dims=("sgrna"))

            sigma_b = pm.HalfNormal("sigma_b", 2.5)
            delta_b = pm.Normal("delta_b", 0, 1, dims=("cell_line"))
            b = pm.Deterministic("b", 0 + delta_b * sigma_b, dims=("cell_line"))

            sigma_d = pm.HalfNormal("sigma_d", 2.5)
            delta_d = pm.Normal("delta_d", 0, 1, dims=("gene", "lineage"))
            d = pm.Deterministic("d", 0 + delta_d * sigma_d, dims=("gene", "lineage"))

            mu_f = pm.Normal("mu_f", -1, 1)
            sigma_f = pm.HalfNormal("sigma_f", 2.5)
            delta_f = pm.Normal("delta_f", 0, 1, dims=("cell_line"))
            f = pm.Deterministic("f", mu_f + delta_f * sigma_f, dims=("cell_line"))

            mu_h = pm.Normal("mu_h", -1, 1)
            sigma_h = pm.HalfNormal("sigma_h", 2.5)
            delta_h = pm.Normal("delta_h", 0, 1, dims=("gene"))
            h = pm.Deterministic("h", mu_h + delta_h * sigma_h, dims=("gene"))

            mu_k = pm.Normal("mu_k", -1, 1)
            sigma_k = pm.HalfNormal("sigma_k", 2.5)
            delta_k = pm.Normal("delta_k", 0, 1, dims=("gene", "lineage"))
            k = pm.Deterministic(
                "k", mu_k + delta_k * sigma_k, dims=("gene", "lineage")
            )

            mu_m = pm.Normal("mu_m", -1, 1)
            sigma_m = pm.HalfNormal("sigma_m", 2.5)
            delta_m = pm.Normal("delta_m", 0, 1, dims=("gene", "lineage"))
            m = pm.Deterministic(
                "m", mu_m + delta_m * sigma_m, dims=("gene", "lineage")
            )

            eta = pm.Deterministic(
                "eta",
                z
                + a[s]
                + b[c]
                + d[g, ll]
                + cn_cell * f[c]
                + cn_gene * h[g]
                + rna * k[g, ll]
                + mut * m[g, ll],
            )
            mu = pm.Deterministic("mu", pmmath.exp(eta))

            # alpha_hyperparams = pm.Gamma("alpha_hyperparams", 2, 0.5, shape=2)
            # alpha = pm.Gamma(
            #     "alpha", alpha_hyperparams[0], alpha_hyperparams[1], dims=("gene")
            # )
            alpha = pm.Gamma("alpha", 2.0, 0.5)
            y = pm.NegativeBinomial(  # noqa: F841
                "ct_final",
                mu * model_data.ct_initial,
                alpha,
                observed=model_data.ct_final,
            )
        return model
