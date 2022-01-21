"""A hierarchical negative binomial model with a second tier."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3.math as pmmath
import stan
from pandera import Check, Column, DataFrameSchema
from stan.model import Model as StanModel

from speclet.data_processing.common import get_cats
from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
    common_indices,
    set_achilles_categorical_columns,
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
    ct_initial: np.ndarray
    ct_final: np.ndarray
    sgrna_idx: np.ndarray
    sgrna_to_gene_idx: np.ndarray
    cellline_idx: np.ndarray


class HierarchcalNegativeBinomialSecondTier:
    """A hierarchical negative binomial model with a second tier."""

    _stan_code_file: Final[Path] = stan_models_dir() / "hierarchical_nb_secondtier.stan"

    def __init__(self) -> None:
        """Create a negative binomial Bayesian model object."""
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
            }
        )

    def vars_regex(self, fit_method: ModelFitMethod) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        _vars = ["~^mu$", "~^eta$"]
        if fit_method in {ModelFitMethod.PYMC3_ADVI, ModelFitMethod.PYMC3_MCMC}:
            _vars += ["delta_gamma", "delta_beta"]
        elif fit_method is ModelFitMethod.STAN_MCMC:
            _vars += ["~log_lik", "~y_hat"]
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
            ct_initial=data.counts_initial_adj.values.astype(float),
            ct_final=data.counts_final.values.astype(int),
            sgrna_idx=indices.sgrna_idx,
            sgrna_to_gene_idx=indices.sgrna_to_gene_idx,
            cellline_idx=indices.cellline_idx,
        )

    def data_processing_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data processing pipeline.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            NegativeBinomialModelData: Processed and validated modeling data.
        """
        return (
            data.dropna(axis=0, how="any", subset=["counts_final", "counts_initial"])
            .pipe(append_total_read_counts)
            .pipe(add_useful_read_count_columns)
            .pipe(set_achilles_categorical_columns)
            .pipe(self.validate_data)
        )

    @property
    def stan_code(self) -> str:
        """Stan code for the model."""
        raise NotImplementedError("Stan model not yet implemented.")
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
        model_data.sgrna_to_gene_idx = model_data.sgrna_to_gene_idx + 1
        model_data.cellline_idx = model_data.cellline_idx + 1
        return stan.build(
            self.stan_code, data=asdict(model_data), random_seed=random_seed
        )

    def _model_coords(self, valid_data: pd.DataFrame) -> dict[str, list[str]]:
        return {
            "sgrna": get_cats(valid_data, "sgrna"),
            "gene": get_cats(valid_data, "hugo_symbol"),
            "cell_line": get_cats(valid_data, "depmap_id"),
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
                "beta_s": ["sgrna", "cell_line"],
                "mu_beta": ["gene"],
            },
        }

    def pymc3_model(self, data: pd.DataFrame) -> pm.Model:
        """PyMC3 model.

        Args:
            data (pd.DataFrame): Data to model.

        Returns:
            pm.Model: PyMC3 model.
        """
        valid_data = self.data_processing_pipeline(data)
        model_data = self._make_data_structure(valid_data)
        coords = self._model_coords(valid_data)
        coords["one"] = ["1"]

        with pm.Model(coords=coords) as model:

            mu_a = pm.Normal("mu_a", 0, 5)
            sigma_a = pm.Gamma("sigma_a", 2.0, 0.5)
            sigma_b = pm.Gamma("sigma_b", 2.0, 0.5)
            sigma_gamma = pm.Gamma("sigma_gamma", 2.0, 0.5)

            a_g = pm.Normal("a_g", mu_a, sigma_a, dims=("gene", "one"))
            b_c = pm.Normal("b_c", 0, sigma_b, dims=("one", "cell_line"))
            delta_gamma = pm.Normal("delta_gamma", 0, 1, dims=("gene", "cell_line"))
            gamma_gc = pm.Deterministic(
                "gamma_gc", 0 + delta_gamma * sigma_gamma, dims=("gene", "cell_line")
            )

            mu_beta = pm.Deterministic(
                "mu_beta", gamma_gc + a_g + b_c, dims=("gene", "cell_line")
            )
            sigma_beta = pm.Gamma("sigma_beta", 2.0, 0.5)
            delta_beta = pm.Normal("delta_beta", 0, 1, dims=("sgrna", "cell_line"))
            beta_s = pm.Deterministic(
                "beta_s",
                mu_beta[model_data.sgrna_to_gene_idx, :] + delta_beta * sigma_beta,
                dims=("sgrna", "cell_line"),
            )

            eta = pm.Deterministic(
                "eta", beta_s[model_data.sgrna_idx, model_data.cellline_idx]
            )
            mu = pm.Deterministic("mu", pmmath.exp(eta))
            alpha = pm.Gamma("alpha", 2, 0.3)
            y = pm.NegativeBinomial(  # noqa: F841
                "ct_final",
                mu * model_data.ct_initial,
                alpha,
                observed=model_data.ct_final,
            )
        return model
