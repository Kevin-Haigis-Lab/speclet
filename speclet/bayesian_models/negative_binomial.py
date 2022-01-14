"""Simple negative binomial model."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3.math as pmmath
import stan
from pandera import Column, DataFrameSchema
from stan.model import Model as StanModel

from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
)
from speclet.data_processing.validation import check_finite, check_nonnegative
from speclet.io import stan_models_dir
from speclet.modeling.stan_helpers import read_code_file
from speclet.project_enums import ModelFitMethod


@dataclass
class NegativeBinomialModelData:
    """Data for `NegativeBinomialModel`."""

    N: int
    ct_initial: np.ndarray
    ct_final: np.ndarray


class NegativeBinomialModel:
    """Negative binomial generalized linear model."""

    _stan_code_file: Final[Path] = stan_models_dir() / "negative_binomial.stan"

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
            }
        )

    def vars_regex(self, fit_method: ModelFitMethod) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        return ["~reciprocal_phi", "~mu"]

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.data_schema.validate(data)

    def _make_data_structure(self, data: pd.DataFrame) -> NegativeBinomialModelData:
        return NegativeBinomialModelData(
            N=data.shape[0],
            ct_initial=data.counts_initial_adj.values.tolist(),
            ct_final=data.counts_final.values.tolist(),
        )

    def data_processing_pipeline(self, data: pd.DataFrame) -> NegativeBinomialModelData:
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
            .pipe(self._validate_data)
            .pipe(self._make_data_structure)
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
        model_data = self.data_processing_pipeline(data)
        return stan.build(
            self.stan_code, data=asdict(model_data), random_seed=random_seed
        )

    @property
    def stan_idata_addons(self) -> dict[str, Any]:
        """Information to add to the InferenceData posterior object."""
        return {
            "posterior_predictive": ["y_hat"],
            "observed_data": ["ct_final"],
            "log_likelihood": {"ct_final": "log_lik"},
            "constant_data": ["ct_initial"],
        }

    def pymc3_model(self, data: pd.DataFrame) -> pm.Model:
        """PyMC3  model for a simple negative binomial model.

        Not implemented.

        Args:
            data (pd.DataFrame): Data to model.

        Returns:
            pm.Model: PyMC3 model.
        """
        model_data = self.data_processing_pipeline(data)
        with pm.Model() as model:
            beta = pm.Normal("beta", 0, 5)
            eta = pm.Deterministic("eta", beta)
            mu = pm.Deterministic("mu", pmmath.exp(eta))
            alpha = pm.Gamma("alpha", 2, 0.3)
            y = pm.NegativeBinomial(  # noqa: F841
                "ct_final",
                mu * model_data.ct_initial,
                alpha,
                observed=model_data.ct_final,
            )
        return model
