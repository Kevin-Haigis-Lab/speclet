"""Simple negative binomial model."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.math as pmmath
from pandera import Column, DataFrameSchema

from speclet.data_processing.crispr import (
    add_useful_read_count_columns,
    append_total_read_counts,
)
from speclet.data_processing.validation import check_finite, check_nonnegative
from speclet.project_enums import ModelFitMethod


@dataclass
class NegBinomModelData:
    """Data for `NegativeBinomialModel`."""

    N: int
    ct_initial: np.ndarray
    ct_final: np.ndarray


class NegativeBinomialModel:
    """Negative binomial generalized linear model."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a negative binomial Bayesian model object."""
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

    def vars_regex(self, fit_method: ModelFitMethod | None = None) -> list[str]:
        """Regular expression to help with plotting only interesting variables."""
        _vars = ["~^mu$"]
        return _vars

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.data_schema.validate(data)

    def _make_data_structure(self, data: pd.DataFrame) -> NegBinomModelData:
        return NegBinomModelData(
            N=data.shape[0],
            ct_initial=data.counts_initial_adj.values.tolist(),
            ct_final=data.counts_final.values.tolist(),
        )

    def data_processing_pipeline(self, data: pd.DataFrame) -> NegBinomModelData:
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

    def pymc_model(
        self,
        data: pd.DataFrame,
        skip_data_processing: bool = False,
    ) -> pm.Model:
        """Simple negative binomial model in PyMC.

        Args:
            data (pd.DataFrame): Data to model.
            skip_data_processing (bool, optional). Skip data pre-processing step?
            Defaults to `False`.

        Returns:
            pm.Model: PyMC model.
        """
        model_data: NegBinomModelData
        if not skip_data_processing:
            model_data = self.data_processing_pipeline(data)
        else:
            model_data = self._make_data_structure(data)

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
