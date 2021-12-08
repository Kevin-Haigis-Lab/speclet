"""Simple negative binomial model."""
from pathlib import Path
from typing import Any, Final

import pandas as pd
import stan
from pandera import Column, DataFrameSchema
from stan.model import Model as StanModel

from speclet.data_processing.validation import check_finite, check_positive
from speclet.io import stan_models_dir


class NegativeBinomialModel:

    _stan_code: Final[Path] = stan_models_dir() / "negative_binomial.stan"

    def __init__(self) -> None:
        assert self._stan_code.exists(), "Cannot find Stan code."
        assert self._stan_code.is_file(), "Path to Stan code is not a file."
        return None

    @property
    def data_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "counts_initial_adj": Column(
                    float, checks=[check_positive(), check_finite()]
                ),
                "counts_final": Column(int, checks=[check_positive(), check_finite()]),
            }
        )

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.data_schema.validate(data)

    def _make_data_structure(self, data: pd.DataFrame) -> dict[str, Any]:
        return {
            "N": data.shape[0],
            "ct_initial": data.counts_initial_adj.values,
            "ct_final": data.counts_final.values,
        }

    def stan_mcmc(
        self,
        data: pd.DataFrame,
    ) -> StanModel:
        stan_data = self._validate_data(data).pipe(self._make_data_structure)
        return stan.build(str(self._stan_code), data=stan_data)

    def __call__(self, data: pd.DataFrame) -> StanModel:
        return self.stan_mcmc(data=data)
