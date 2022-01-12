"""Classic eight-schools example."""

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import pymc3 as pm
import stan
from stan.model import Model as StanModel

_eight_schools_stan_code = """
data {
    int<lower=0> J;
    real y[J];
    real<lower=0> sigma[J];
}

parameters {
    real mu;
    real<lower=0> tau;
    real theta_tilde[J];
}

transformed parameters {
    real theta[J];

    for (j in 1:J) {
        theta[j] = mu + tau * theta_tilde[j];
    }
}

model {
    mu ~ normal(0, 5);
    tau ~ cauchy(0, 5);
    theta_tilde ~ normal(0, 1);
    y ~ normal(theta, sigma);
}

generated quantities {
    vector[J] log_lik;
    vector[J] y_hat;

    for (j in 1:J) {
        log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
        y_hat[j] = normal_rng(theta[j], sigma[j]);
    }
}
"""


@dataclass(frozen=True)
class SchoolsData:
    """Data structure for the eight-schools data."""

    J: int
    y: list[float]
    sigma: list[int]


class EightSchoolsModel:
    """Classic eight-schools example model."""

    def __init__(self) -> None:
        """Classic eight-schools example model."""
        return None

    @property
    def schools_data(self) -> SchoolsData:
        """Eight-schools data."""
        return SchoolsData(
            J=8, y=[28, 8, -3, 7, -1, 1, 18, 12], sigma=[15, 10, 16, 11, 9, 11, 10, 18]
        )

    def stan_model(
        self, data: pd.DataFrame, random_seed: Optional[int] = None
    ) -> StanModel:
        """Stan model for a the eight-schools example.

        Args:
            data (pd.DataFrame): Ignored.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            StanModel: Stan model.
        """
        return stan.build(
            _eight_schools_stan_code,
            data=asdict(self.schools_data),
            random_seed=random_seed,
        )

    @property
    def stan_idata_addons(self) -> dict[str, Any]:
        """Information to add to the InferenceData posterior object."""
        return {
            "posterior_predictive": "y_hat",
            "observed_data": ["y"],
            "log_likelihood": {"y": "log_lik"},
            "coords": {"school": np.arange(self.schools_data.J)},
            "dims": {
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta_tilde": ["school"],
            },
        }

    def pymc3_model(self, data: pd.DataFrame) -> pm.Model:
        """PyMC3  model for a simple negative binomial model.

        Not implemented.

        Args:
            data (pd.DataFrame): Ignored.

        Returns:
            pm.Model: PyMC3 model.
        """
        school_data = self.schools_data
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 5)
            tau = pm.HalfCauchy("tau", 5)
            theta_tilde = pm.Normal("eta", 0, 1)
            theta = pm.Deterministic("theta", mu + tau * theta_tilde)
            y = pm.Normal(  # noqa: F841
                name="y",
                mu=theta,
                sigma=np.array(school_data.sigma),
                observed=school_data.y,
            )
        return model
