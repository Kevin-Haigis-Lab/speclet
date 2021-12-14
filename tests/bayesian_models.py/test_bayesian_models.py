from pathlib import Path

import pandas as pd
import pymc3 as pm
import pytest
import stan
from stan.model import Model as StanModel

from speclet import bayesian_models as bayes
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.project_enums import ModelFitMethod


@pytest.fixture
def valid_crispr_data(depmap_test_data: Path) -> pd.DataFrame:
    return CrisprScreenDataManager(data_file=depmap_test_data).get_data()


_schools_code = """
data {
  int<lower=0> J;         // number of schools
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""


def test_pytsan_working() -> None:
    schools_data = {
        "J": 8,
        "y": [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    }
    posterior = stan.build(_schools_code, data=schools_data, random_seed=1)
    assert isinstance(posterior, StanModel)
    fit = posterior.sample(num_chains=1, num_warmup=100, num_samples=100)
    for param in ["tau", "eta", "mu"]:
        assert param in fit.keys()


@pytest.mark.parametrize("bayesian_model", bayes.BayesianModel)
@pytest.mark.parametrize(
    "method", [ModelFitMethod.PYMC3_MCMC, ModelFitMethod.STAN_MCMC]
)
def test_all_bayesian_models(
    bayesian_model: bayes.BayesianModel,
    method: ModelFitMethod,
    valid_crispr_data: pd.DataFrame,
) -> None:
    model_obj = bayes.get_bayesian_model(bayesian_model)()
    if method is ModelFitMethod.PYMC3_MCMC:
        pymc3_model = model_obj.pymc3_model(data=valid_crispr_data)
        assert isinstance(pymc3_model, pm.Model)
    elif method is ModelFitMethod.STAN_MCMC:
        stan_model = model_obj.stan_model(data=valid_crispr_data)
        assert isinstance(stan_model, StanModel)
