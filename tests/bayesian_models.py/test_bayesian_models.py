from pathlib import Path

import arviz as az
import pandas as pd
import pymc as pm
import pytest
import stan
from stan.model import Model as StanModel

from speclet import bayesian_models as bayes
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.project_enums import ModelFitMethod, assert_never


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
    fit = posterior.sample(num_chains=1, num_warmup=1000, num_samples=1000)
    for param in ["tau", "eta", "mu"]:
        assert param in fit.keys()


@pytest.mark.slow
# @pytest.mark.parametrize("bayesian_model", bayes.BayesianModel)
# @pytest.mark.parametrize("method", [ModelFitMethod.PYMC_MCMC, ModelFitMethod.STAN_MCMC])
@pytest.mark.parametrize("bayesian_model", [bayes.BayesianModel.HIERARCHICAL_NB])
@pytest.mark.parametrize("method", [ModelFitMethod.PYMC_MCMC])
def test_all_bayesian_models_build(
    bayesian_model: bayes.BayesianModel,
    method: ModelFitMethod,
    valid_crispr_data: pd.DataFrame,
) -> None:
    model_obj = bayes.get_bayesian_model(bayesian_model)()
    if method is ModelFitMethod.PYMC_MCMC:
        pymc_model = model_obj.pymc_model(data=valid_crispr_data)
        assert isinstance(pymc_model, pm.Model)
    elif method is ModelFitMethod.STAN_MCMC:
        stan_model = model_obj.stan_model(data=valid_crispr_data)
        assert isinstance(stan_model, StanModel)


@pytest.mark.slow
# @pytest.mark.parametrize("bayesian_model", bayes.BayesianModel)
# @pytest.mark.parametrize("method", [ModelFitMethod.PYMC_MCMC, ModelFitMethod.STAN_MCMC])
@pytest.mark.parametrize("bayesian_model", [bayes.BayesianModel.HIERARCHICAL_NB])
@pytest.mark.parametrize("method", [ModelFitMethod.PYMC_MCMC])
def test_all_bayesian_models_sample(
    bayesian_model: bayes.BayesianModel,
    method: ModelFitMethod,
    valid_crispr_data: pd.DataFrame,
) -> None:
    model_obj = bayes.get_bayesian_model(bayesian_model)()
    n_draws = 103
    if method is ModelFitMethod.STAN_MCMC:
        stan_model = model_obj.stan_model(data=valid_crispr_data)
        trace = az.from_pystan(
            stan_model.sample(num_chains=1, num_samples=n_draws),
            **model_obj.stan_idata_addons(data=valid_crispr_data)
        )
    elif method is ModelFitMethod.PYMC_MCMC:
        with model_obj.pymc_model(data=valid_crispr_data):
            trace = pm.sample(
                draws=n_draws, tune=100, cores=1, chains=1, return_inferencedata=True
            )
    elif method is ModelFitMethod.PYMC_ADVI:
        with model_obj.pymc_model(data=valid_crispr_data):
            approx = pm.fit(n=100)
            trace = az.from_pymc3(approx.sample(n_draws))
    else:
        assert_never(method)

    assert isinstance(trace, az.InferenceData)
    assert trace["posterior"].coords.dims["draw"] == n_draws
