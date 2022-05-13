from pathlib import Path

import arviz as az
import pandas as pd
import pymc as pm
import pytest

from speclet import bayesian_models as bayes
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.project_enums import ModelFitMethod, assert_never


@pytest.fixture
def valid_crispr_data(depmap_test_data: Path) -> pd.DataFrame:
    return CrisprScreenDataManager(data_file=depmap_test_data).get_data()


@pytest.mark.slow
@pytest.mark.parametrize("bayesian_model", bayes.BayesianModel)
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
    return None


@pytest.mark.slow
@pytest.mark.parametrize("bayesian_model", bayes.BayesianModel)
@pytest.mark.parametrize("method", [ModelFitMethod.PYMC_MCMC])
def test_all_bayesian_models_sample(
    bayesian_model: bayes.BayesianModel,
    method: ModelFitMethod,
    valid_crispr_data: pd.DataFrame,
) -> None:
    model_obj = bayes.get_bayesian_model(bayesian_model)()
    n_draws = 103
    if method is ModelFitMethod.PYMC_MCMC:
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
