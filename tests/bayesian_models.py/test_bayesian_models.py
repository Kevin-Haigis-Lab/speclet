from pathlib import Path

import arviz as az
import pandas as pd
import pymc as pm
import pymc.sampling_jax
import pytest

from speclet import bayesian_models as bayes
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.project_enums import ModelFitMethod, assert_never


@pytest.fixture
def valid_crispr_data(depmap_test_data: Path) -> pd.DataFrame:
    return CrisprScreenDataManager(data_file=depmap_test_data).get_data()


def _modify_for_single_lineage(crispr_df: pd.DataFrame) -> pd.DataFrame:
    crispr_df["lineage"] = ["colorectal"] * len(crispr_df)
    crispr_df["lineage"] = pd.Categorical(crispr_df["lineage"])
    return crispr_df


@pytest.mark.slow
@pytest.mark.parametrize("bayesian_model", bayes.BayesianModel)
def test_all_bayesian_pymc_models_build(
    bayesian_model: bayes.BayesianModel, valid_crispr_data: pd.DataFrame
) -> None:
    model_obj = bayes.get_bayesian_model(bayesian_model)()
    if bayesian_model is bayes.BayesianModel.LINEAGE_HIERARCHICAL_NB:
        valid_crispr_data = _modify_for_single_lineage(valid_crispr_data)
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

    if bayesian_model is bayes.BayesianModel.LINEAGE_HIERARCHICAL_NB:
        valid_crispr_data = _modify_for_single_lineage(valid_crispr_data)

    if method is ModelFitMethod.PYMC_MCMC:
        with model_obj.pymc_model(data=valid_crispr_data):
            trace = pm.sample(
                draws=n_draws, tune=100, cores=1, chains=1, return_inferencedata=True
            )
    elif method is ModelFitMethod.PYMC_NUMPYRO:
        with model_obj.pymc_model(data=valid_crispr_data):
            trace = pymc.sampling_jax.sample_numpyro_nuts(
                draws=n_draws, tune=100, chains=1
            )
    elif method is ModelFitMethod.PYMC_ADVI:
        with model_obj.pymc_model(data=valid_crispr_data):
            approx = pm.fit(n=100)
            trace = pm.to_inference_data(trace=approx.sample(n_draws))
    else:
        assert_never(method)

    assert isinstance(trace, az.InferenceData)
    assert trace["posterior"].coords.dims["draw"] == n_draws
