from typing import Optional

import arviz as az
import pandas as pd
import pymc as pm
import pytest
from stan.model import Model as StanModel

from speclet.bayesian_models.eight_schools import EightSchoolsModel
from speclet.modeling import model_fitting_api as mf_api
from speclet.modeling.fitting_arguments import (
    ModelingSamplingArguments,
    PymcFitArguments,
    PymcSampleArguments,
    StanMCMCSamplingArguments,
)
from speclet.project_enums import ModelFitMethod


def test_fit_stan_mcmc(centered_eight_stan_model: StanModel) -> None:
    post = mf_api.fit_stan_mcmc(centered_eight_stan_model)
    assert isinstance(post, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize("n_prior_pred", [None, 100])
def test_fit_pymc_mcmc(
    centered_eight_pymc_model: pm.Model, n_prior_pred: Optional[int]
) -> None:
    args = PymcSampleArguments(**{"chains": 1, "cores": 1})
    post = mf_api.fit_pymc_mcmc(
        centered_eight_pymc_model,
        prior_pred_samples=n_prior_pred,
        sampling_kwargs=args,
    )
    assert isinstance(post, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize("n_prior_pred", [None, 100])
def test_fit_pymc_vi(
    centered_eight_pymc_model: pm.Model, n_prior_pred: Optional[int]
) -> None:
    args = PymcFitArguments(**{"n": 1000, "draws": 100})
    post = mf_api.fit_pymc_vi(
        centered_eight_pymc_model,
        prior_pred_samples=n_prior_pred,
        fit_kwargs=args,
    )
    assert isinstance(post.inference_data, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_fit_model_dispatcher(fit_method: ModelFitMethod) -> None:
    args = ModelingSamplingArguments(
        stan_mcmc=StanMCMCSamplingArguments(num_chains=1),
        pymc_mcmc=PymcSampleArguments(chains=1),
        pymc_advi=PymcFitArguments(n=100),
    )
    post = mf_api.fit_model(
        EightSchoolsModel(),
        data=pd.DataFrame(),
        fit_method=fit_method,
        sampling_kwargs=args,
    )
    assert isinstance(post, az.InferenceData)
