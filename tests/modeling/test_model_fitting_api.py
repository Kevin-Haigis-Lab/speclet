from typing import Optional

import arviz as az
import pymc3 as pm
import pytest
from stan.model import Model as StanModel

from speclet.modeling import model_fitting_api as mf_api
from speclet.modeling.fitting_arguments import Pymc3FitArguments, Pymc3SampleArguments


def test_fit_stan_mcmc(centered_eight_stan_model: StanModel) -> None:
    post = mf_api.fit_stan_mcmc(centered_eight_stan_model)
    assert isinstance(post, az.InferenceData)


@pytest.mark.parametrize("n_prior_pred", [None, 100])
def test_fit_pymc3_mcmc(
    centered_eight_pymc3_model: pm.Model, n_prior_pred: Optional[int]
) -> None:
    args = Pymc3SampleArguments(**{"chains": 1, "cores": 1})
    post = mf_api.fit_pymc3_mcmc(
        centered_eight_pymc3_model,
        prior_pred_samples=n_prior_pred,
        sampling_kwargs=args,
    )
    assert isinstance(post, az.InferenceData)


@pytest.mark.parametrize("n_prior_pred", [None, 100])
def test_fit_pymc3_vi(
    centered_eight_pymc3_model: pm.Model, n_prior_pred: Optional[int]
) -> None:
    args = Pymc3FitArguments(**{"n": 1000, "draws": 100})
    post = mf_api.fit_pymc3_vi(
        centered_eight_pymc3_model,
        prior_pred_samples=n_prior_pred,
        fit_kwargs=args,
    )
    assert isinstance(post.inference_data, az.InferenceData)


# TODO: test `fit_model` function.
