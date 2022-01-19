"""Concerning arguments for fitting Bayesian models."""

from typing import Callable, Iterable, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, PositiveInt, confloat

from speclet.types import BasicTypes, VIMethod

TargetAcceptFloat = confloat(ge=0.5, lt=1.0)


def _randint() -> int:
    return np.random.randint(0, 10000)


class StanMCMCSamplingArguments(BaseModel):
    """MCMC sampling arguments for Pystan."""

    num_chains: PositiveInt = 4
    num_samples: PositiveInt = 1000
    num_warmup: PositiveInt = 1000
    save_warmup: bool = False
    num_thin: int = 1
    delta: TargetAcceptFloat = 0.8  # type: ignore
    max_depth: PositiveInt = 10
    random_seed: PositiveInt = Field(default_factory=_randint)
    refresh: PositiveInt = 100


class Pymc3SampleArguments(BaseModel):
    """PyMC3 model `sample()` keyword arguments."""

    draws: int = 1000
    step: Optional[Union[Callable, Iterable[Callable]]] = None
    init: str = "auto"
    n_init: int = 200000
    chain_idx: PositiveInt = 0
    chains: Optional[PositiveInt] = None
    cores: Optional[PositiveInt] = None
    tune: PositiveInt = 1000
    progressbar: bool = True
    random_seed: PositiveInt = Field(default_factory=_randint)
    discard_tuned_samples: bool = True
    compute_convergence_checks: bool = True
    return_inferencedata: bool = True  # not default
    idata_kwargs: Optional[dict[str, BasicTypes]] = None
    target_accept: TargetAcceptFloat = 0.8  # type: ignore


class Pymc3FitArguments(BaseModel):
    """PyMC3 model `fit()` keyword arguments."""

    n: PositiveInt = 10000
    method: VIMethod = "advi"
    draws: PositiveInt = 1000
    random_seed: PositiveInt = Field(default_factory=_randint)
    inf_kwargs: Optional[dict[str, BasicTypes]] = None
    progressbar: bool = True


class ModelingSamplingArguments(BaseModel):
    """Keyword sampling arguments for methods of fitting a Bayesian models."""

    stan_mcmc: Optional[StanMCMCSamplingArguments] = None
    pymc3_mcmc: Optional[Pymc3SampleArguments] = None
    pymc3_advi: Optional[Pymc3FitArguments] = None
