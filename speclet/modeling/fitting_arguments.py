"""Concerning arguments for fitting Bayesian models."""

from typing import Callable, Iterable, Optional, Union

from pydantic import BaseModel, PositiveInt, confloat

from speclet.types import BasicTypes, VIMethod

TargetAcceptFloat = confloat(ge=0.5, lt=1.0)


class PymcSampleArguments(BaseModel):
    """PyMC model `sample()` keyword arguments."""

    draws: int = 1000
    init: str = "auto"
    step: Optional[Union[Callable, Iterable[Callable]]] = None
    n_init: int = 200000
    chain_idx: PositiveInt = 0
    chains: Optional[PositiveInt] = None
    cores: Optional[PositiveInt] = None
    tune: PositiveInt = 1000
    progressbar: bool = True
    discard_tuned_samples: bool = True
    compute_convergence_checks: bool = True
    return_inferencedata: bool = True
    idata_kwargs: Optional[dict[str, BasicTypes]] = None
    target_accept: TargetAcceptFloat = 0.8  # type: ignore


class PymcFitArguments(BaseModel):
    """PyMC model `fit()` keyword arguments."""

    n: PositiveInt = 10000
    method: VIMethod = "advi"
    draws: PositiveInt = 1000
    inf_kwargs: Optional[dict[str, BasicTypes]] = None
    progressbar: bool = True


class ModelingSamplingArguments(BaseModel):
    """Keyword sampling arguments for methods of fitting a Bayesian models."""

    pymc_mcmc: Optional[PymcSampleArguments] = None
    pymc_advi: Optional[PymcFitArguments] = None
