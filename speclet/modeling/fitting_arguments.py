"""Concerning arguments for fitting Bayesian models."""

from typing import Any, Literal

from pydantic import BaseModel, PositiveInt, confloat

from speclet.types import BasicTypes, VIMethod

TargetAcceptFloat = confloat(ge=0.5, lt=1.0)


class PymcSampleArguments(BaseModel):
    """PyMC model `sample()` keyword arguments."""

    draws: int = 1000
    init: str = "auto"
    n_init: int = 200000
    chains: PositiveInt | None = None
    cores: PositiveInt | None = None
    tune: PositiveInt = 1000
    progressbar: bool = True
    target_accept: TargetAcceptFloat = 0.8  # type: ignore


class PymcSamplingNumpyroArguments(BaseModel):
    """PyMC model `sampling_jax.sample_numpyro_nuts()` keyword arguments."""

    draws: PositiveInt = 1000
    tune: PositiveInt = 1000
    chains: PositiveInt = 4
    target_accept: TargetAcceptFloat = 0.8  # type: ignore
    progress_bar: bool = True
    chain_method: Literal["sequential", "parallel", "vectorized"] = "parallel"
    postprocessing_backend: Literal["cpu", "gpu"] = "cpu"
    idata_kwargs: dict[str, Any] | None = None
    nuts_kwargs: dict[str, Any] | None = None


class PymcFitArguments(BaseModel):
    """PyMC model `fit()` keyword arguments."""

    n: PositiveInt = 10000
    method: VIMethod = "advi"
    draws: PositiveInt = 1000
    inf_kwargs: dict[str, BasicTypes] | None = None
    progressbar: bool = True


class ModelingSamplingArguments(BaseModel):
    """Keyword sampling arguments for methods of fitting a Bayesian models."""

    pymc_mcmc: PymcSampleArguments | None = None
    pymc_advi: PymcFitArguments | None = None
    pymc_numpyro: PymcSamplingNumpyroArguments | None = None
