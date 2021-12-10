"""Concerning arguments for fitting Bayesian models."""

from typing import Any, Callable, Iterable, Optional, Union

import pymc3 as pm
from pydantic import BaseModel, PositiveInt, confloat

from speclet.types import BasicTypes, VIMethod

TargetAcceptFloat = confloat(ge=0.5, lt=1.0)


class StanMCMCSamplingArguments(BaseModel):
    """MCMC sampling arguments for Stan."""  # TODO: this is incomplete...

    num_chains: PositiveInt = 4
    num_samples: PositiveInt = 1000


class Pymc3SampleArguments(BaseModel):
    """Model `sample()` keyword arguments (PyMC3 v3.11.2)."""

    _pymc3_version: str = "3.11.2"
    draws: int = 1000
    step: Optional[Union[Callable, Iterable[Callable]]] = None
    init: str = "auto"
    n_init: int = 200000
    chain_idx: PositiveInt = 0
    chains: Optional[PositiveInt] = None
    cores: Optional[PositiveInt] = None
    tune: PositiveInt = 1000
    progressbar: bool = True
    random_seed: Optional[PositiveInt] = None
    discard_tuned_samples: bool = True
    compute_convergence_checks: bool = True
    return_inferencedata: Optional[bool] = True  # not default
    idata_kwargs: Optional[dict[str, BasicTypes]] = None
    target_accept: TargetAcceptFloat = 0.8  # type: ignore

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3SampleArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        if self._pymc3_version != pm.__version__:
            raise NotImplementedError(
                f"Support for {self._pymc3_version} -> current ver {pm.__version__}"
            )


class Pymc3FitArguments(BaseModel):
    """Model `fit()` keyword arguments (PyMC3 v3.11.2)."""

    _pymc3_version: str = "3.11.2"
    n: PositiveInt = 10000
    method: VIMethod = "advi"
    random_seed: Optional[PositiveInt] = None
    inf_kwargs: Optional[dict[str, BasicTypes]] = None
    progressbar: bool = True

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3FitArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        if self._pymc3_version != pm.__version__:
            raise NotImplementedError(
                f"Support for {self._pymc3_version} -> current ver {pm.__version__}"
            )


class ModelingSamplingArguments(BaseModel):
    """Keyword sampling arguments for methods of fitting a Bayesian models."""

    stan_mcmc: Optional[StanMCMCSamplingArguments] = None
    pymc3_mcmc: Optional[Pymc3SampleArguments] = None
    pymc3_advi: Optional[Pymc3FitArguments] = None
