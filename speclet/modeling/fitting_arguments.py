"""Concerning arguments for fitting Bayesian models."""

from importlib.metadata import version
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, PositiveInt, confloat

from speclet.types import BasicTypes, VIMethod

TargetAcceptFloat = confloat(ge=0.5, lt=1.0)


def _randint() -> int:
    return np.random.randint(0, 10000)


def _check_versions(expected: str, libname: str) -> None:
    actual_ver = version(libname)
    if expected != actual_ver:
        raise NotImplementedError(
            f"Support for {libname} {expected} -> current ver {actual_ver}"
        )
    return None


class StanMCMCSamplingArguments(BaseModel):
    """MCMC sampling arguments for Stan."""

    _pystan_ver: str = "3.3.0"
    _httpstan_ver: str = "4.6.1"
    num_chains: PositiveInt = 4
    num_samples: PositiveInt = 1000
    num_warmup: PositiveInt = 1000
    save_warmup: bool = False
    num_thin: int = 1
    delta: TargetAcceptFloat = 0.8  # type: ignore
    max_depth: PositiveInt = 10
    random_seed: PositiveInt = Field(default_factory=_randint)

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3SampleArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        _check_versions(self._pystan_ver, "pystan")
        _check_versions(self._httpstan_ver, "httpstan")
        return None


class Pymc3SampleArguments(BaseModel):
    """Model `sample()` keyword arguments (PyMC3 v3.11.4)."""

    _pymc3_ver: str = "3.11.4"
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

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3SampleArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        _check_versions(self._pymc3_ver, "pymc3")
        return None


class Pymc3FitArguments(BaseModel):
    """Model `fit()` keyword arguments (PyMC3 v3.11.4)."""

    _pymc3_ver: str = "3.11.4"
    n: PositiveInt = 10000
    method: VIMethod = "advi"
    draws: PositiveInt = 1000
    random_seed: PositiveInt = Field(default_factory=_randint)
    inf_kwargs: Optional[dict[str, BasicTypes]] = None
    progressbar: bool = True

    def __init__(self, **data: dict[str, Any]) -> None:
        """Create a Pymc3FitArguments object.

        Raises:
            NotImplementedError: If the currently installed PyMC3 version is different
            from the supported version.
        """
        super().__init__(**data)
        _check_versions(self._pymc3_ver, "pymc3")
        return None


class ModelingSamplingArguments(BaseModel):
    """Keyword sampling arguments for methods of fitting a Bayesian models."""

    stan_mcmc: Optional[StanMCMCSamplingArguments] = None
    pymc3_mcmc: Optional[Pymc3SampleArguments] = None
    pymc3_advi: Optional[Pymc3FitArguments] = None
