"""Automated checks for successful MCMC sampling."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import arviz as az
import numpy as np

CheckResult = tuple[bool, str]
PosteriorCheck = Callable[[az.InferenceData], CheckResult]


class CheckBFMI:
    """Check the BFMI is within a given range."""

    def __init__(self, min_bfmi: float = 0.2, max_bfmi: float = 1.7) -> None:
        """Check the BFMI is within a given range.

        Args:
            min_bfmi (float, optional): Minimum BFMI. Defaults to 0.2.
            max_bfmi (float, optional): Maximum BFMI. Defaults to 1.7.
        """
        self.min_bfmi = min_bfmi
        self.max_bfmi = max_bfmi
        return None

    def __call__(self, trace: az.InferenceData) -> CheckResult:
        """Check all BFMI values fall within an acceptable range."""
        bfmi: np.ndarray = az.bfmi(trace)
        res = (self.min_bfmi <= bfmi) * (bfmi <= self.max_bfmi)
        if np.all(res):
            return True, "All BFMI within range."
        else:
            n_fails = np.sum(res)
            _bfmi = ",".join([f"{x:0.3f}" for x in bfmi])
            return False, f"{n_fails} BFMI outside range: {_bfmi}."

    def __str__(self) -> str:
        return "check_bfmi"

    def __repr__(self) -> str:
        return str(self)


class CheckStepSize:
    """Check the average sample size is above some minimum value."""

    def __init__(self, min_ss: float = 0.0005) -> None:
        """Check the average sample size is above some minimum value.

        Args:
            min_ss (float, optional): Minimum average step size. Defaults to 0.0005.
        """
        self.min_ss = min_ss
        return None

    def __call__(self, trace: az.InferenceData) -> CheckResult:
        """Check the average sample size is above some minimum value."""
        assert hasattr(trace, "sample_stats"), "No sampling statistics available."
        assert "step_size" in trace.sample_stats, "No step size information."
        avg_step_size = trace.sample_stats["step_size"].mean(axis=1).values
        res = avg_step_size >= self.min_ss
        if np.all(res):
            return True, "All average step sizes above minimum."
        else:
            n_fails = np.sum(res)
            _ss = ",".join([f"{x:0.2e}" for x in avg_step_size])
            return False, f"{n_fails} agf. step sizes less than threshold: {_ss}"

    def __str__(self) -> str:
        return "check_step_size"

    def __repr__(self) -> str:
        return str(self)


class CheckMarginalPosterior:
    """Check a marginal posterior has expected properties."""

    def __init__(
        self, var_name: str, min_avg: float = -np.inf, max_avg: float = np.inf
    ) -> None:
        """Check a marginal posterior distribution has expected properties.

        Args:
            var_name (str): Model variable name.
            min_avg (float): Minimum value for the average of the marginal posterior
            distribution.
            max_avg (float): Maximum value for the average of the marginal posterior
            distribution.
        """
        self.var_name = var_name
        self.min_avg = min_avg
        self.max_avg = max_avg
        return None

    def __call__(self, trace: az.InferenceData) -> CheckResult:
        """Check a marginal posterior distribution has expected properties."""
        assert self.var_name in trace.posterior, f"{self.var_name} not in posterior."
        assert hasattr(trace, "posterior"), "No posterior data in trace object."
        values = trace.posterior.get(self.var_name)
        assert values is not None, f"Could not get values for var {self.var_name}."
        avgs = values.mean(axis=1).values
        res = (self.min_avg <= avgs) * (avgs <= self.max_avg)
        if np.all(res):
            msg = f"Avg. marginal distribution of {self.var_name} within range."
            return True, msg
        else:
            _avgs = ",".join([f"{x:0.2e}" for x in avgs])
            msg = f"Marginal distribution of {self.var_name} outside of range: {_avgs}."
            return False, msg

    def __str__(self) -> str:
        return f"marginal-posterior-{self.var_name}"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class PosteriorCheckResults:
    """Results of a posteriors checks."""

    all_passed: bool
    message: str
    check_results: dict[str, CheckResult]


def _get_checker_name(check: Any) -> str:
    if hasattr(check, "name"):
        return check.name
    elif hasattr(check, "__str__"):
        return str(check)
    elif hasattr(check, "__name__"):
        return check.__name__
    return "(unnamed)"


class FailedSamplingStatisticsChecksError(BaseException):
    """Failed posterior check(s)."""

    ...


def check_mcmc_sampling(
    trace: az.InferenceData, checks: Iterable[PosteriorCheck]
) -> PosteriorCheckResults:
    """Check posterior attributes.

    Args:
        trace (az.InferenceData): Posterior data.
        checks (Iterable[SampleStatCheck]): A collection of checks to run on the
        posterior data.

    Returns:
        PosteriorCheckResults: Result of the checks.
    """
    results: dict[str, CheckResult] = {}
    for check in checks:
        name = _get_checker_name(check)
        results[name] = check(trace)

    all_passed = all([res[0] for res in results.values()])
    message = "\n".join([res[1] for res in results.values()])
    return PosteriorCheckResults(
        all_passed=all_passed, message=message, check_results=results
    )
