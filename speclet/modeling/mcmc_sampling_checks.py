"""Automated checks for successful MCMC sampling."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import arviz as az
import numpy as np

CheckResult = tuple[bool, str]
SampleStatCheck = Callable[[az.InferenceData], CheckResult]


class CheckBFMI:
    """Check the BFMI is within a given range."""

    name: str = "check_bfmi"

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
            _bfmi = ",".join(list(bfmi))
            return False, f"{n_fails} BFMI outside range: {_bfmi}."


class CheckStepSize:
    """Check the average sample size is above some minimum value."""

    name: str = "check_step_size"

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
            _ss = ",".join(list(avg_step_size))
            return False, f"{n_fails} agf. step sizes less than threshold: {_ss}"


@dataclass
class SampleStatCheckResults:
    """Results of a sampling statistics check."""

    all_passed: bool
    message: str
    check_results: dict[str, CheckResult]


def _get_checker_name(check: Any) -> str:
    if hasattr(check, "name"):
        return check.name
    if hasattr(check, "__name__"):
        return check.__name__
    else:
        return "(unnamed)"


class FailedSamplingStatisticsChecksError(BaseException):
    """Failed sampling statistics checks."""

    ...


def check_mcmc_sampling(
    trace: az.InferenceData, checks: Iterable[SampleStatCheck]
) -> SampleStatCheckResults:
    """Check MCMC sampling statistics.

    Args:
        trace (az.InferenceData): MCMC posterior.
        checks (Iterable[SampleStatCheck]): A collection of checks to run on the
        posterior sampling statistics.

    Returns:
        SampleStatCheckResults: Result of the checks.
    """
    results: dict[str, CheckResult] = {}
    for check in checks:
        name = _get_checker_name(check)
        results[name] = check(trace)

    all_passed = all([res[0] for res in results.values()])
    message = "\n".join([res[1] for res in results.values()])
    return SampleStatCheckResults(
        all_passed=all_passed, message=message, check_results=results
    )
