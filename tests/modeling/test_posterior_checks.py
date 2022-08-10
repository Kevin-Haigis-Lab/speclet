from typing import Any, Callable

import arviz as az
import numpy as np
import pytest
import xarray as xr
from arviz import ess as original_ess

import speclet.modeling.posterior_checks as post_checks


def _mock_bfmi(trace: az.InferenceData) -> np.ndarray:
    n_chains = len(trace.posterior.coords["chain"])
    return np.asarray([0.67] * n_chains)


def _set_n_chains(trace: az.InferenceData, n_chains: int) -> az.InferenceData:
    trace.posterior = trace.posterior.sel(chain=list(range(n_chains)))
    return trace


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ["min_bfmi", "max_bfmi", "pass_check"],
    [(0.2, 1.0, True), (0.8, 1.0, False), (0.2, 0.5, False)],
)
def test_check_bfmi(
    centered_eight_idata: az.InferenceData,
    min_bfmi: float,
    max_bfmi: float,
    pass_check: bool,
    monkeypatch: pytest.MonkeyPatch,
    n_chains: int,
) -> None:
    monkeypatch.setattr(az, "bfmi", _mock_bfmi)
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    check = post_checks.CheckBFMI(min_bfmi=min_bfmi, max_bfmi=max_bfmi)
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ["min_ss", "pass_check"],
    [(0.001, True), (2, False)],
)
def test_check_step_size(
    centered_eight_idata: az.InferenceData,
    min_ss: float,
    pass_check: bool,
    n_chains: int,
) -> None:
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    check = post_checks.CheckStepSize(min_ss=min_ss)
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ["min_avg", "max_avg", "pass_check"],
    [(-50, 50, True), (100, np.inf, False), (-np.inf, -100, False)],
)
def test_check_marginal_posterior(
    centered_eight_idata: az.InferenceData,
    min_avg: float,
    max_avg: float,
    pass_check: bool,
    n_chains: int,
) -> None:
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    check = post_checks.CheckMarginalPosterior("tau", min_avg=min_avg, max_avg=max_avg)
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ["var_name", "skip_if_missing", "pass_check"],
    [("tau", False, True), ("fake-var", True, True)],
)
def test_check_marginal_posterior_if_missing_variable(
    var_name: str,
    skip_if_missing: bool,
    pass_check: bool,
    centered_eight_idata: az.InferenceData,
    n_chains: int,
) -> None:
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    check = post_checks.CheckMarginalPosterior(
        var_name, skip_if_missing=skip_if_missing
    )
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ["checks", "pass_check"],
    [
        (
            [
                post_checks.CheckBFMI(),
                post_checks.CheckStepSize(),
                post_checks.CheckMarginalPosterior("tau"),
            ],
            True,
        ),
        (
            [
                post_checks.CheckBFMI(min_bfmi=1.0),
                post_checks.CheckStepSize(),
                post_checks.CheckMarginalPosterior("tau"),
            ],
            False,
        ),
    ],
)
def test_check_mcmc_sampling(
    centered_eight_idata: az.InferenceData,
    monkeypatch: pytest.MonkeyPatch,
    checks: list[post_checks.PosteriorCheck],
    pass_check: bool,
    n_chains: int,
) -> None:
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    monkeypatch.setattr(az, "bfmi", _mock_bfmi)
    res = post_checks.check_mcmc_sampling(centered_eight_idata, checks)
    assert res.all_passed == pass_check
    assert len(res.check_results) == len(checks)


def _generate_mock_ess(
    var_name: str, ess_value: float | np.ndarray
) -> Callable[..., xr.Dataset]:
    def f(trace: az.InferenceData, **kwargs: Any) -> xr.Dataset:
        ess = original_ess(trace, **kwargs)
        ess[var_name].values = ess_value
        return ess

    return f


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ("min_frac_ess", "ess_value", "passes"), [(0.1, 500, True), (1, 1, False)]
)
def test_check_effective_sample_size(
    centered_eight_idata: az.InferenceData,
    monkeypatch: pytest.MonkeyPatch,
    min_frac_ess: float,
    ess_value: float,
    n_chains: int,
    passes: bool,
) -> None:
    var_name = "tau"
    monkeypatch.setattr(
        az, "ess", _generate_mock_ess(var_name=var_name, ess_value=ess_value)
    )
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    check = post_checks.CheckEffectiveSampleSize(
        var_name=var_name, min_frac_ess=min_frac_ess
    )
    res = check(centered_eight_idata)
    assert res[0] == passes


@pytest.mark.parametrize("n_chains", (1, 4))
@pytest.mark.parametrize(
    ("min_frac_ess", "ess_value", "passes"), [(0.1, 500, True), (1, 1, False)]
)
def test_check_effective_sample_size_multiple_dims(
    centered_eight_idata: az.InferenceData,
    monkeypatch: pytest.MonkeyPatch,
    min_frac_ess: float,
    ess_value: float,
    n_chains: int,
    passes: bool,
) -> None:
    var_name = "theta"

    # Make the return ESS array with only one value as the test ESS value.
    dim_name = centered_eight_idata.posterior[var_name].dims[2]
    shape = len(centered_eight_idata.posterior[var_name].coords[dim_name])
    n_draws = len(centered_eight_idata.posterior.coords["draw"])
    ess_array = np.array([n_draws * 2] * shape)
    i = np.random.choice(np.arange(shape))
    ess_array[i] = ess_value

    monkeypatch.setattr(
        az, "ess", _generate_mock_ess(var_name=var_name, ess_value=ess_array)
    )
    centered_eight_idata = _set_n_chains(centered_eight_idata, n_chains=n_chains)
    check = post_checks.CheckEffectiveSampleSize(
        var_name=var_name, min_frac_ess=min_frac_ess
    )
    res = check(centered_eight_idata)
    assert res[0] == passes
