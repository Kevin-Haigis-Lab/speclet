import arviz as az
import numpy as np
import pytest

import speclet.modeling.posterior_checks as post_checks


def _mock_bfmi(trace: az.InferenceData) -> np.ndarray:
    n_chains = len(trace.posterior.coords["chain"])
    return np.asarray([0.67] * n_chains)


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
) -> None:
    monkeypatch.setattr(az, "bfmi", _mock_bfmi)
    check = post_checks.CheckBFMI(min_bfmi=min_bfmi, max_bfmi=max_bfmi)
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


@pytest.mark.parametrize(
    ["min_ss", "pass_check"],
    [(0.001, True), (2, False)],
)
def test_check_step_size(
    centered_eight_idata: az.InferenceData,
    min_ss: float,
    pass_check: bool,
) -> None:
    check = post_checks.CheckStepSize(min_ss=min_ss)
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


@pytest.mark.parametrize(
    ["min_avg", "max_avg", "pass_check"],
    [(-50, 50, True), (100, np.inf, False), (-np.inf, -100, False)],
)
def test_check_marginal_posterior(
    centered_eight_idata: az.InferenceData,
    min_avg: float,
    max_avg: float,
    pass_check: bool,
) -> None:
    check = post_checks.CheckMarginalPosterior("tau", min_avg=min_avg, max_avg=max_avg)
    res = check(centered_eight_idata)
    assert res[0] == pass_check
    return None


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
) -> None:
    monkeypatch.setattr(az, "bfmi", _mock_bfmi)
    res = post_checks.check_mcmc_sampling(centered_eight_idata, checks)
    assert res.all_passed == pass_check
    assert len(res.check_results) == len(checks)
