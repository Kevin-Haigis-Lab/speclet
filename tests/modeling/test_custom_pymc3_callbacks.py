import numpy as np
import pymc3 as pm
import pytest
from pytest import CaptureFixture

from speclet.modeling import custom_pymc3_callbacks as pymc3calls


@pytest.fixture(scope="module")
def mock_model() -> pm.Model:
    np.random.seed(1)
    X = np.ones(5)
    y = X * 2 + np.random.randn(len(X))

    with pm.Model() as m:
        a = pm.Normal("a", 50, 100)
        b = pm.Normal("b", 50, 100)
        mu = a + b * X
        sigma = pm.HalfCauchy("error", 1)
        obs = pm.Normal("obs", mu, sigma, observed=y)  # noqa: F841

    return m


class TestDivergenceFractionCallback:
    def test_divergences(self, mock_model: pm.Model) -> None:
        n_tune_steps = 200
        cb = pymc3calls.DivergenceFractionCallback(
            n_tune_steps=n_tune_steps, max_frac=0.01
        )
        with pytest.raises(pymc3calls.TooManyDivergences):
            with mock_model:
                _ = pm.sample(
                    draws=1000,
                    tune=n_tune_steps,
                    chains=1,
                    cores=1,
                    callback=cb,
                    random_seed=404,
                    return_inferencedata=True,
                )

    def test_min_samples_param(self, mock_model: pm.Model) -> None:
        n_tune_steps = 200
        cb = pymc3calls.DivergenceFractionCallback(
            n_tune_steps=n_tune_steps, min_samples=10000
        )
        with mock_model:
            trace = pm.sample(
                draws=1000,
                tune=n_tune_steps,
                chains=1,
                cores=1,
                callback=cb,
                random_seed=123,
                return_inferencedata=True,
            )

        assert trace.posterior["a"].shape[1] == 1000

    def test_max_frac_param(self, mock_model: pm.Model) -> None:
        n_tune_steps = 200
        cb = pymc3calls.DivergenceFractionCallback(
            n_tune_steps=n_tune_steps, max_frac=1.1
        )
        with mock_model:
            trace = pm.sample(
                draws=1000,
                tune=n_tune_steps,
                chains=1,
                cores=1,
                callback=cb,
                random_seed=123,
                return_inferencedata=True,
            )

        assert trace.posterior["a"].shape[1] == 1000


class TestProgressPrinterCallback:
    @pytest.mark.parametrize(
        "draws, tune, every_n, tuning, expected_rows",
        [
            (2, 2, 1, True, 8),
            (2, 2, 10, True, 2),
            (100, 100, 50, True, 8),
            (100, 100, 50, False, 4),
        ],
    )
    def test_progress_is_printed_at_expected_freq(
        self,
        mock_model: pm.Model,
        capsys: CaptureFixture,
        draws: int,
        tune: int,
        every_n: int,
        tuning: bool,
        expected_rows: int,
    ) -> None:
        cb = pymc3calls.ProgressPrinterCallback(every_n=every_n, tuning=tuning)
        with mock_model:
            _ = pm.sample(
                draws=draws,
                tune=tune,
                chains=2,
                cores=1,
                callback=cb,
                progressbar=False,
                return_inferencedata=True,
            )
        out = capsys.readouterr().out
        assert len(out.strip().split("\n")) == expected_rows
