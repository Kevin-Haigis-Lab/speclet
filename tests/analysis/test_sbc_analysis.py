from pathlib import Path
from typing import Callable

import arviz as az
import pandas as pd
import pytest

from src.analysis import sbc_analysis as sbcanal
from src.modeling import simulation_based_calibration_helpers as sbchelp

#### ---- SBCAnalysis ---- ####


def mock_get_sbc_results(*args, **kwargs):
    return 1


class TestSBCAnalysis:
    @pytest.mark.parametrize("pattern", ("A", "my-pattern", "a_pattern", "32nf343n"))
    @pytest.mark.parametrize("n_simulations", (0, 6, 100))
    def test_init(self, tmp_path: Path, pattern: str, n_simulations: int):
        sbc_analyzer = sbcanal.SBCAnalysis(
            root_dir=tmp_path, pattern=pattern, n_simulations=n_simulations
        )
        assert isinstance(sbc_analyzer, sbcanal.SBCAnalysis)

    @pytest.mark.parametrize("pattern", ("A", "my-pattern", "a_pattern", "32nf343n"))
    @pytest.mark.parametrize("n_simulations", (0, 6, 100))
    def test_get_simulation_directories_and_filemanagers(
        self, tmp_path: Path, pattern: str, n_simulations: int
    ):
        for i in range(n_simulations):
            (tmp_path / f"{pattern}_{i}").mkdir()

        sbc_analyzer = sbcanal.SBCAnalysis(
            root_dir=tmp_path, pattern=pattern, n_simulations=n_simulations
        )
        assert isinstance(sbc_analyzer, sbcanal.SBCAnalysis)
        dirs = sbc_analyzer.get_simulation_directories()
        assert len(dirs) == n_simulations
        assert all([d.is_dir() for d in dirs])
        file_managers = sbc_analyzer.get_simulation_file_managers()
        assert len(file_managers) == n_simulations
        assert not any([fm.all_data_exists() for fm in file_managers])

    @pytest.mark.parametrize("multithreaded", (False, True))
    @pytest.mark.parametrize("pattern", ("A", "my-pattern", "a_pattern", "32nf343n"))
    @pytest.mark.parametrize("n_simulations", (0, 6, 100))
    def test_get_simulation_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        multithreaded: bool,
        pattern: str,
        n_simulations: int,
        return_true: Callable,
    ):

        monkeypatch.setattr(
            sbcanal.sbc.SBCFileManager, "get_sbc_results", mock_get_sbc_results
        )
        monkeypatch.setattr(sbcanal.sbc.SBCFileManager, "all_data_exists", return_true)

        for i in range(n_simulations):
            (tmp_path / f"{pattern}_{i}").mkdir()

        sbc_analyzer = sbcanal.SBCAnalysis(
            root_dir=tmp_path, pattern=pattern, n_simulations=n_simulations
        )
        sbc_results = sbc_analyzer.get_simulation_results(multithreaded=multithreaded)
        assert len(sbc_results) == n_simulations

    def test_posterior_accuracy_simple(self, tmp_path: Path):
        sbc_analyzer = sbcanal.SBCAnalysis(
            root_dir=tmp_path, pattern="pattern", n_simulations=100
        )
        sim_posteriors_df = pd.DataFrame(
            {
                "parameter_name": ["a", "a", "a", "a"],
                "within_hdi": [True, False, True, False],
            }
        )
        acc_results = sbc_analyzer.run_posterior_accuracy_test(sim_posteriors_df)
        assert pytest.approx(
            acc_results[acc_results.parameter_name == "a"].within_hdi[0], 0.5
        )

    def test_posterior_accuracy_multiple(self, tmp_path: Path):
        sbc_analyzer = sbcanal.SBCAnalysis(
            root_dir=tmp_path, pattern="pattern", n_simulations=100
        )
        sim_posteriors_df = pd.DataFrame(
            {
                "parameter_name": ["a", "a", "a", "a", "b", "b", "b"],
                "within_hdi": [True, False, True, False, True, False, False],
            }
        )
        acc_results = sbc_analyzer.run_posterior_accuracy_test(sim_posteriors_df)

        for p in sim_posteriors_df.parameter_name.unique():
            assert len(acc_results[acc_results.parameter_name == p]) == 1
        assert pytest.approx(
            acc_results[acc_results.parameter_name == "a"].within_hdi.values[0], 0.5
        )
        assert pytest.approx(
            acc_results[acc_results.parameter_name == "b"].within_hdi.values[0],
            1.0 / 3.0,
        )

    def test_mcmc_diagnostics(self, tmp_path: Path, centered_eight: az.InferenceData):
        n_sims = 10
        pattern = "perm"
        for i in range(n_sims):
            dir = tmp_path / f"{pattern}{i}"
            dir.mkdir()
            sbchelp.SBCFileManager(dir).save_sbc_results(
                priors={},
                inference_obj=centered_eight,
                posterior_summary=pd.DataFrame(),
            )

        sbc_analyzer = sbcanal.SBCAnalysis(
            root_dir=tmp_path, pattern=pattern, n_simulations=n_sims
        )
        mcmc_diagnostics = sbc_analyzer.mcmc_diagnostics()
        assert isinstance(mcmc_diagnostics, dict)
        assert len(mcmc_diagnostics.keys()) > 0


#### ---- Uniformity test ---- ####


@pytest.mark.parametrize(
    "k_draws, exp, low, high", [(2, 50, 25, 75), (3, 100 / 3, 20, 50)]
)
def test_expected_range_under_uniform(
    k_draws: int, exp: float, low: float, high: float
):
    _exp, _lower, _upper = sbcanal.expected_range_under_uniform(
        n_sims=100, k_draws=k_draws
    )
    assert _exp == exp
    assert low < _lower < exp
    assert exp < _upper < high
