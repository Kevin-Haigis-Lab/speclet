from pathlib import Path
from typing import Callable

import pytest

from src.analysis import sbc_analysis as sbcanal

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
