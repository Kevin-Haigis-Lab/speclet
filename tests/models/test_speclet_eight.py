from pathlib import Path

import pytest

from src.models.speclet_eight import SpecletEight


@pytest.fixture(scope="function")
def sp8(tmp_path: Path) -> SpecletEight:
    return SpecletEight("test-model", root_cache_dir=tmp_path)


def test_init(tmp_path: Path) -> None:
    sp8 = SpecletEight("test-model", root_cache_dir=tmp_path)
    assert sp8.model is None
    assert sp8.data_manager is not None


def test_get_data(sp8: SpecletEight) -> None:
    data = sp8.data_manager.get_data()
    for c in ["counts_initial_adj", "counts_final"]:
        assert c in data.columns


def test_build_model(sp8: SpecletEight) -> None:
    assert sp8.model is None
    sp8.build_model()
    assert sp8.model is not None
    assert sp8.observed_var_name is not None
