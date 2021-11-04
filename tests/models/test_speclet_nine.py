from pathlib import Path

import numpy as np
import pytest

from src.models.speclet_nine import SpecletNine


@pytest.fixture(scope="function")
def sp9(tmp_path: Path) -> SpecletNine:
    return SpecletNine("test-model", root_cache_dir=tmp_path)


def test_init(tmp_path: Path) -> None:
    sp9 = SpecletNine("test-model", root_cache_dir=tmp_path)
    assert sp9.model is None
    assert sp9.data_manager is not None


def test_get_data(sp9: SpecletNine) -> None:
    data = sp9.data_manager.get_data()
    for c in ["counts_initial_adj", "counts_final"]:
        assert c in data.columns


def test_build_model(sp9: SpecletNine) -> None:
    assert sp9.model is None
    sp9.build_model()
    assert sp9.model is not None
    assert sp9.observed_var_name is not None


@pytest.mark.slow
def test_sample_model(sp9: SpecletNine) -> None:
    data = sp9.data_manager.get_data()
    ct_i = np.abs(np.random.normal(loc=10, scale=2, size=data.shape[0]))
    # ct_i = np.ones(data.shape[0])
    ct_f = np.abs(ct_i + np.random.normal(loc=0, scale=5, size=data.shape[0]))
    data["counts_initial_adj"] = ct_i.astype(np.int32)
    data["counts_final"] = ct_f.astype(np.int32)
    sp9.data_manager.set_data(data, apply_transformations=False)

    sp9.build_model()
    assert sp9.mcmc_results is None
    _ = sp9.mcmc_sample_model(
        prior_pred_samples=100, sample_kwargs={"draws": 10, "tune": 10, "chains": 1}
    )
    assert sp9.mcmc_results is not None


# TODO: add test for ADVI sampling
