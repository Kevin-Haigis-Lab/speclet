from pathlib import Path
from typing import Any, Dict

import arviz as az
import numpy as np
import pandas as pd
import pytest

from src.modeling import simulation_based_calibration_helpers as sbc


class TestSBCFileManager:
    def test_init(self, tmp_path: Path):
        fm = sbc.SBCFileManager(dir=tmp_path)
        assert not fm.all_data_exists()

    @pytest.fixture()
    def priors(self) -> Dict[str, Any]:
        return dict(
            alpha=np.random.uniform(0, 100, size=3),
            beta_log=np.random.uniform(0, 100, size=(10, 15)),
        )

    @pytest.fixture
    def posterior_summary(self) -> pd.DataFrame:
        return pd.DataFrame(dict(x=[5, 6, 7], y=["a", "b", "c"]))

    def test_saving(
        self, tmp_path: Path, priors: Dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)
        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()

    def test_reading(
        self, tmp_path: Path, priors: Dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)

        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()
        read_results = fm.get_sbc_results()
        assert isinstance(read_results, sbc.SBCResults)
        assert isinstance(read_results.inference_obj, az.InferenceData)
        for k in read_results.priors:
            np.testing.assert_array_equal(read_results.priors[k], priors[k])

        for c in read_results.posterior_summary.columns:
            np.testing.assert_array_equal(
                read_results.posterior_summary[c].values, posterior_summary[c].values
            )
