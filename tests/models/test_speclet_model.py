from pathlib import Path

import pymc3 as pm
import pytest

from src.managers.model_data_managers import CrcDataManager, MockDataManager
from src.models import speclet_model


class MockSpecletModelClass(speclet_model.SpecletModel):

    data_manager = MockDataManager()

    def model_specification(self) -> pm.Model:
        data = self.data_manager.get_data()
        with pm.Model() as model:
            b = pm.Normal("b", 0, 10)
            a = pm.Normal("a", 0, 10)
            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal(  # noqa: F841
                "y", a + b * data["x"].values, sigma, observed=data["y"].values
            )
        return model


class TestSpecletModel:
    def test_build_model_fails_with_no_data_manager(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        with pytest.raises(AttributeError, match="without a data manager"):
            sp.build_model()

    def test_build_model_fails_with_no_model_spec(self, tmp_path: Path):
        sp = speclet_model.SpecletModel("test-model", root_cache_dir=tmp_path)
        sp.data_manager = CrcDataManager(debug=True)
        with pytest.raises(AttributeError, match="`model` attribute cannot be None"):
            sp.build_model()

    def test_build_model(self, tmp_path: Path):
        sp = MockSpecletModelClass(
            name="test-model", root_cache_dir=tmp_path, debug=True
        )
        sp.build_model()
        assert sp.model is not None
        assert isinstance(sp.model, pm.Model)
