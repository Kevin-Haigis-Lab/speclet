from pathlib import Path
from typing import Tuple
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.models import configuration as c
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.project_enums import ModelParameterization as MP


@given(st.builds(c.ModelConfigs).filter(lambda x: len(x.configurations) > 0))
def test_model_names_are_unique_fails(model_configs: c.ModelConfigs):
    model_configs.configurations.append(model_configs.configurations[0])
    with pytest.raises(c.ModelNamesAreNotAllUnique):
        c.check_model_names_are_unique(model_configs)


@given(st.builds(c.ModelConfigs).filter(lambda x: len(x.configurations) > 0))
def test_model_names_are_unique_does_not_fail(model_configs: c.ModelConfigs):
    for idx, config in enumerate(model_configs.configurations):
        _config = config.copy()
        _config.name = str(uuid4())
        model_configs.configurations[idx] = _config
    assert c.check_model_names_are_unique(model_configs) is None


def test_get_model_configurations(mock_model_config: Path):
    config = c.get_model_configurations(mock_model_config)
    assert len(config.configurations) == 2


def test_get_model_configuration(mock_model_config: Path):
    names: Tuple[str, ...] = ("my-test-model", "second-test-model", "not-a-model-name")
    results: Tuple[bool, ...] = (True, True, False)
    assert len(names) == len(results)
    for name, result in zip(names, results):
        config = c.get_configuration_for_model(mock_model_config, name)
        assert (config is not None) == result


def test_configure_model(mock_model_config: Path, tmp_path: Path):
    sp = SpecletTestModel("my-test-model", tmp_path)
    c.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.NONCENTERED
    assert sp.config.cov2


def test_configure_model_no_change(mock_model_config: Path, tmp_path: Path):
    sp = SpecletTestModel("my-test-model-that-doesnot-exist", tmp_path)
    c.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.CENTERED
    assert not sp.config.cov2
