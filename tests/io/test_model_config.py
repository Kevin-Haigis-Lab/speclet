from pathlib import Path
from typing import Tuple
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.io import model_config as c


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
    assert len(config.configurations) == 3


def test_get_model_configuration(mock_model_config: Path):
    names: Tuple[str, ...] = (
        "my-test-model",
        "second-test-model",
        "not-a-model-name",
        "no-config-test",
    )
    results: Tuple[bool, ...] = (True, True, False, True)
    assert len(names) == len(results)
    for name, result in zip(names, results):
        config = c.get_configuration_for_model(mock_model_config, name)
        assert (config is not None) == result
