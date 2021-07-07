from pathlib import Path
from typing import Tuple
from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.io import model_config
from src.project_enums import ModelFitMethod, SpecletPipeline


def _filter_empty_configs(configs: model_config.ModelConfigs) -> bool:
    return len(configs.configurations) > 0


@given(st.builds(model_config.ModelConfigs).filter(_filter_empty_configs))
def test_model_names_are_unique_fails(model_configs: model_config.ModelConfigs):
    model_configs.configurations.append(model_configs.configurations[0])
    with pytest.raises(model_config.ModelNamesAreNotAllUnique):
        model_config.check_model_names_are_unique(model_configs)


@given(st.builds(model_config.ModelConfigs).filter(_filter_empty_configs))
def test_model_names_are_unique_does_not_fail(model_configs: model_config.ModelConfigs):
    for idx, config in enumerate(model_configs.configurations):
        _config = config.copy()
        _config.name = str(uuid4())
        model_configs.configurations[idx] = _config
    assert model_config.check_model_names_are_unique(model_configs) is None


def test_get_model_configurations(mock_model_config: Path):
    config = model_config.get_model_configurations(mock_model_config)
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
        config = model_config.get_configuration_for_model(mock_model_config, name)
        assert (config is not None) == result


@pytest.mark.parametrize(
    "name", ("my-test-model", "second-test-model", "no-config-test")
)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_get_model_sampling_kwargs_dict(
    name: str,
    pipeline: SpecletPipeline,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
):
    sampling_kwargs = model_config.get_sampling_kwargs(
        mock_model_config, name, pipeline=pipeline, fit_method=fit_method
    )
    assert isinstance(sampling_kwargs, dict)


@pytest.mark.DEV
@pytest.mark.parametrize(
    "name, pipeline, fit_method, exists",
    (
        ("my-test-model", SpecletPipeline.FITTING, ModelFitMethod.ADVI, False),
        ("second-test-model", SpecletPipeline.FITTING, ModelFitMethod.ADVI, True),
        ("second-test-model", SpecletPipeline.FITTING, ModelFitMethod.MCMC, False),
        ("second-test-model", SpecletPipeline.SBC, ModelFitMethod.ADVI, True),
        ("second-test-model", SpecletPipeline.SBC, ModelFitMethod.MCMC, True),
        ("no-config-test", SpecletPipeline.FITTING, ModelFitMethod.MCMC, False),
        ("no-config-test", SpecletPipeline.SBC, ModelFitMethod.MCMC, False),
        ("no-config-test", SpecletPipeline.FITTING, ModelFitMethod.ADVI, False),
    ),
)
def test_get_model_sampling_kwargs_exist(
    name: str,
    pipeline: SpecletPipeline,
    fit_method: ModelFitMethod,
    exists: bool,
    mock_model_config: Path,
):
    sampling_kwargs = model_config.get_sampling_kwargs(
        mock_model_config, name, fit_method=fit_method, pipeline=pipeline
    )
    assert (sampling_kwargs != {}) == exists
