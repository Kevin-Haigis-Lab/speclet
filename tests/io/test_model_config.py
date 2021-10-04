from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from src.io import model_config
from src.project_enums import ModelFitMethod, SpecletPipeline


def _filter_empty_configs(configs: model_config.ModelConfigs) -> bool:
    return len(configs.configurations) > 0


@given(st.builds(model_config.ModelConfigs).filter(_filter_empty_configs))
def test_model_names_are_unique_fails(model_configs: model_config.ModelConfigs) -> None:
    print(model_configs)
    model_configs.configurations.append(model_configs.configurations[0])
    with pytest.raises(model_config.ModelNamesAreNotAllUnique):
        model_config.check_model_names_are_unique(model_configs)


@given(st.builds(model_config.ModelConfigs).filter(_filter_empty_configs))
def test_model_names_are_unique_does_not_fail(
    model_configs: model_config.ModelConfigs,
) -> None:
    for idx, config in enumerate(model_configs.configurations):
        _config = config.copy()
        _config.name = str(uuid4())
        model_configs.configurations[idx] = _config
    assert model_config.check_model_names_are_unique(model_configs) is None


def test_get_model_configurations(mock_model_config: Path) -> None:
    config = model_config.get_model_configurations(mock_model_config)
    assert len(config.configurations) == 3


def test_get_model_configuration(mock_model_config: Path) -> None:
    names: tuple[str, ...] = (
        "my-test-model",
        "second-test-model",
        "not-a-model-name",
        "no-config-test",
    )
    results: tuple[bool, ...] = (True, True, False, True)
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
) -> None:
    sampling_kwargs = model_config.get_sampling_kwargs(
        mock_model_config, name, pipeline=pipeline, fit_method=fit_method
    )
    assert isinstance(sampling_kwargs, dict)


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
) -> None:
    sampling_kwargs = model_config.get_sampling_kwargs(
        mock_model_config, name, fit_method=fit_method, pipeline=pipeline
    )
    assert (sampling_kwargs != {}) == exists


@given(config=st.builds(model_config.ModelConfig))
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
def test_get_model_sampling_from_config(
    config: model_config.ModelConfig,
    fit_method: ModelFitMethod,
    pipeline: SpecletPipeline,
) -> None:
    kwargs = model_config.get_sampling_kwargs_from_config(
        config, pipeline=pipeline, fit_method=fit_method
    )
    if config.pipeline_sampling_parameters is None:
        assert kwargs == {}
    assert isinstance(kwargs, dict)


@given(
    config=st.builds(model_config.ModelConfig),
    expected_kwargs=st.dictionaries(st.text(), st.integers()),
)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
def test_get_model_sampling_from_config_correct_pipeline_fitmethod(
    config: model_config.ModelConfig,
    expected_kwargs: dict[str, int],
    fit_method: ModelFitMethod,
    pipeline: SpecletPipeline,
) -> None:
    pipeline_params = {pipeline: {fit_method: expected_kwargs.copy()}}
    config.pipeline_sampling_parameters = pipeline_params
    kwargs = model_config.get_sampling_kwargs_from_config(
        config, pipeline=pipeline, fit_method=fit_method
    )
    assert kwargs == expected_kwargs


@pytest.mark.DEV
def test_model_config_with_optional_pipeline_field() -> None:
    yaml_txt = """
- name: with-pipelines
  description: "A description."
  model: speclet-simple
  fit_methods:
      - MCMC
      - ADVI
  pipelines:
      - fitting
      - sbc
  debug: true
- name: without-pipelines
  description: "A description."
  model: speclet-simple
  fit_methods:
    - MCMC
    - ADVI
  debug: true
    """
    configs = model_config.ModelConfigs(configurations=yaml.safe_load(yaml_txt))
    assert len(configs.configurations) == 2
    for config in configs.configurations:
        if config.name == "with-pipelines":
            assert len(config.pipelines) == 2
        elif config.name == "without-pipelines":
            assert len(config.pipelines) == 0
        else:
            assert 1 == 2  # Should never get here.
