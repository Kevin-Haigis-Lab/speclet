from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from speclet import model_configuration as model_config
from speclet.bayesian_models.ceres_mimic import CeresMimic
from speclet.bayesian_models.speclet_five import SpecletFive
from speclet.bayesian_models.speclet_four import SpecletFour
from speclet.bayesian_models.speclet_model import SpecletModel
from speclet.bayesian_models.speclet_one import SpecletOne
from speclet.bayesian_models.speclet_pipeline_test_model import (
    SpecletTestModel,
    SpecletTestModelConfiguration,
)
from speclet.bayesian_models.speclet_seven import SpecletSeven
from speclet.bayesian_models.speclet_six import SpecletSix
from speclet.bayesian_models.speclet_two import SpecletTwo
from speclet.project_enums import ModelFitMethod, ModelOption
from speclet.project_enums import ModelParameterization as MP
from speclet.project_enums import SpecletPipeline


@pytest.fixture(scope="function")
def model_configs(mock_model_config: Path) -> model_config.ModelConfigs:
    return model_config.read_model_configurations(mock_model_config)


def test_configure_model(mock_model_config: Path, tmp_path: Path) -> None:
    sp = SpecletTestModel("my-test-model", tmp_path)
    model_config.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.NONCENTERED
    assert sp.config.cov2


def test_configure_model_no_change(mock_model_config: Path, tmp_path: Path) -> None:
    sp = SpecletTestModel("my-test-model-that-doesnot-exist", tmp_path)
    model_config.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.CENTERED
    assert not sp.config.cov2


@pytest.mark.parametrize("model_option", ModelOption)
def test_all_model_options_return_a_type(model_option: ModelOption) -> None:
    model_type = model_config.get_model_class(model_option)
    assert model_type is not None


model_options_expected_class_parameterization: list[
    tuple[ModelOption, type[SpecletModel]]
] = [
    (ModelOption.SPECLET_TEST_MODEL, SpecletTestModel),
    (ModelOption.CRC_CERES_MIMIC, CeresMimic),
    (ModelOption.SPECLET_ONE, SpecletOne),
    (ModelOption.SPECLET_TWO, SpecletTwo),
    (ModelOption.SPECLET_FOUR, SpecletFour),
    (ModelOption.SPECLET_FIVE, SpecletFive),
    (ModelOption.SPECLET_SIX, SpecletSix),
    (ModelOption.SPECLET_SEVEN, SpecletSeven),
]


@pytest.mark.parametrize(
    "model_option, expected_class", model_options_expected_class_parameterization
)
def test_get_model_class(model_option: ModelOption, expected_class: type) -> None:
    model_type = model_config.get_model_class(model_option)
    assert model_type == expected_class


def check_test_model_configurations(
    sp_model: SpecletTestModel, model_name: str
) -> None:
    if model_name == "my-test-model":
        assert sp_model.config.some_param is MP.NONCENTERED
        assert (
            sp_model.config.another_param
            is SpecletTestModelConfiguration().another_param
        )
        assert sp_model.config.cov1 == SpecletTestModelConfiguration().cov1
        assert sp_model.config.cov2
    elif model_name == "second-test-model":
        assert sp_model.config.some_param is MP.NONCENTERED
        assert (
            sp_model.config.another_param
            is SpecletTestModelConfiguration().another_param
        )
        assert sp_model.config.cov1
        assert sp_model.config.cov2
    elif model_name == "no-config-test":
        assert sp_model.config.some_param is SpecletTestModelConfiguration().some_param
        assert (
            sp_model.config.another_param
            is SpecletTestModelConfiguration().another_param
        )
        assert sp_model.config.cov1 == SpecletTestModelConfiguration().cov1
        assert sp_model.config.cov2 == SpecletTestModelConfiguration().cov2


@pytest.mark.parametrize(
    "model_name", ["my-test-model", "second-test-model", "no-config-test"]
)
def test_instantiate_and_configure_model(
    model_name: str, tmp_path: Path, mock_model_config: Path
) -> None:
    config = model_config.get_configuration_for_model(mock_model_config, model_name)
    assert config is not None
    sp_model = model_config.instantiate_and_configure_model(
        config,
        root_cache_dir=tmp_path,
    )
    assert isinstance(sp_model, SpecletTestModel)
    check_test_model_configurations(sp_model, model_name)


@pytest.mark.parametrize(
    "model_name", ["my-test-model", "second-test-model", "no-config-test"]
)
def test_get_config_and_instantiate_model(
    model_name: str, tmp_path: Path, mock_model_config: Path
) -> None:
    sp_model = model_config.get_config_and_instantiate_model(
        mock_model_config,
        name=model_name,
        root_cache_dir=tmp_path,
    )
    assert isinstance(sp_model, SpecletTestModel)
    check_test_model_configurations(sp_model, model_name)


@pytest.mark.DEV
def test_model_names_are_unique_fails(mock_model_config: Path) -> None:
    model_configs = model_config.read_model_configurations(mock_model_config)
    model_configs.configurations.append(model_configs.configurations[0])
    with pytest.raises(model_config.ModelNamesAreNotAllUnique):
        model_config.check_model_names_are_unique(model_configs)


@pytest.mark.DEV
def test_model_names_are_unique_does_not_fail(
    model_configs: model_config.ModelConfigs,
) -> None:
    for idx, config in enumerate(model_configs.configurations):
        _config = config.copy()
        _config.name = str(uuid4())
        model_configs.configurations[idx] = _config
    assert model_config.check_model_names_are_unique(model_configs)


def test_get_model_configurations(mock_model_config: Path) -> None:
    config = model_config.read_model_configurations(mock_model_config)
    assert len(config.configurations) == 4


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


@pytest.mark.parametrize("pipeline", SpecletPipeline)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_get_no_model_sampling_kwargs(
    pipeline: SpecletPipeline,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
) -> None:
    sampling_kwargs = model_config.get_sampling_kwargs(
        mock_model_config, "my-test-model", pipeline=pipeline, fit_method=fit_method
    )
    assert sampling_kwargs is None


@pytest.mark.parametrize("pipeline", SpecletPipeline)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
def test_get_sampling_kwargs(
    pipeline: SpecletPipeline,
    fit_method: ModelFitMethod,
    mock_model_config: Path,
) -> None:
    sampling_kwargs = model_config.get_sampling_kwargs(
        mock_model_config, "second-test-model", pipeline=pipeline, fit_method=fit_method
    )
    assert sampling_kwargs is not None


@pytest.mark.parametrize(
    "name, pipeline, fit_method, exists",
    (
        ("my-test-model", SpecletPipeline.FITTING, ModelFitMethod.ADVI, False),
        ("second-test-model", SpecletPipeline.FITTING, ModelFitMethod.ADVI, True),
        ("second-test-model", SpecletPipeline.FITTING, ModelFitMethod.MCMC, True),
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
    assert (sampling_kwargs is not None) == exists


def test_model_config_with_optional_pipeline_field() -> None:
    yaml_txt = """
- name: with-pipelines
  description: "A description."
  model: speclet-simple
  pipelines:
    fitting: ['MCMC', 'ADVI']
    sbc: ['MCMC', 'ADVI']

- name: without-pipelines
  description: "A description."
  model: speclet-simple
  pipelines:
    fitting: ['MCMC', 'ADVI']
"""
    configs = model_config.ModelConfigs(configurations=yaml.safe_load(yaml_txt))
    assert len(configs.configurations) == 2
    for config in configs.configurations:
        if config.name == "with-pipelines":
            assert len(config.pipelines[SpecletPipeline.FITTING]) == 2
            assert len(config.pipelines[SpecletPipeline.SBC]) == 2
        elif config.name == "without-pipelines":
            assert len(config.pipelines[SpecletPipeline.FITTING]) == 2
            assert config.pipelines.get(SpecletPipeline.SBC) is None
        else:
            assert 1 == 2  # Should never get here.
