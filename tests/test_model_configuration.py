from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from src import model_configuration as model_config
from src.misc.check_kwarg_dict import KeywordsNotInCallableParametersError
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_five import SpecletFive
from src.models.speclet_four import SpecletFour
from src.models.speclet_model import SpecletModel
from src.models.speclet_one import SpecletOne
from src.models.speclet_pipeline_test_model import (
    SpecletTestModel,
    SpecletTestModelConfiguration,
)
from src.models.speclet_seven import SpecletSeven
from src.models.speclet_six import SpecletSix
from src.models.speclet_two import SpecletTwo
from src.project_enums import ModelFitMethod, ModelOption
from src.project_enums import ModelParameterization as MP
from src.project_enums import SpecletPipeline


@pytest.mark.skip("Needs refactoring")
def test_configure_model(mock_model_config: Path, tmp_path: Path) -> None:
    sp = SpecletTestModel("my-test-model", tmp_path)
    model_config.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.NONCENTERED
    assert sp.config.cov2


@pytest.mark.skip("Needs refactoring")
def test_configure_model_no_change(mock_model_config: Path, tmp_path: Path) -> None:
    sp = SpecletTestModel("my-test-model-that-doesnot-exist", tmp_path)
    model_config.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.CENTERED
    assert not sp.config.cov2


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
@given(sampling_kwargs=st.dictionaries(st.text(), st.text()))
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
def test_check_sampling_kwargs_raises(
    sampling_kwargs: dict[str, str],
    fit_method: ModelFitMethod,
    pipeline: SpecletPipeline,
) -> None:
    if sampling_kwargs == {}:
        model_config.check_sampling_kwargs(
            sampling_kwargs, fit_method=fit_method, pipeline=pipeline
        )
    else:
        sampling_kwargs["fjieorjgiers;orgrhj"] = "vjdiorgvherogjheiorg"
        with pytest.raises(KeywordsNotInCallableParametersError):
            model_config.check_sampling_kwargs(
                sampling_kwargs, fit_method=fit_method, pipeline=pipeline
            )


@pytest.mark.skip("Needs refactoring")
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
def test_check_sampling_kwargs_empty_always_passes(
    fit_method: ModelFitMethod, pipeline: SpecletPipeline
) -> None:
    sampling_kwargs: dict[str, Any] = {}
    model_config.check_sampling_kwargs(
        sampling_kwargs, fit_method=fit_method, pipeline=pipeline
    )


@pytest.mark.skip("Needs refactoring")
@pytest.mark.parametrize(
    "sampling_kwargs",
    [
        {"draws": 100, "prior_pred_samples": "hi"},
        {"prior_pred_samples": 10},
        {"draws": 100.0001, "prior_pred_samples": False},
    ],
)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
def test_check_sampling_kwargs_all_fitmethods(
    sampling_kwargs: dict[str, Any],
    fit_method: ModelFitMethod,
    pipeline: SpecletPipeline,
) -> None:
    model_config.check_sampling_kwargs(
        sampling_kwargs, fit_method=fit_method, pipeline=pipeline
    )


@pytest.mark.skip("Needs refactoring")
@pytest.mark.parametrize(
    "sampling_kwargs, intended_fit_method",
    [
        ({"draws": 100, "tune": 20, "cores": 2, "chains": 40}, ModelFitMethod.MCMC),
        ({"draws": 100, "target_accept": 0.0001}, ModelFitMethod.MCMC),
        ({"draws": 100, "method": 20, "n_iterations": 2}, ModelFitMethod.ADVI),
        ({"draws": 77, "method": "e", "n_iterations": "f"}, ModelFitMethod.ADVI),
        ({"draws": 33, "method": False, "n_iterations": True}, ModelFitMethod.ADVI),
    ],
)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("pipeline", SpecletPipeline)
def test_check_sampling_kwargs_fitmethod_specfic(
    sampling_kwargs: dict[str, Any],
    intended_fit_method: ModelFitMethod,
    fit_method: ModelFitMethod,
    pipeline: SpecletPipeline,
) -> None:
    if fit_method == intended_fit_method:
        model_config.check_sampling_kwargs(
            sampling_kwargs, fit_method=fit_method, pipeline=pipeline
        )
    else:
        with pytest.raises(KeywordsNotInCallableParametersError):
            model_config.check_sampling_kwargs(
                sampling_kwargs, fit_method=fit_method, pipeline=pipeline
            )


def _filter_empty_configs(configs: model_config.ModelConfigs) -> bool:
    return len(configs.configurations) > 0


@pytest.mark.skip("Needs refactoring")
@given(st.builds(model_config.ModelConfigs).filter(_filter_empty_configs))
def test_model_names_are_unique_fails(model_configs: model_config.ModelConfigs) -> None:
    print(model_configs)
    model_configs.configurations.append(model_configs.configurations[0])
    with pytest.raises(model_config.ModelNamesAreNotAllUnique):
        model_config.check_model_names_are_unique(model_configs)


@pytest.mark.skip("Needs refactoring")
@given(st.builds(model_config.ModelConfigs).filter(_filter_empty_configs))
def test_model_names_are_unique_does_not_fail(
    model_configs: model_config.ModelConfigs,
) -> None:
    for idx, config in enumerate(model_configs.configurations):
        _config = config.copy()
        _config.name = str(uuid4())
        model_configs.configurations[idx] = _config
    assert model_config.check_model_names_are_unique(model_configs)


@pytest.mark.skip("Needs refactoring")
def test_get_model_configurations(mock_model_config: Path) -> None:
    config = model_config.read_model_configurations(mock_model_config)
    assert len(config.configurations) == 3


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
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


@pytest.mark.skip("Needs refactoring")
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
    if config.sampling_arguments is None:
        assert kwargs == {}
    assert isinstance(kwargs, dict)


@pytest.mark.skip("Needs refactoring")
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
    config.sampling_arguments = pipeline_params  # type: ignore
    kwargs = model_config.get_sampling_kwargs_from_config(
        config, pipeline=pipeline, fit_method=fit_method
    )
    assert kwargs == expected_kwargs


@pytest.mark.skip("Needs refactoring")
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
