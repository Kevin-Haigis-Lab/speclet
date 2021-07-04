from pathlib import Path
from typing import Any, List, Tuple, Type

import pytest

from src.models import configuration as c
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
from src.project_enums import ModelOption
from src.project_enums import ModelParameterization as MP


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


@pytest.mark.parametrize("model_option", ModelOption)
def test_all_model_options_return_a_type(model_option: ModelOption):
    model_type = c.get_model_class(model_option)
    assert model_type is not None


model_options_expected_class_parameterization: List[
    Tuple[ModelOption, Type[SpecletModel]]
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
def test_get_model_class(model_option: ModelOption, expected_class: Type):
    model_type = c.get_model_class(model_option)
    assert model_type == expected_class


def mock_configure_model(*args: Any, **kwargs: Any) -> None:
    return None


@pytest.mark.parametrize(
    "model_option, expected_class", model_options_expected_class_parameterization
)
def test_instantiate_model(
    model_option: ModelOption,
    expected_class: Type,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(c, "configure_model", mock_configure_model)
    sp_model = c.instantiate_and_configure_model(
        model_option,
        name="TEST-MODEL",
        root_cache_dir=tmp_path,
        debug=True,
        config_path=Path("."),
    )
    assert isinstance(sp_model, expected_class)


@pytest.mark.parametrize("model_name", ["my-test-model", "second-test-model"])
def test_instantiate_and_configure_model(
    model_name: str,
    tmp_path: Path,
):
    sp_model = c.instantiate_and_configure_model(
        ModelOption.SPECLET_TEST_MODEL,
        name=model_name,
        root_cache_dir=tmp_path,
        debug=True,
        config_path=Path("tests/models/mock-model-config.yaml"),
    )

    assert isinstance(sp_model, SpecletTestModel)

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
