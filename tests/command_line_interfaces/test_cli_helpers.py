from pathlib import Path
from typing import Any, List, Tuple, Type

import pytest

from src.command_line_interfaces import cli_helpers
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


def test_clean_model_names():
    assert cli_helpers.clean_model_names("model_name") == "model_name"
    assert cli_helpers.clean_model_names("model name") == "model-name"
    assert cli_helpers.clean_model_names("model named Jerry") == "model-named-Jerry"


@pytest.mark.parametrize("model_option", ModelOption)
def test_all_model_options_return_a_type(model_option: ModelOption):
    model_type = cli_helpers.get_model_class(model_option)
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
    model_type = cli_helpers.get_model_class(model_option)
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
    monkeypatch.setattr(cli_helpers, "configure_model", mock_configure_model)
    sp_model = cli_helpers.instantiate_and_configure_model(
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
    sp_model = cli_helpers.instantiate_and_configure_model(
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
