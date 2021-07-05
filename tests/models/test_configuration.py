from pathlib import Path
from typing import List, Tuple, Type

import pytest

from src.io import model_config as mc
from src.models import configuration
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


def test_configure_model(mock_model_config: Path, tmp_path: Path):
    sp = SpecletTestModel("my-test-model", tmp_path)
    configuration.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.NONCENTERED
    assert sp.config.cov2


def test_configure_model_no_change(mock_model_config: Path, tmp_path: Path):
    sp = SpecletTestModel("my-test-model-that-doesnot-exist", tmp_path)
    configuration.configure_model(sp, config_path=mock_model_config)
    assert sp.config.some_param is MP.CENTERED
    assert not sp.config.cov2


@pytest.mark.parametrize("model_option", ModelOption)
def test_all_model_options_return_a_type(model_option: ModelOption):
    model_type = configuration.get_model_class(model_option)
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
    model_type = configuration.get_model_class(model_option)
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
):
    config = mc.get_configuration_for_model(mock_model_config, model_name)
    assert config is not None
    sp_model = configuration.instantiate_and_configure_model(
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
):
    sp_model = configuration.get_config_and_instantiate_model(
        mock_model_config,
        name=model_name,
        root_cache_dir=tmp_path,
    )
    assert isinstance(sp_model, SpecletTestModel)
    check_test_model_configurations(sp_model, model_name)
