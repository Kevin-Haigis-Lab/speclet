from pathlib import Path
from typing import Any

import pytest

from speclet import model_configuration as model_config
from speclet.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from speclet.managers.model_fitting_pipeline_resource_manager import (
    ResourceRequestUnkown,
)
from speclet.project_enums import ModelFitMethod, ModelOption, SlurmPartitions

TEST_MODEL_NAME = "test-model"


def mock_get_configuration_for_model(
    *args: Any, **kwargs: Any
) -> model_config.ModelConfig:
    return model_config.ModelConfig(
        name=TEST_MODEL_NAME,
        description="A model for testing model fitting resource manager.",
        model=ModelOption.SPECLET_TEST_MODEL,
        fit_methods=[],
        config={},
        pipelines=[],
        debug=True,
    )


@pytest.fixture(autouse=True)
def get_mock_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_config, "get_configuration_for_model", mock_get_configuration_for_model
    )


@pytest.mark.parametrize("model", ModelOption)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("debug", (True, False))
def test_resources_available_for_models(
    model: ModelOption, fit_method: ModelFitMethod, debug: bool
) -> None:
    rm = RM(name=TEST_MODEL_NAME, fit_method=fit_method, config_path=Path("."))
    rm.config.model = model

    assert int(rm.memory) > 0
    assert rm.time is not None
    assert SlurmPartitions(rm.partition) in SlurmPartitions


@pytest.mark.parametrize("fit_method", [a.value for a in ModelFitMethod])
def test_create_resource_manager_with_wrong_types(fit_method: str) -> None:
    rm = RM(
        name=TEST_MODEL_NAME,
        fit_method=fit_method,
        config_path="some-directory/not-a-real-file.txt",
    )

    assert rm.config.name == TEST_MODEL_NAME
    assert rm.config.model is ModelOption.SPECLET_TEST_MODEL

    assert int(rm.memory) > 0
    assert rm.time is not None
    assert SlurmPartitions(rm.partition) in SlurmPartitions


@pytest.mark.parametrize("debug", (True, False))
def test_resource_manager_detects_debug(debug: bool) -> None:
    rm = RM(
        name="my-model-debug",
        fit_method=ModelFitMethod.ADVI,
        config_path=Path("some-directory/not-a-real-file.txt"),
        debug=debug,
    )

    if debug:
        assert rm.debug
        assert rm.is_debug_cli() == "--debug"
    else:
        assert not rm.debug
        assert rm.is_debug_cli() == "--no-debug"


def test_error_on_invalid_model_params() -> None:
    rm = RM(name=TEST_MODEL_NAME, fit_method=ModelFitMethod.ADVI, config_path=Path("."))

    # Have to assign afterwards because of pydantic validation.
    rm.config.model = "SomeRandomModel"

    with pytest.raises(ResourceRequestUnkown):
        rm.memory
    with pytest.raises(ResourceRequestUnkown):
        rm.time
    with pytest.raises(ResourceRequestUnkown):
        rm.partition
