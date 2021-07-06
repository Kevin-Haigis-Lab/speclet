from pathlib import Path

import pytest

from src import project_enums
from src.io import model_config
from src.managers.sbc_pipeline_resource_mangement import SBCResourceManager as RM
from src.modeling.simulation_based_calibration_enums import MockDataSizes
from src.project_enums import ModelFitMethod, ModelOption, SlurmPartitions

TEST_MODEL_NAME = "test-model"


def mock_get_configuration_for_model(*args, **kwargs) -> model_config.ModelConfig:
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


def check_resource_manager_requests(rm: RM, fit_method: ModelFitMethod):
    assert int(rm.memory) > 0
    assert rm.time is not None
    assert SlurmPartitions(rm.partition) in SlurmPartitions
    if fit_method is ModelFitMethod.ADVI:
        assert rm.cores == 1
    elif fit_method is ModelFitMethod.MCMC:
        assert rm.cores > 1
    else:
        project_enums.assert_never(fit_method)


@pytest.mark.parametrize("model", ModelOption)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("mock_data_size", MockDataSizes)
def test_resources_available_for_models(
    model: ModelOption, fit_method: ModelFitMethod, mock_data_size: MockDataSizes
):
    rm = RM(
        name=TEST_MODEL_NAME,
        mock_data_size=mock_data_size,
        fit_method=fit_method,
        config_path=Path("."),
    )
    rm.config.model = model
    check_resource_manager_requests(rm, fit_method)


@pytest.mark.parametrize("model", ModelOption)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("mock_data_size", MockDataSizes)
def test_resources_available_for_models_wrong_types(
    model: ModelOption, fit_method: ModelFitMethod, mock_data_size: MockDataSizes
):
    rm = RM(
        name=TEST_MODEL_NAME,  # type: ignore
        mock_data_size=mock_data_size.value,  # type: ignore
        fit_method=fit_method.value,  # type: ignore
        config_path=Path(".").as_posix(),  # type: ignore
    )
    rm.config.model = model
    check_resource_manager_requests(rm, fit_method)
