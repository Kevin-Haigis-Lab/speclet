import pytest

from src.managers.sbc_pipeline_resource_mangement import SBCResourceManager as RM
from src.modeling.simulation_based_calibration_enums import MockDataSizes
from src.project_enums import ModelFitMethod, ModelOption


@pytest.mark.parametrize("model", [a.value for a in ModelOption])
@pytest.mark.parametrize("fit_method", [a.value for a in ModelFitMethod])
@pytest.mark.parametrize("data_size", [a.value for a in MockDataSizes])
def test_resources_available_for_models_string_inputs(
    model: str, fit_method: str, data_size: str
):
    rm = RM(
        model=model,  # type: ignore
        name="test-model",
        mock_data_size=data_size,  # type: ignore
        fit_method=fit_method,  # type: ignore
    )
    assert int(rm.memory) > 0
    assert rm.time is not None
    assert rm.cores >= 1


@pytest.mark.parametrize("model", ModelOption)
@pytest.mark.parametrize("fit_method", ModelFitMethod)
@pytest.mark.parametrize("data_size", MockDataSizes)
def test_resources_available_for_models(
    model: ModelOption, fit_method: ModelFitMethod, data_size: MockDataSizes
):
    rm = RM(
        model=model,  # type: ignore
        name="test-model",
        mock_data_size=data_size,  # type: ignore
        fit_method=fit_method,  # type: ignore
    )
    assert int(rm.memory) > 0
    assert rm.time is not None
    assert rm.cores >= 1
