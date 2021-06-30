import pytest

from src.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from src.pipelines.pipeline_classes import ModelOption
from src.project_enums import ModelFitMethod


@pytest.mark.parametrize("model", [a.value for a in ModelOption])
@pytest.mark.parametrize("fit_method", [a.value for a in ModelFitMethod])
def test_resources_available_for_models(model: str, fit_method: str):
    rm = RM(
        model=model,  # type: ignore
        name="test-model",
        fit_method=fit_method,  # type: ignore
    )
    assert int(rm.memory) > 0
    assert rm.time is not None


def test_resource_manager_detects_debug():
    rm = RM(
        model=ModelOption.speclet_test_model,
        name="my-model-debug",
        fit_method=ModelFitMethod.ADVI,
    )
    assert rm.debug
    assert rm.is_debug_cli() == "--debug"

    rm = RM(
        model=ModelOption.speclet_test_model,
        name="my-model",
        fit_method=ModelFitMethod.ADVI,
    )
    assert not rm.debug
    assert rm.is_debug_cli() == "--no-debug"
