import pytest

from src.managers.model_fitting_pipeline_resource_manager import (
    ModelFittingPipelineResourceManager as RM,
)
from src.pipelines.pipeline_classes import ModelOption
from src.project_enums import ModelFitMethod


@pytest.mark.parametrize("model", [a.value for a in ModelOption])
@pytest.mark.parametrize("fit_method", [a.value for a in ModelFitMethod])
@pytest.mark.parametrize("debug", [True, False])
def test_resources_available_for_models(model: str, fit_method: str, debug: bool):
    rm = RM(
        model=model,  # type: ignore
        name="test-model",
        fit_method=fit_method,  # type: ignore
        debug=debug,
    )
    assert int(rm.memory) > 0
    assert rm.time is not None
