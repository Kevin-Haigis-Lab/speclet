"""Manage resources for the simulation-based calibration pipeline."""

from datetime import timedelta as td
from pathlib import Path
from typing import Dict, TypeVar

from pydantic import validate_arguments

from src.io import model_config
from src.managers import pipeline_resource_manager as prm
from src.managers.pipeline_resource_manager import PipelineResourceManager
from src.modeling.simulation_based_calibration_enums import MockDataSizes
from src.project_enums import ModelFitMethod, ModelOption

T = TypeVar("T")
ResourceLookupDict = Dict[ModelFitMethod, Dict[MockDataSizes, T]]
ModelResourceLookupDict = Dict[ModelOption, ResourceLookupDict[T]]

# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
MemoryLookupDict = ModelResourceLookupDict[int]

# Time required for each configuration.
#   key: [model][debug][fit_method]
TimeLookupDict = ModelResourceLookupDict[td]


sbc_pipeline_memory_lookup: MemoryLookupDict = {
    ModelOption.SPECLET_FOUR: {
        ModelFitMethod.ADVI: {
            MockDataSizes.SMALL: 4,
            MockDataSizes.MEDIUM: 8,
            MockDataSizes.LARGE: 16,
        },
        ModelFitMethod.MCMC: {
            MockDataSizes.SMALL: 8,
            MockDataSizes.MEDIUM: 16,
            MockDataSizes.LARGE: 32,
        },
    }
}

sbc_pipeline_time_lookup: TimeLookupDict = {
    ModelOption.SPECLET_FOUR: {
        ModelFitMethod.ADVI: {
            MockDataSizes.SMALL: td(minutes=15),
            MockDataSizes.MEDIUM: td(hours=1),
            MockDataSizes.LARGE: td(hours=8),
        },
        ModelFitMethod.MCMC: {
            MockDataSizes.SMALL: td(hours=1),
            MockDataSizes.MEDIUM: td(hours=8),
            MockDataSizes.LARGE: td(hours=12),
        },
    }
}


class SBCResourceManager(PipelineResourceManager):
    """Manage the SLURM resource request for a SBC run."""

    @validate_arguments
    def __init__(
        self,
        name: str,
        mock_data_size: MockDataSizes,
        fit_method: ModelFitMethod,
        config_path: Path,
    ) -> None:
        """Create a resource manager.

        Args:
            name (str): Unique, identifiable, descriptive name for the model.
            mock_data_size (str): Size of the mock data.
            fit_method (ModelFitMethod): Method used to fit the model.
            config_path (Path): Path to a model configuration file.
        """
        self.name = name
        self.mock_data_size = mock_data_size
        self.fit_method = fit_method
        self.config_path = config_path
        _config = model_config.get_configuration_for_model(
            self.config_path, name=self.name
        )
        if _config is None:
            raise model_config.ModelConfigurationNotFound(self.name)
        self.config = _config

    @property
    def memory(self) -> str:
        """Memory (RAM) request.

        Returns:
            str: Amount of RAM required.
        """
        return str(self._retrieve_memory_requirement() * 1000)

    @property
    def time(self) -> str:
        """Time request.

        Returns:
            str: Amount of time required.
        """
        if self.fit_method is ModelFitMethod.MCMC:
            return "02:00:00"
        else:
            return "00:15:00"

    def _retrieve_memory_requirement(self) -> int:
        default_memory_tbl: ResourceLookupDict[int] = {
            ModelFitMethod.ADVI: {
                MockDataSizes.SMALL: 4,
                MockDataSizes.MEDIUM: 8,
                MockDataSizes.LARGE: 16,
            },
            ModelFitMethod.MCMC: {
                MockDataSizes.SMALL: 4,
                MockDataSizes.MEDIUM: 16,
                MockDataSizes.LARGE: 32,
            },
        }
        return self._lookup_value_with_default(
            sbc_pipeline_memory_lookup, default_memory_tbl
        )

    def _retrieve_time_requirement(self) -> td:
        default_time_tbl: ResourceLookupDict[td] = {
            ModelFitMethod.ADVI: {
                MockDataSizes.SMALL: td(minutes=15),
                MockDataSizes.MEDIUM: td(hours=1),
                MockDataSizes.LARGE: td(hours=8),
            },
            ModelFitMethod.MCMC: {
                MockDataSizes.SMALL: td(hours=1),
                MockDataSizes.MEDIUM: td(hours=8),
                MockDataSizes.LARGE: td(hours=12),
            },
        }
        return self._lookup_value_with_default(
            sbc_pipeline_time_lookup, default_time_tbl
        )

    def _lookup_value_with_default(
        self,
        primary_tbl: ModelResourceLookupDict[T],
        default_tbl: ResourceLookupDict[T],
    ) -> T:
        try:
            return primary_tbl[self.config.model][self.fit_method][self.mock_data_size]
        except KeyError:
            return default_tbl[self.fit_method][self.mock_data_size]

    @property
    def cores(self) -> int:
        """Number of cores to request.

        Returns:
            str: Number of cores needed for fitting.
        """
        if self.fit_method is ModelFitMethod.MCMC:
            return 4
        else:
            return 1

    @property
    def partition(self) -> str:
        """Partition on SLURM to request.

        Returns:
            str: The partition to request from SLURM.
        """
        return prm.slurm_partition_required_for_duration(
            self._retrieve_time_requirement()
        ).value
