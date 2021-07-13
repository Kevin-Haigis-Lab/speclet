"""Utilities for snakemake workflow: 010_010_run-crc-sampling-snakemake."""

from datetime import timedelta as td
from pathlib import Path
from typing import Dict, TypeVar

from pydantic import validate_arguments

from src import formatting
from src.exceptions import ResourceRequestUnkown
from src.io import model_config
from src.managers import pipeline_resource_manager as prm
from src.managers.pipeline_resource_manager import PipelineResourceManager
from src.project_enums import ModelFitMethod, ModelOption

T = TypeVar("T")
ResourceLookupDict = Dict[ModelOption, Dict[bool, Dict[ModelFitMethod, T]]]

# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
MemoryLookupDict = ResourceLookupDict[int]

# Time required for each configuration.
#   key: [model][debug][fit_method]
TimeLookupDict = ResourceLookupDict[td]

fitting_pipeline_memory_lookup: MemoryLookupDict = {
    ModelOption.SPECLET_TEST_MODEL: {
        True: {ModelFitMethod.ADVI: 8, ModelFitMethod.MCMC: 8},
        False: {ModelFitMethod.ADVI: 8, ModelFitMethod.MCMC: 8},
    },
    ModelOption.CRC_CERES_MIMIC: {
        True: {ModelFitMethod.ADVI: 15, ModelFitMethod.MCMC: 20},
        False: {ModelFitMethod.ADVI: 20, ModelFitMethod.MCMC: 40},
    },
    ModelOption.SPECLET_ONE: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 30},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.SPECLET_TWO: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 30},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.SPECLET_FOUR: {
        True: {ModelFitMethod.ADVI: 4, ModelFitMethod.MCMC: 8},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.SPECLET_FIVE: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.SPECLET_SIX: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.SPECLET_SEVEN: {
        True: {ModelFitMethod.ADVI: 6, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
fitting_pipeline_time_lookup: TimeLookupDict = {
    ModelOption.SPECLET_TEST_MODEL: {
        True: {ModelFitMethod.ADVI: td(minutes=5), ModelFitMethod.MCMC: td(minutes=5)},
        False: {
            ModelFitMethod.ADVI: td(minutes=10),
            ModelFitMethod.MCMC: td(minutes=10),
        },
    },
    ModelOption.CRC_CERES_MIMIC: {
        True: {
            ModelFitMethod.ADVI: td(minutes=30),
            ModelFitMethod.MCMC: td(minutes=30),
        },
        False: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(hours=6)},
    },
    ModelOption.SPECLET_ONE: {
        True: {ModelFitMethod.ADVI: td(minutes=30), ModelFitMethod.MCMC: td(hours=8)},
        False: {ModelFitMethod.ADVI: td(hours=12), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.SPECLET_TWO: {
        True: {ModelFitMethod.ADVI: td(minutes=30), ModelFitMethod.MCMC: td(hours=8)},
        False: {ModelFitMethod.ADVI: td(hours=12), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.SPECLET_FOUR: {
        True: {ModelFitMethod.ADVI: td(minutes=30), ModelFitMethod.MCMC: td(hours=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.SPECLET_FIVE: {
        True: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.SPECLET_SIX: {
        True: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.SPECLET_SEVEN: {
        True: {ModelFitMethod.ADVI: td(hours=12), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
}


class ModelFittingPipelineResourceManager(PipelineResourceManager):
    """Resource manager for the pipeline to fit models on O2."""

    name: str
    fit_method: ModelFitMethod
    config_path: Path
    config: model_config.ModelConfig

    @validate_arguments
    def __init__(
        self, name: str, fit_method: ModelFitMethod, config_path: Path
    ) -> None:
        """Create a resource manager of the model-fitting pipeline.

        Args:
            name (str): Identifiable and descriptive name of the model.
            fit_method (ModelFitMethod): Method being used to fit the model.
            config_path (Path): Path to a model configuration file.
        """
        self.name = name
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
        return self._retrieve_memory_requirement()

    @property
    def time(self) -> str:
        """Time request.

        Returns:
            str: Amount of time required.
        """
        duration = self._retrieve_time_requirement()
        return self._format_duration_for_slurm(duration)

    @property
    def partition(self) -> str:
        """Partition on SLURM to request.

        Returns:
            str: The partition to request from SLURM.
        """
        return prm.slurm_partition_required_for_duration(
            self._retrieve_time_requirement()
        ).value

    @property
    def cores(self) -> int:
        """Compute cores request.

        Returns:
            int: Number of cores.
        """
        return 1

    def _retrieve_memory_requirement(self) -> str:
        try:
            mem = fitting_pipeline_memory_lookup[self.config.model][self.debug][
                self.fit_method
            ]
            return str(mem * 1000)
        except KeyError as err:
            raise ResourceRequestUnkown("memory", err.args[0])

    def _retrieve_time_requirement(self) -> td:
        try:
            return fitting_pipeline_time_lookup[self.config.model][self.debug][
                self.fit_method
            ]
        except KeyError as err:
            raise ResourceRequestUnkown("time", err.args[0])

    def _format_duration_for_slurm(self, duration: td) -> str:
        return formatting.format_timedelta(
            duration, fmt=formatting.TimeDeltaFormat.DRMAA
        )

    @property
    def debug(self) -> bool:
        """Determine the debug status of model name.

        Returns:
            bool: Whether or not the model should be in debug mode.
        """
        return self.config.debug

    def is_debug_cli(self):
        """Get the correct flag for indicating debug mode through a CLI.

        Returns:
            [type]: The flag for a CLI to indicate debug status.
        """
        return "--debug" if self.debug else "--no-debug"
