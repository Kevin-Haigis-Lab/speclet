"""Utilities for snakemake workflow: 010_010_run-crc-sampling-snakemake."""

from datetime import timedelta as td
from typing import Dict, TypeVar

from pydantic import validate_arguments

from src.pipelines.pipeline_classes import ModelOption, SlurmPartitions
from src.project_enums import ModelFitMethod

T = TypeVar("T")
ResourceLookupDict = Dict[ModelOption, Dict[bool, Dict[ModelFitMethod, T]]]

# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
sample_models_memory_lookup: ResourceLookupDict[int] = {
    ModelOption.speclet_test_model: {
        True: {ModelFitMethod.advi: 8, ModelFitMethod.mcmc: 8},
        False: {ModelFitMethod.advi: 8, ModelFitMethod.mcmc: 8},
    },
    ModelOption.crc_ceres_mimic: {
        True: {ModelFitMethod.advi: 15, ModelFitMethod.mcmc: 20},
        False: {ModelFitMethod.advi: 20, ModelFitMethod.mcmc: 40},
    },
    ModelOption.speclet_one: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 30},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
    ModelOption.speclet_two: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 30},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
    ModelOption.speclet_three: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 60},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
    ModelOption.speclet_four: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 60},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
    ModelOption.speclet_five: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 60},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
    ModelOption.speclet_six: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 60},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
    ModelOption.speclet_seven: {
        True: {ModelFitMethod.advi: 7, ModelFitMethod.mcmc: 60},
        False: {ModelFitMethod.advi: 40, ModelFitMethod.mcmc: 150},
    },
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
sample_models_time_lookup: ResourceLookupDict[td] = {
    ModelOption.speclet_test_model: {
        True: {
            ModelFitMethod.advi: td(minutes=5),
            ModelFitMethod.mcmc: td(minutes=5),
        },
        False: {
            ModelFitMethod.advi: td(minutes=10),
            ModelFitMethod.mcmc: td(minutes=10),
        },
    },
    ModelOption.crc_ceres_mimic: {
        True: {
            ModelFitMethod.advi: td(minutes=30),
            ModelFitMethod.mcmc: td(minutes=30),
        },
        False: {ModelFitMethod.advi: td(hours=3), ModelFitMethod.mcmc: td(hours=6)},
    },
    ModelOption.speclet_one: {
        True: {ModelFitMethod.advi: td(minutes=30), ModelFitMethod.mcmc: td(hours=8)},
        False: {ModelFitMethod.advi: td(hours=12), ModelFitMethod.mcmc: td(days=2)},
    },
    ModelOption.speclet_two: {
        True: {ModelFitMethod.advi: td(minutes=30), ModelFitMethod.mcmc: td(hours=8)},
        False: {ModelFitMethod.advi: td(hours=12), ModelFitMethod.mcmc: td(days=2)},
    },
    ModelOption.speclet_three: {
        True: {ModelFitMethod.advi: td(minutes=30), ModelFitMethod.mcmc: td(days=1)},
        False: {ModelFitMethod.advi: td(hours=10), ModelFitMethod.mcmc: td(days=2)},
    },
    ModelOption.speclet_four: {
        True: {ModelFitMethod.advi: td(hours=3), ModelFitMethod.mcmc: td(days=1)},
        False: {ModelFitMethod.advi: td(hours=10), ModelFitMethod.mcmc: td(days=2)},
    },
    ModelOption.speclet_five: {
        True: {ModelFitMethod.advi: td(hours=3), ModelFitMethod.mcmc: td(days=1)},
        False: {ModelFitMethod.advi: td(hours=10), ModelFitMethod.mcmc: td(days=2)},
    },
    ModelOption.speclet_six: {
        True: {ModelFitMethod.advi: td(hours=3), ModelFitMethod.mcmc: td(days=1)},
        False: {ModelFitMethod.advi: td(hours=10), ModelFitMethod.mcmc: td(days=2)},
    },
    ModelOption.speclet_seven: {
        True: {ModelFitMethod.advi: td(hours=3), ModelFitMethod.mcmc: td(days=1)},
        False: {ModelFitMethod.advi: td(hours=10), ModelFitMethod.mcmc: td(days=2)},
    },
}


class ResourceRequestUnkown(NotImplementedError):
    """Exception raised when a resource request cannot be fullfilled."""

    pass


class ModelFittingPipelineResourceManager:
    """Resource manager for the pipeline to fit models on O2."""

    model: ModelOption
    name: str
    fit_method: ModelFitMethod
    debug: bool

    @validate_arguments
    def __init__(
        self,
        model: ModelOption,
        name: str,
        fit_method: ModelFitMethod,
        debug: bool,
    ) -> None:
        """Create a resource manager of the model-fitting pipeline.

        Args:
            model (ModelOption): Type of model being bit.
            name (str): Identifiable and descriptive name of the model.
            fit_method (ModelFitMethod): Method being used to fit the model.
            debug (bool): Is the model in debug mode?
        """
        self.model = model
        self.name = name
        self.fit_method = fit_method
        self.debug = debug

    @property
    def memory(self) -> str:
        """Memory (RAM) request.

        Returns:
            str: Amount of RAM required.
        """
        try:
            return self._retrieve_memory_requirement()
        except Exception:
            raise ResourceRequestUnkown(
                f"Unable to request memory from '{self.model.value}'."
            )

    @property
    def time(self) -> str:
        """Time request.

        Returns:
            str: Amount of time required.
        """
        try:
            duration = self._retrieve_time_requirement()
            return self._format_duration_for_slurm(duration)
        except Exception:
            raise ResourceRequestUnkown(
                f"No time known for model '{self.model.value}'."
            )

    @property
    def partition(self) -> str:
        """Partition on SLURM to request.

        Returns:
            str: The partition to request from SLURM.
        """
        return self._decide_partition_requirement().value

    def _decide_partition_requirement(self) -> SlurmPartitions:
        """Time O2 partition for the `sample_models` step."""
        duration = self._retrieve_time_requirement()
        if duration <= td(hours=12):
            return SlurmPartitions.short
        elif duration <= td(days=5):
            return SlurmPartitions.medium
        else:
            return SlurmPartitions.long

    def _retrieve_memory_requirement(self) -> str:
        return str(
            sample_models_memory_lookup[self.model][self.debug][self.fit_method] * 1000
        )

    def _retrieve_time_requirement(self) -> td:
        return sample_models_time_lookup[self.model][self.debug][self.fit_method]

    def _format_duration_for_slurm(self, duration: td) -> str:
        return str(duration).replace(" day, ", "-").replace(" days, ", "-")

    def is_debug(self) -> bool:
        """Determine the debug status of model name.

        Returns:
            bool: Whether or not the model should be in debug mode.
        """
        return "debug" in self.name

    def cli_is_debug(self):
        """Get the correct flag for indicating debug mode through a CLI.

        Returns:
            [type]: The flag for a CLI to indicate debug status.
        """
        return "--debug" if self.is_debug() else "--no-debug"
