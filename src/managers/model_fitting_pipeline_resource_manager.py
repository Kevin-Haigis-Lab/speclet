"""Utilities for snakemake workflow: 010_010_run-crc-sampling-snakemake."""

from datetime import timedelta as td
from typing import Dict, TypeVar

from pydantic import validate_arguments

from src.exceptions import ResourceRequestUnkown
from src.pipelines.pipeline_classes import ModelOption, SlurmPartitions
from src.project_enums import ModelFitMethod

T = TypeVar("T")
ResourceLookupDict = Dict[ModelOption, Dict[bool, Dict[ModelFitMethod, T]]]

# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
sample_models_memory_lookup: ResourceLookupDict[int] = {
    ModelOption.speclet_test_model: {
        True: {ModelFitMethod.ADVI: 8, ModelFitMethod.MCMC: 8},
        False: {ModelFitMethod.ADVI: 8, ModelFitMethod.MCMC: 8},
    },
    ModelOption.crc_ceres_mimic: {
        True: {ModelFitMethod.ADVI: 15, ModelFitMethod.MCMC: 20},
        False: {ModelFitMethod.ADVI: 20, ModelFitMethod.MCMC: 40},
    },
    ModelOption.speclet_one: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 30},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.speclet_two: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 30},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.speclet_four: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.speclet_five: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.speclet_six: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
    ModelOption.speclet_seven: {
        True: {ModelFitMethod.ADVI: 7, ModelFitMethod.MCMC: 60},
        False: {ModelFitMethod.ADVI: 40, ModelFitMethod.MCMC: 150},
    },
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
sample_models_time_lookup: ResourceLookupDict[td] = {
    ModelOption.speclet_test_model: {
        True: {ModelFitMethod.ADVI: td(minutes=5), ModelFitMethod.MCMC: td(minutes=5)},
        False: {
            ModelFitMethod.ADVI: td(minutes=10),
            ModelFitMethod.MCMC: td(minutes=10),
        },
    },
    ModelOption.crc_ceres_mimic: {
        True: {
            ModelFitMethod.ADVI: td(minutes=30),
            ModelFitMethod.MCMC: td(minutes=30),
        },
        False: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(hours=6)},
    },
    ModelOption.speclet_one: {
        True: {ModelFitMethod.ADVI: td(minutes=30), ModelFitMethod.MCMC: td(hours=8)},
        False: {ModelFitMethod.ADVI: td(hours=12), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.speclet_two: {
        True: {ModelFitMethod.ADVI: td(minutes=30), ModelFitMethod.MCMC: td(hours=8)},
        False: {ModelFitMethod.ADVI: td(hours=12), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.speclet_four: {
        True: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.speclet_five: {
        True: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.speclet_six: {
        True: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
    ModelOption.speclet_seven: {
        True: {ModelFitMethod.ADVI: td(hours=3), ModelFitMethod.MCMC: td(days=1)},
        False: {ModelFitMethod.ADVI: td(hours=10), ModelFitMethod.MCMC: td(days=2)},
    },
}


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
    ) -> None:
        """Create a resource manager of the model-fitting pipeline.

        Args:
            model (ModelOption): Type of model being bit.
            name (str): Identifiable and descriptive name of the model.
            fit_method (ModelFitMethod): Method being used to fit the model.
        """
        self.model = model
        self.name = name
        self.fit_method = fit_method
        self.debug = self._is_debug()

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
        try:
            mem = sample_models_memory_lookup[self.model][self.debug][self.fit_method]
            return str(mem * 1000)
        except KeyError as err:
            raise ResourceRequestUnkown("memory", err.args[0])

    def _retrieve_time_requirement(self) -> td:
        try:
            return sample_models_time_lookup[self.model][self.debug][self.fit_method]
        except KeyError as err:
            raise ResourceRequestUnkown("time", err.args[0])

    def _format_duration_for_slurm(self, duration: td) -> str:
        return str(duration).replace(" day, ", "-").replace(" days, ", "-")

    def _is_debug(self) -> bool:
        """Determine the debug status of model name.

        Returns:
            bool: Whether or not the model should be in debug mode.
        """
        return "debug" in self.name

    def is_debug_cli(self):
        """Get the correct flag for indicating debug mode through a CLI.

        Returns:
            [type]: The flag for a CLI to indicate debug status.
        """
        return "--debug" if self.debug else "--no-debug"
