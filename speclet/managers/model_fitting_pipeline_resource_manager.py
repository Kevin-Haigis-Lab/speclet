"""Manage resources for the model fitting pipeline."""

from datetime import timedelta as td
from pathlib import Path
from typing import TypeVar

from pydantic import validate_arguments

from speclet import formatting
from speclet import model_configuration as model_config
from speclet.bayesian_models import BayesianModel
from speclet.exceptions import ResourceRequestUnkown
from speclet.managers import pipeline_resource_manager as prm
from speclet.project_enums import ModelFitMethod

T = TypeVar("T")
ResourceLookupDict = dict[BayesianModel, dict[bool, dict[ModelFitMethod, T]]]

# RAM required for each configuration (in GB -> mult by 1000).
#   key: [model][debug][fit_method]
MemoryLookupDict = ResourceLookupDict[int]

# Time required for each configuration.
#   key: [model][debug][fit_method]
TimeLookupDict = ResourceLookupDict[td]

fitting_pipeline_memory_lookup: MemoryLookupDict = {
    BayesianModel.SIMPLE_NEGATIVE_BINOMIAL: {
        True: {ModelFitMethod.PYMC3_ADVI: 4, ModelFitMethod.PYMC3_MCMC: 4},
        False: {ModelFitMethod.PYMC3_ADVI: 8, ModelFitMethod.PYMC3_MCMC: 8},
    },
}


# Time required for each configuration.
#   key: [model][debug][fit_method]
fitting_pipeline_time_lookup: TimeLookupDict = {
    BayesianModel.SIMPLE_NEGATIVE_BINOMIAL: {
        True: {
            ModelFitMethod.PYMC3_ADVI: td(minutes=5),
            ModelFitMethod.PYMC3_MCMC: td(minutes=5),
        },
        False: {
            ModelFitMethod.PYMC3_ADVI: td(minutes=10),
            ModelFitMethod.PYMC3_MCMC: td(minutes=10),
        },
    },
}


class ModelFittingPipelineResourceManager:
    """Resource manager for the pipeline to fit models on O2."""

    name: str
    fit_method: ModelFitMethod
    config_path: Path
    _debug: bool

    @validate_arguments
    def __init__(
        self,
        name: str,
        fit_method: ModelFitMethod,
        config_path: Path,
        debug: bool = False,
    ) -> None:
        """Create a resource manager of the model-fitting pipeline.

        Args:
            name (str): Identifiable and descriptive name of the model.
            fit_method (ModelFitMethod): Method being used to fit the model.
            config_path (Path): Path to a model configuration file.
            debug (bool, optional): In debug mode? Defaults to False.
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
        self._debug = debug

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
        return self._debug

    def is_debug_cli(self) -> str:
        """Get the correct flag for indicating debug mode through a CLI.

        Returns:
            [type]: The flag for a CLI to indicate debug status.
        """
        return "--debug" if self.debug else "--no-debug"
