"""Manage resources for the model fitting pipeline."""

from datetime import timedelta as td
from pathlib import Path

from pydantic import validate_arguments

from speclet import formatting
from speclet.pipelines import slurm_resources as slurm
from speclet.project_enums import ModelFitMethod


class SlurmResourceManager:
    """Resource manager the SLURM job scheduler."""

    @validate_arguments
    def __init__(
        self, name: str, fit_method: ModelFitMethod, config_path: Path
    ) -> None:
        """Resource manager the SLURM job scheduler.

        Args:
            name (str): Identifiable and descriptive name of the model.
            fit_method (ModelFitMethod): Method being used to fit the model.
            config_path (Path): Path to a model configuration file.
        """
        self.name = name
        self.fit_method = fit_method
        self.config_path = config_path
        self.slurm_resources = slurm.get_resource_config(
            config_path, name
        ).slurm_resources[fit_method]
        return None

    @property
    def memory(self) -> str:
        """Memory (RAM) request.

        Returns:
            str: Amount of RAM required.
        """
        return str(int(self.slurm_resources.mem * 1000))

    @property
    def time(self) -> str:
        """Time request.

        Returns:
            str: Amount of time required.
        """
        return self._format_duration_for_slurm(self._duration())

    def _duration(self) -> td:
        return td(hours=self.slurm_resources.time)

    @property
    def partition(self) -> str:
        """Partition on SLURM to request.

        Returns:
            str: The partition to request from SLURM.
        """
        if self.slurm_resources.partition is not None:
            return self.slurm_resources.partition.value
        return slurm.partition_required_for_duration(self._duration()).value

    @property
    def cores(self) -> int:
        """Number of CPU cores to request..

        Returns:
            int: Number of cores.
        """
        return self.slurm_resources.cores

    def _format_duration_for_slurm(self, duration: td) -> str:
        return formatting.format_timedelta(
            duration, fmt=formatting.TimeDeltaFormat.DRMAA
        )
