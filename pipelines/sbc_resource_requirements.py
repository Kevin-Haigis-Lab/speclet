"""Manage resources for the simulation-based calibration pipeline."""

from pipeline_classes import ModelFitMethod, ModelOption
from pydantic import validate_arguments


class SBCResourceManager:
    """Manage the SLURM resource request for a SBC run."""

    @validate_arguments
    def __init__(
        self,
        model: ModelOption,
        name: str,
        mock_data_size: str,
        fit_method: ModelFitMethod,
    ) -> None:
        """Create a resource manager.

        Args:
            model (str): Type of model.
            name (str): Unique, identifiable, descriptive name for the model.
            mock_data_size (str): Size of the mock data.
        """
        self.model = model
        self.name = name
        self.mock_data_size = mock_data_size
        self.fit_method = fit_method

    @property
    def memory(self) -> str:
        """Memory (RAM) request.

        Returns:
            str: Amount of RAM required.
        """
        if self.fit_method is ModelFitMethod.mcmc:
            return "3000"
        else:
            return "1600"

    @property
    def time(self) -> str:
        """Time request.

        Returns:
            str: Amount of time required.
        """
        if self.fit_method is ModelFitMethod.mcmc:
            return "02:00:00"
        else:
            return "00:15:00"

    @property
    def cores(self) -> int:
        """Number of cores to request.

        Returns:
            str: Number of cores needed for fitting.
        """
        if self.fit_method is ModelFitMethod.mcmc:
            return 4
        else:
            return 1
