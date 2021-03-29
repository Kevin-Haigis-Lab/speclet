#!/usr/bin/env python3

"""Model protocols."""

from typing import Protocol

from src.modeling import pymc3_sampling_api as pmapi
from src.modeling.sampling_pymc3_models import SamplingArguments


class SelfSufficientModel(Protocol):
    """Protocol for class with PyMC3 models that can build and sample on their own."""

    def build_model(self) -> None:
        """Build the model and store as attribute."""
        ...

    def mcmc_sample_model(
        self, sampling_args: SamplingArguments
    ) -> pmapi.MCMCSamplingResults:
        """Fit a model with MCMC.

        Args:
            sampling_args (SamplingArguments): Arguments for the sampling procedure.
        """
        ...

    def advi_sample_model(
        self, sampling_args: SamplingArguments
    ) -> pmapi.ApproximationSamplingResults:
        """Fit a model with ADVI.

        Args:
            sampling_args (SamplingArguments): Arguments for the sampling procedure.
        """
        ...

    def run_simulation_based_calibration(self) -> None:
        """Run a round of simulation-based calibration."""
        ...
