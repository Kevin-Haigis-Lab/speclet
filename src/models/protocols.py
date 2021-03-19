#!/usr/bin/env python3

"""Model protocols."""

from typing import Protocol


class SelfSufficientModel(Protocol):
    """Protocol for class with PyMC3 models that can build and sample on their own."""

    def build_model(self) -> None:
        """Build the model and store as attribute."""
        ...

    def sample_model(self) -> None:
        """Sample from the model."""
        ...

    def run_simulation_based_calibration(self) -> None:
        """Run a round of simulation-based calibration."""
        ...
