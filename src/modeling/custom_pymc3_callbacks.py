#!/usr/bin/env python3

"""Sampling and fitting callbacks for PyMC3 models."""

from typing import Dict

import pymc3 as pm


class TooManyDivergences(Exception):
    """Raised when there have been too many divergences during MCMC sampling."""

    pass


class DivergenceFractionCallback:
    """Interrupt sampling if the proportion of divergences is too large."""

    def __init__(
        self, n_tune_steps: int, max_frac: float = 0.02, min_samples: int = 100
    ) -> None:
        """Create a 'DivergenceFractionCallback' callback.

        Args:
            n_tune_steps (int): The number of tuning steps during sampling.
            max_frac (float, optional): The maximum proportion of divergences to allow. Defaults to 0.02.
            min_samples (int, optional): The minimum number of sampling steps before interrupting. Defaults to 100.
        """
        self.n_tune_steps = n_tune_steps
        self.max_frac = max_frac
        self.min_samples = min_samples
        self.divergence_counts: Dict[int, int] = {}

    def __call__(
        self, trace: pm.backends.ndarray.NDArray, draw: pm.parallel_sampling.Draw
    ) -> None:
        """Responder to sampling callback.

        Args:
            trace (pm.backends.ndarray.NDArray): The current MCMC trace.
            draw (pm.parallel_sampling.Draw): The current MCMC draw.

        Raises:
            TooManyDivergences: Throws if the proportion of divergences it too high.
        """
        # Ignore tuning steps.
        if draw.tuning:
            return

        # Count divergences.
        current_count = 0
        if draw.stats[0]["diverging"]:
            try:
                current_count = self.divergence_counts[draw.chain] + 1
            except:
                current_count = 1
            self.divergence_counts[draw.chain] = current_count

        # Leave if not enough steps to check.
        if trace.draw_idx - self.n_tune_steps < self.min_samples:
            return

        # Check fraction of steps that were divergences.
        if current_count / trace.draw_idx >= self.max_frac:
            raise TooManyDivergences(
                f"Too many divergences: {current_count} of {trace.draw_idx} steps ({current_count / trace.draw_idx} %). Stopping early."
            )
