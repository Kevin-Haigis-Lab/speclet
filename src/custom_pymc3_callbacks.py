#!/usr/bin/env python3

from typing import Dict

import pymc3 as pm


class TooManyDivergences(Exception):
    """
    Raised when there have been too many divergences during MCMC sampling.
    """

    pass


class DivergenceFractionCallback:
    def __init__(
        self, n_tune_steps: int, max_frac: float = 0.02, min_samples: int = 100
    ) -> None:
        self.n_tune_steps = n_tune_steps
        self.max_frac = max_frac
        self.min_samples = min_samples
        self.divergence_counts: Dict[int, int] = {}

    def __call__(
        self, trace: pm.backends.ndarray.NDArray, draw: pm.parallel_sampling.Draw
    ) -> None:

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

        # Check fraction of steps that were diverenges.
        if current_count / trace.draw_idx >= self.max_frac:
            raise TooManyDivergences(
                f"Too many divergences: {current_count} of {trace.draw_idx} steps ({current_count / trace.draw_idx} %). Stopping early."
            )
