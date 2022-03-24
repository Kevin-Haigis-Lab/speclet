"""Sampling and fitting callbacks for PyMC3 models."""

from collections import defaultdict
from datetime import datetime
from typing import DefaultDict

from pymc.backends.ndarray import NDArray
from pymc.parallel_sampling import Draw


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
            max_frac (float, optional): The maximum proportion of divergences to allow.
              Defaults to 0.02.
            min_samples (int, optional): The minimum number of sampling steps before
              interrupting. Defaults to 100.
        """
        self.n_tune_steps = n_tune_steps
        self.max_frac = max_frac
        self.min_samples = min_samples
        self.divergence_counts: dict[int, int] = {}

    def __call__(self, trace: NDArray, draw: Draw) -> None:
        """Responder to sampling callback.

        Args:
            trace (NDArray): The current MCMC trace.
            draw (Draw): The current MCMC draw.

        Raises:
            TooManyDivergences: Throws if the proportion of divergences it too high.
        """
        # Ignore tuning steps.
        if draw.tuning:
            return

        # Count divergences.
        current_count = 0
        if draw.stats[0]["diverging"]:
            current_count = self.divergence_counts.get(draw.chain, 0) + 1
            self.divergence_counts[draw.chain] = current_count

        # Leave if not enough steps to check.
        if trace.draw_idx - self.n_tune_steps < self.min_samples:
            return

        # Check fraction of steps that were divergences.
        if current_count / trace.draw_idx >= self.max_frac:
            msg = f"Too many divergences: {current_count} of {trace.draw_idx} "
            msg += f"steps ({current_count / trace.draw_idx} %). Stopping early."
            raise TooManyDivergences(msg)


def _print_draw_table(draw: Draw, n_divergences: int) -> None:
    tuning = "tune" if draw.tuning else "sampling"
    msg = ", ".join(
        [
            f"{str(datetime.now())}",
            f"chain={draw.chain}",
            f"draw={draw.draw_idx}",
            tuning,
            f"divergences={n_divergences}",
        ]
    )
    return print(msg)


class ProgressPrinterCallback:
    """A simpler replacement to the progress bar for use in log files.

    With this callback, it is recommended to set `progressbar=False` in
    `pymc.sample()`.
    """

    every_n: int
    tuning: bool
    _num_divergences: DefaultDict[int, int]

    def __init__(self, every_n: int = 50, tuning: bool = True) -> None:
        """Create a ProgressPrinterCallback object.

        Args:
            every_n (int, optional): Print updates every `n` draws. Defaults to 50.
            tuning (bool, optional): Should tuning draws be included? Defaults to True.
        """
        self.every_n = every_n
        self.tuning = tuning
        self._num_divergences = defaultdict(lambda: 0)
        return None

    def __call__(self, trace: NDArray, draw: Draw) -> None:
        """Responder to sampling callback.

        Args:
            trace (NDArray): The current MCMC trace.
            draw (Draw): The current MCMC draw.
        """
        if not self.tuning and draw.tuning:
            return None

        if not draw.tuning:
            # Do not count divergences during tuning.
            self._num_divergences[draw.chain] += draw.stats[0]["diverging"]

        if draw.draw_idx % self.every_n == 0 or draw.is_last:
            _print_draw_table(draw, n_divergences=self._num_divergences[draw.chain])

        return None
