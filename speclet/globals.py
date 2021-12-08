"""Global constants."""

from dataclasses import dataclass

from speclet.project_config import read_project_configuration


@dataclass(frozen=True)
class Pymc3Constants:
    """PyMC3 global constants."""

    hdi_prob: float


def get_pymc3_constants() -> Pymc3Constants:
    """Get the PyMC3 global constant data.

    Returns:
        Pymc3Constants: PyMC3 global constants.
    """
    project_config = read_project_configuration()
    return Pymc3Constants(hdi_prob=project_config.modeling.highest_density_interval)
