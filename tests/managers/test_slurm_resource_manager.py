from pathlib import Path

import pytest

from speclet.managers.slurm_resource_manager import SlurmResourceManager as SlurmRM
from speclet.project_enums import ModelFitMethod

_test_config = """
- name: test-config
  description: "
    Just a test config.
  "
  active: true
  model: LINEAGE_HIERARCHICAL_NB
  data_file: DEPMAP_TEST_DATA
  model_kwargs:
    lineage: "prostate"
  sampling_kwargs:
    pymc_numpyro:
      draws: 1000
      tune: 500
      target_accept: 0.98
  slurm_resources:
    PYMC_MCMC:
      mem: 11
      time: 13
      cores: 3
    PYMC_NUMPYRO:
      mem: 7
      time: 271
      cores: 9
      gpu:
        gpu: "RTX 8000"
"""


@pytest.fixture
def example_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as file:
        file.write(_test_config)
    return config_file


@pytest.mark.parametrize(
    ["fit_method", "n_cores"],
    [(ModelFitMethod.PYMC_MCMC, 3), (ModelFitMethod.PYMC_NUMPYRO, 9)],
)
def test_resource_manager_cpus(
    example_config: Path, fit_method: ModelFitMethod, n_cores: int
) -> None:
    rm = SlurmRM(name="test-config", fit_method=fit_method, config_path=example_config)
    assert rm.cores == n_cores


@pytest.mark.parametrize(
    ["fit_method", "mem"],
    [(ModelFitMethod.PYMC_MCMC, "11000"), (ModelFitMethod.PYMC_NUMPYRO, "7000")],
)
def test_resource_manager_memory(
    example_config: Path, fit_method: ModelFitMethod, mem: str
) -> None:
    rm = SlurmRM(name="test-config", fit_method=fit_method, config_path=example_config)
    assert rm.memory == mem


@pytest.mark.parametrize(
    ["fit_method", "time"],
    [
        (ModelFitMethod.PYMC_MCMC, "13:00:00"),
        (ModelFitMethod.PYMC_NUMPYRO, "271:00:00"),
    ],
)
def test_resource_manager_time(
    example_config: Path, fit_method: ModelFitMethod, time: str
) -> None:
    rm = SlurmRM(name="test-config", fit_method=fit_method, config_path=example_config)
    assert rm.time == time


@pytest.mark.parametrize(
    ["fit_method", "partition"],
    [
        (ModelFitMethod.PYMC_MCMC, "medium"),
        (ModelFitMethod.PYMC_NUMPYRO, "gpu_quad"),
    ],
)
def test_resource_manager_partition(
    example_config: Path, fit_method: ModelFitMethod, partition: str
) -> None:
    rm = SlurmRM(name="test-config", fit_method=fit_method, config_path=example_config)
    assert rm.partition == partition


@pytest.mark.parametrize(
    ["fit_method", "gres"],
    [
        (ModelFitMethod.PYMC_MCMC, "none"),
        (ModelFitMethod.PYMC_NUMPYRO, "gpu:rtx8000:1"),
    ],
)
def test_resource_manager_gres(
    example_config: Path, fit_method: ModelFitMethod, gres: str
) -> None:
    rm = SlurmRM(name="test-config", fit_method=fit_method, config_path=example_config)
    assert rm.gres == gres
