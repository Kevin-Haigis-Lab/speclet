from datetime import timedelta as td

import pytest

from speclet.pipelines import slurm_resources as slurm
from speclet.project_enums import SlurmPartition


@pytest.mark.parametrize("gpu", (None, False))
@pytest.mark.parametrize(
    ["time", "expt_partition"],
    [
        (td(minutes=1), SlurmPartition.SHORT),
        (td(hours=1), SlurmPartition.SHORT),
        (td(hours=12), SlurmPartition.SHORT),
        (td(hours=12, minutes=1), SlurmPartition.MEDIUM),
        (td(days=1), SlurmPartition.MEDIUM),
        (td(days=5), SlurmPartition.MEDIUM),
        (td(days=5, minutes=1), SlurmPartition.LONG),
        (td(days=15), SlurmPartition.LONG),
    ],
)
def test_determine_necessary_partition_without_gpu(
    time: td, gpu: slurm.GPUModule | bool | None, expt_partition: SlurmPartition
) -> None:
    print(gpu)
    partition = slurm.determine_necessary_partition(time, gpu)
    assert partition is expt_partition
    return None


@pytest.mark.parametrize("gpu", [True] + list(slurm.GPUModule))
@pytest.mark.parametrize(
    "time",
    [
        td(minutes=1),
        td(hours=1),
        td(hours=12),
        td(hours=12, minutes=1),
        td(days=1),
        td(days=5),
        td(days=5, minutes=1),
        td(days=15),
    ],
)
def test_determine_necessary_partition_with_gpu(
    time: td, gpu: slurm.GPUModule | bool | None
) -> None:
    print(gpu)
    partition = slurm.determine_necessary_partition(time, gpu)
    assert partition is SlurmPartition.GPU_QUAD
    return None


@pytest.mark.parametrize("gpu_mod", slurm.GPUModule)
def test_get_gres_name(gpu_mod: slurm.GPUModule) -> None:
    gres = slurm.get_gres_name(gpu_mod)
    assert gres is not None
