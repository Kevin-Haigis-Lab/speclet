"""Define a protocol for pipeline resource managers."""

from datetime import timedelta as td
from typing import Protocol

from src.project_enums import SlurmPartitions


class PipelineResourceManager(Protocol):
    """Protocol for pipeline resource managers."""

    memory: str
    time: str
    cores: int
    partition: str


#### ---- Common functions ---- ####


def slurm_partition_required_for_duration(time_req: td) -> SlurmPartitions:
    """SLURM partition best suited for a job with a given time requirement.

    Args:
        time_req (td): Time that will be requested for the job.

    Returns:
        SlurmPartitions: The SLURM partition most appropriate for the time.
    """
    if time_req <= td(hours=12):
        return SlurmPartitions.SHORT
    elif time_req <= td(days=5):
        return SlurmPartitions.MEDIUM
    else:
        return SlurmPartitions.LONG
