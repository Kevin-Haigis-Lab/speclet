"""Interactions with SLURM from within Python."""

import os
import subprocess
from enum import Enum

from speclet.loggers import logger


class SlurmEnvVariables(Enum):
    """Slurm environment variables."""

    SLURM_JOB_ID = "SLURM_JOB_ID"


def get_job_id() -> str | None:
    """Get the current Slurm job ID."""
    return os.getenv(SlurmEnvVariables.SLURM_JOB_ID.value)


def cancel_current_slurm_job() -> None:
    """Cancel the current Slurm job."""
    if (job_id := get_job_id()) is None:
        logger.info("Could not find job ID.")
        return None
    logger.info(f"Attempting to cancel current job (ID {job_id})")
    cmd = ["scancel", job_id]
    res = subprocess.run(cmd, capture_output=True)
    logger.info(f"response code: {res.returncode}.")
    logger.info(f"stdout: {str(res.stdout)}")
    logger.info(f"stderr: {str(res.stderr)}")
    return None
