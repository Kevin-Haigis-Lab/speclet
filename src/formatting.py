"""Formatters and formatting."""

from datetime import timedelta
from enum import Enum, unique
from typing import Union


@unique
class TimeDeltaFormat(str, Enum):
    """Pre-built timedelta formatting options."""

    SLURM = "{days}-{hours:02d}:{minutes:02d}"


def format_timedelta(tdelta: timedelta, fmt: Union[str, TimeDeltaFormat]) -> str:
    """Format time delta object.

    Args:
        tdelta (timedelta): Time delta object.
        fmt (Union[str, TimeDeltaFormat]): Format string with keys for 'days', 'hours',
          and 'minutes'.

    Returns:
        [type]: Formatted time delta.
    """
    if isinstance(fmt, TimeDeltaFormat):
        fmt = fmt.value

    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)
