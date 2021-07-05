"""Formatters and formatting."""

from datetime import timedelta
from enum import Enum, unique
from typing import Callable, Dict, Optional, Union


@unique
class TimeDeltaFormat(str, Enum):
    """Pre-built timedelta formatting options."""

    SLURM = "{days}-{hours:02d}:{minutes:02d}"
    DRMAA = "{hours:02d}:{minutes:02d}:00"


def _drmaa_timedelta_format_callabck(d: Dict[str, int]) -> None:
    d["hours"] = (d["days"] * 24) + d["hours"]


def format_timedelta(
    tdelta: timedelta,
    fmt: Union[str, TimeDeltaFormat],
    callback: Optional[Callable[[Dict[str, int]], None]] = None,
) -> str:
    """Format time delta object.

    Args:
        tdelta (timedelta): Time delta object.
        fmt (Union[str, TimeDeltaFormat]): Format string with keys for 'days', 'hours',
          and 'minutes'.

    Returns:
        [type]: Formatted time delta.
    """
    if isinstance(fmt, TimeDeltaFormat):
        if fmt is TimeDeltaFormat.DRMAA and callback is None:
            callback = _drmaa_timedelta_format_callabck

        fmt = fmt.value

    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    if callback is not None:
        callback(d)
    return fmt.format(**d)
