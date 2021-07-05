from datetime import timedelta

import pytest

from src import formatting


@pytest.mark.parametrize(
    "input, expected",
    [
        (timedelta(days=3), "3-00:00"),
        (timedelta(days=3, hours=4), "3-04:00"),
        (timedelta(days=3, hours=4, minutes=2), "3-04:02"),
        (timedelta(days=3, hours=12), "3-12:00"),
        (timedelta(days=10, hours=15, minutes=45), "10-15:45"),
        (timedelta(days=10, hours=15, minutes=45, seconds=59), "10-15:45"),
        (timedelta(days=100, hours=15, minutes=45), "100-15:45"),
        (timedelta(days=5, hours=28), "6-04:00"),
        (timedelta(days=0, minutes=70), "0-01:10"),
    ],
)
def test_format_timedelta_slurm(input: timedelta, expected: str):
    output = formatting.format_timedelta(input, fmt=formatting.TimeDeltaFormat.SLURM)
    assert output == expected
