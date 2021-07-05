"""Enums for simulation based calibration."""

from enum import Enum, unique


@unique
class MockDataSizes(str, Enum):
    """Options for dataset seizes when generating mock data."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
