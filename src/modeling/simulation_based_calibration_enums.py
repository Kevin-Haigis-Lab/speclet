"""Enums for simulation based calibration."""

from enum import Enum, unique


@unique
class MockDataSizes(str, Enum):
    """Options for dataset seizes when generating mock data."""

    small = "small"
    medium = "medium"
    large = "large"
