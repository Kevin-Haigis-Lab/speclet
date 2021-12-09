"""Common types used throughout the project."""

from typing import Literal, Union

BasicTypes = Union[float, str, int, bool, None]

VIMethod = Literal["advi", "fullrank_advi", "svgd", "asvgd", "nfvi", "nfv"]
