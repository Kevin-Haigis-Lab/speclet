"""Functions to help using Stan."""

from pathlib import Path


def read_code_file(stan_file: Path) -> str:
    """Read in a Stan code file to a string.

    Args:
        stan_file (Path): Path to the stan code.

    Returns:
        str: Stan code as a string.
    """
    with open(stan_file, "r") as file:
        code = "".join(list(file))
    return code
