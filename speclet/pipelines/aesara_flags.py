"""Aesara flags for pipelines."""

import os
from typing import Optional

import aesara


def get_aesara_compile_dir() -> str:
    """Get the Aesara compilation directory from the config.

    Returns:
        str: String of path the compilation directory.
    """
    compile_dir: str = aesara.config.compiledir
    return compile_dir


def _aesara_gcc_config() -> Optional[str]:
    return os.getenv("AESARA_GCC_FLAG")


def get_aesara_flags(
    unique_id: str, aesara_compiledir: Optional[str] = None, gcc_flags: bool = True
) -> str:
    """Get shell statement to set Aesara flags for a pipeline submission.

    Args:
        unique_id (str): Unique identifier for this compilation directory.
        aesara_compiledir (Optional[str], optional): Optional parent directory to use
          instead of the default Aesara compilation directory. Defaults to None.
        gcc_flags (bool): Include the gcc vars set in the project configuration. This is
        an optional configuration, so if it is None, then this theano config is not set.
        Defaults to True.

    Returns:
        str: String with a statement for setting some Aesara configurations.
    """
    if aesara_compiledir is None:
        aesara_compiledir = get_aesara_compile_dir()

    # Start with compile directory var.
    theano_vars = f"compiledir={aesara_compiledir}/{unique_id}"

    # Add gcc config flags, if available.
    if gcc_flags and (gcc_vars := _aesara_gcc_config()) is not None:
        theano_vars += "," + gcc_vars

    return f"AESARA_FLAGS='{theano_vars}'"
