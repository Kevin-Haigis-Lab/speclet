"""Theano flags for pipelines."""

from typing import Optional

import theano

from speclet.project_configuration import read_project_configuration


def get_theano_compile_dir() -> str:
    """Get the Theano compilation directory from the config.

    Returns:
        str: String of path the compilation directory.
    """
    compile_dir: str = theano.config.compiledir
    return compile_dir


def _theano_gcc_config() -> Optional[str]:
    gcc_vars = read_project_configuration().misc.theano_gcc_flag
    return gcc_vars


def get_theano_flags(
    unique_id: str, theano_compiledir: Optional[str] = None, gcc_flags: bool = True
) -> str:
    """Get shell statement to set Theano flags for a pipeline submission.

    Args:
        unique_id (str): Unique identifier for this compilation directory.
        theano_compiledir (Optional[str], optional): Optional parent directory to use
          instead of the default Theano compilation directory. Defaults to None.
        gcc_flags (bool): Include the gcc vars set in the project configuration. This is
        an optional configuration, so if it is None, then this theano config is not set.
        Defaults to True.

    Returns:
        str: String with a statement for setting some Theano configurations.
    """
    if theano_compiledir is None:
        theano_compiledir = get_theano_compile_dir()

    # Start with compile directory var.
    theano_vars = f"compiledir={theano_compiledir}/{unique_id}"

    # Add gcc config flags, if available.
    if gcc_flags and (gcc_vars := _theano_gcc_config()) is not None:
        theano_vars += "," + gcc_vars

    return f"THEANO_FLAGS='{theano_vars}'"
