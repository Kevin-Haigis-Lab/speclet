"""Theano flags for pipelines."""

from typing import Optional

import theano


def get_theano_compile_dir() -> str:
    """Get the Theano compilation directory from the config.

    Returns:
        str: String of path the compilation directory.
    """
    compile_dir: str = theano.config.compiledir
    return compile_dir


def get_theano_flags(unique_id: str, theano_compiledir: Optional[str] = None) -> str:
    """Get shell statement to set Theano flags for a pipeline submission.

    Args:
        unique_id (str): Unique identifier for this compilation directory.
        theano_compiledir (Optional[str], optional): Optional parent directory to use
          instead of the default Theano compilation directory. Defaults to None.

    Returns:
        str: String with a statement for setting some Theano configurations.
    """
    if theano_compiledir is None:
        theano_compiledir = get_theano_compile_dir()
    return f"THEANO_FLAGS='compiledir={theano_compiledir}/'" + unique_id
