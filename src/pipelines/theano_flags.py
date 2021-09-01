"""Theano flags for pipelines."""

from typing import Optional

import theano


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
        theano_compiledir = theano.config.compiledir
    return f"THEANO_FLAGS='compiledir={theano_compiledir}/'" + unique_id
