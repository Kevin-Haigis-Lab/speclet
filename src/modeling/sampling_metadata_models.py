#!/usr/bin/env python3

"""Classes for handling model or sampling meta data."""


from typing import Optional

from pydantic import BaseModel

#### ---- Common sampling arguments model ---- ####


class SamplingArguments(BaseModel):
    """Organize arguments/parameters often used for sampling."""

    name: str
    cores: int = 1
    sample: bool = True
    ignore_cache: bool = False
    debug: bool = False
    random_seed: Optional[int] = None
