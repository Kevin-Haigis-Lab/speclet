#!/usr/bin/env python3

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from analysis.pymc3_models import crc_models


def test_nunique_empty():
    assert crc_models.nunique(np.array([])) == 0


def test_nunique_int():
    assert crc_models.nunique(np.array([1])) == 1
    assert crc_models.nunique(np.array([1, 1])) == 1
    assert crc_models.nunique(np.array([1, 2])) == 2


def test_nunique_str():
    assert crc_models.nunique(np.array(["a", "a"])) == 1
    assert crc_models.nunique(np.array(["a", "b"])) == 2
