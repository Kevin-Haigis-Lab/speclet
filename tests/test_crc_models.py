#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pytest
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

from analysis.common_data_processing import get_indices, read_achilles_data
from analysis.pymc3_models import crc_models
from analysis.sampling_pymc3_models import make_sgrna_to_gene_mapping_df

#### ---- Helper functions ---- ####


def test_nunique_empty():
    assert crc_models.nunique(np.array([])) == 0


def test_nunique_int():
    assert crc_models.nunique(np.array([1])) == 1
    assert crc_models.nunique(np.array([1, 1])) == 1
    assert crc_models.nunique(np.array([1, 2])) == 2


def test_nunique_str():
    assert crc_models.nunique(np.array(["a", "a"])) == 1
    assert crc_models.nunique(np.array(["a", "b"])) == 2


class TestCRCModel1:
    @pytest.fixture(scope="class")
    def mock_data(self) -> pd.DataFrame:
        return read_achilles_data(
            Path("modeling_data", "depmap_CRC_data_subsample.csv")
        )

    @pytest.mark.slow
    def test_return_variables(self, mock_data: pd.DataFrame):
        sgrna_idx = get_indices(mock_data, "sgrna")
        sgrna_to_gene_map = make_sgrna_to_gene_mapping_df(mock_data)
        sgrna_to_gene_idx = get_indices(sgrna_to_gene_map, "hugo_symbol")
        cellline_idx = get_indices(mock_data, "depmap_id")
        batch_idx = get_indices(mock_data, "pdna_batch")
        lfc_data = mock_data.lfc.values

        model, shared_vars = crc_models.model_1(
            sgrna_idx=sgrna_idx,
            sgrna_to_gene_idx=sgrna_to_gene_idx,
            cellline_idx=cellline_idx,
            batch_idx=batch_idx,
            lfc_data=lfc_data,
        )

        assert isinstance(model, pm.Model)
        assert len(shared_vars.keys()) == 5
