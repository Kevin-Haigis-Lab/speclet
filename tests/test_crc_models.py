#!/usr/bin/env python3

from itertools import product
from string import ascii_lowercase as letters
from string import ascii_uppercase as LETTERS
from typing import Dict

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pytest

from src import common_data_processing as dphelp
from src import crc_models

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


#### ---- Models ---- ####


def make_mock_sgrna(of_length: int = 20) -> str:
    return "".join(np.random.choice(list(letters), of_length, replace=True))


@pytest.fixture
def mock_data() -> pd.DataFrame:
    genes = np.random.choice(list(LETTERS), 10, replace=False)
    sgrna_to_gene_map: Dict[str, str] = {}
    for gene in genes:
        for _ in range(np.random.randint(3, 10)):
            sgrna_to_gene_map[make_mock_sgrna()] = gene

    cell_lines = ["line" + str(i) for i in range(5)]
    pdna_batches = ["batch" + str(i) for i in range(3)]
    df = pd.DataFrame(
        product(sgrna_to_gene_map.keys(), cell_lines), columns=["sgrna", "depmap_id"]
    )
    df["hugo_symbol"] = [sgrna_to_gene_map[s] for s in df.sgrna.values]
    df["pdna_batch"] = np.random.choice(pdna_batches, len(df), replace=True)

    df.sort_values(["hugo_symbol", "sgrna"])
    for col in df.columns:
        df = dphelp.make_cat(df, col)

    df["lfc"] = np.random.randn(len(df))
    return df


class TestCRCModel1:
    def make_indices(self, d: pd.DataFrame) -> Dict[str, np.ndarray]:
        sgrna_map = dphelp.make_sgrna_to_gene_mapping_df(d)
        return {
            "sgrna_idx": dphelp.get_indices(d, "sgrna"),
            "sgrna_to_gene_idx": dphelp.get_indices(sgrna_map, "hugo_symbol"),
            "cellline_idx": dphelp.get_indices(d, "depmap_id"),
            "batch_idx": dphelp.get_indices(d, "pdna_batch"),
        }

    @pytest.mark.slow
    def test_return_variables(self, mock_data: pd.DataFrame):
        indices = self.make_indices(mock_data)
        lfc_data = mock_data.lfc.values

        model, shared_vars = crc_models.model_1(
            sgrna_idx=indices["sgrna_idx"],
            sgrna_to_gene_idx=indices["sgrna_to_gene_idx"],
            cellline_idx=indices["cellline_idx"],
            batch_idx=indices["batch_idx"],
            lfc_data=lfc_data,
        )

        assert isinstance(model, pm.Model)
        assert len(shared_vars.keys()) == 5

    @pytest.mark.slow
    def test_mcmc_sampling(self, mock_data: pd.DataFrame):
        indices = self.make_indices(mock_data)
        lfc_data = mock_data.lfc.values

        model, _ = crc_models.model_1(
            sgrna_idx=indices["sgrna_idx"],
            sgrna_to_gene_idx=indices["sgrna_to_gene_idx"],
            cellline_idx=indices["cellline_idx"],
            batch_idx=indices["batch_idx"],
            lfc_data=lfc_data,
        )

        n_chains = 2
        n_draws = 100

        with model:
            trace = pm.sample(
                draws=n_draws,
                tune=100,
                cores=n_chains,
                chains=n_chains,
                return_inferencedata=False,
            )

        assert isinstance(trace, pm.backends.base.MultiTrace)
        assert trace["μ_g"].shape == (n_draws * n_chains,)
        assert trace["μ_α"].shape == (
            n_draws * n_chains,
            dphelp.nunique(mock_data.hugo_symbol),
        )
        assert trace["α_s"].shape == (
            n_draws * n_chains,
            dphelp.nunique(mock_data.sgrna),
        )
        assert trace["β_l"].shape == (
            n_draws * n_chains,
            dphelp.nunique(mock_data.depmap_id),
        )

    @pytest.mark.slow
    def test_advi_sampling(self, mock_data: pd.DataFrame):
        indices = self.make_indices(mock_data)
        lfc_data = mock_data.lfc.values

        model, _ = crc_models.model_1(
            sgrna_idx=indices["sgrna_idx"],
            sgrna_to_gene_idx=indices["sgrna_to_gene_idx"],
            cellline_idx=indices["cellline_idx"],
            batch_idx=indices["batch_idx"],
            lfc_data=lfc_data,
        )

        n_fit = 100
        n_draws = 100

        with model:
            meanfield = pm.fit(n=n_fit)
            trace = meanfield.sample(draws=n_draws)

        assert isinstance(meanfield, pm.Approximation)
        assert len(meanfield.hist) == n_fit
        assert isinstance(trace, pm.backends.base.MultiTrace)
        assert trace["μ_g"].shape == (n_draws,)
        assert trace["μ_α"].shape == (n_draws, dphelp.nunique(mock_data.hugo_symbol))
        assert trace["α_s"].shape == (n_draws, dphelp.nunique(mock_data.sgrna.values))
        assert trace["β_l"].shape == (n_draws, dphelp.nunique(mock_data.depmap_id))
