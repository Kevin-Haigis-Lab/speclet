from pathlib import Path
from typing import Any, Dict

import arviz as az
import numpy as np
import pandas as pd
import pytest

from src.data_processing import common as dphelp
from src.modeling import simulation_based_calibration_helpers as sbc


class TestSBCFileManager:
    def test_init(self, tmp_path: Path):
        fm = sbc.SBCFileManager(dir=tmp_path)
        assert not fm.all_data_exists()

    @pytest.fixture()
    def priors(self) -> Dict[str, Any]:
        return dict(
            alpha=np.random.uniform(0, 100, size=3),
            beta_log=np.random.uniform(0, 100, size=(10, 15)),
        )

    @pytest.fixture
    def posterior_summary(self) -> pd.DataFrame:
        return pd.DataFrame(dict(x=[5, 6, 7], y=["a", "b", "c"]))

    def test_saving(
        self, tmp_path: Path, priors: Dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)
        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()

    def test_reading(
        self, tmp_path: Path, priors: Dict[str, Any], posterior_summary: pd.DataFrame
    ):
        fm = sbc.SBCFileManager(dir=tmp_path)

        fm.save_sbc_results(
            priors=priors,
            inference_obj=az.InferenceData(),
            posterior_summary=posterior_summary,
        )
        assert fm.all_data_exists()
        read_results = fm.get_sbc_results()
        assert isinstance(read_results, sbc.SBCResults)
        assert isinstance(read_results.inference_obj, az.InferenceData)
        for k in read_results.priors:
            np.testing.assert_array_equal(read_results.priors[k], priors[k])

        for c in read_results.posterior_summary.columns:
            np.testing.assert_array_equal(
                read_results.posterior_summary[c].values, posterior_summary[c].values
            )


#### ---- Test mock data generator ---- ####


def generate_data_with_random_params() -> pd.DataFrame:
    n_genes = np.random.randint(2, 20)
    n_sgrnas_per_gene = np.random.randint(2, 20)
    n_cell_lines = np.random.randint(3, 20)
    n_batches = np.random.randint(1, n_cell_lines)
    return sbc.generate_mock_achilles_data(
        n_genes=n_genes,
        n_sgrnas_per_gene=n_sgrnas_per_gene,
        n_cell_lines=n_cell_lines,
        n_batches=n_batches,
    )


def test_mock_data_has_correct_categories_sizes():
    for _ in range(20):
        n_genes = np.random.randint(2, 20)
        n_sgrnas_per_gene = np.random.randint(2, 20)
        n_cell_lines = np.random.randint(3, 20)
        n_batches = np.random.randint(1, n_cell_lines)
        mock_data = sbc.generate_mock_achilles_data(
            n_genes=n_genes,
            n_sgrnas_per_gene=n_sgrnas_per_gene,
            n_cell_lines=n_cell_lines,
            n_batches=n_batches,
        )
        assert n_genes == dphelp.nunique(mock_data.hugo_symbol)
        assert n_genes * n_sgrnas_per_gene == dphelp.nunique(mock_data.sgrna)
        assert n_cell_lines == dphelp.nunique(mock_data.depmap_id)
        assert n_batches >= dphelp.nunique(mock_data.p_dna_batch)


def test_mock_data_has_correct_kras_mutation_types():
    for _ in range(20):
        n_genes = np.random.randint(2, 20)
        n_sgrnas_per_gene = np.random.randint(2, 20)
        n_cell_lines = np.random.randint(4, 20)
        n_batches = np.random.randint(1, n_cell_lines)
        n_kras_types = np.min([np.random.randint(1, n_cell_lines // 2), 7])
        mock_data = sbc.generate_mock_achilles_data(
            n_genes=n_genes,
            n_sgrnas_per_gene=n_sgrnas_per_gene,
            n_cell_lines=n_cell_lines,
            n_batches=n_batches,
            n_kras_types=n_kras_types,
        )
        assert n_kras_types >= dphelp.nunique(mock_data.kras_mutation)


def test_sgrnas_uniquely_map_to_genes():
    for _ in range(20):
        mock_data = generate_data_with_random_params()
        sgrna_gene_map = (
            mock_data[["sgrna", "hugo_symbol"]].drop_duplicates().reset_index(drop=True)
        )
        sgrnas = sgrna_gene_map["sgrna"].values
        assert len(sgrnas) == len(np.unique(sgrnas))


def test_cellline_in_one_batch():
    for _ in range(20):
        mock_data = generate_data_with_random_params()
        cellline_to_batch = (
            mock_data[["depmap_id", "p_dna_batch"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        cell_lines = cellline_to_batch["depmap_id"].values
        assert len(cell_lines) == len(np.unique(cell_lines))


def test_sgrna_for_each_cellline():
    for _ in range(10):
        mock_data = generate_data_with_random_params()
        all_sgrnas = set(mock_data.sgrna.values.to_list())
        for cell_line in mock_data.depmap_id.values.unique():
            cell_line_sgrnas = mock_data[
                mock_data.depmap_id == cell_line
            ].sgrna.values.to_list()
            # Confirm that each combo happens  exactly once.
            assert len(all_sgrnas) == len(cell_line_sgrnas)
            assert len(all_sgrnas.difference(set(cell_line_sgrnas))) == 0
