import random
from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_four import SpecletFour


class TestSpecletFour:
    @pytest.fixture(scope="class")
    def data_manager(self) -> CrcDataManager:
        dm = CrcDataManager(debug=True)
        dm.data = (
            dm.get_data()
            .pipe(achelp.subsample_achilles_data, n_genes=5, n_cell_lines=3)
            .pipe(achelp.set_achilles_categorical_columns)
        )
        assert dphelp.nunique(dm.data["hugo_symbol"]) == 5
        assert dphelp.nunique(dm.data["depmap_id"]) == 3
        return dm

    def test_instantiation(self, tmp_path: Path):
        sp_four = SpecletFour("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp_four.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_four = SpecletFour(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            kras_mutation_minimum=0,
        )
        assert sp_four.model is None
        sp_four.build_model()
        assert sp_four.model is not None
        assert sp_four.observed_var_name is not None
        assert sp_four.mcmc_results is None
        _ = sp_four.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp_four.mcmc_results is not None

    def test_kras_indexing(self, tmp_path: Path):
        dm = CrcDataManager(debug=True)
        dm.data = (
            dm.get_data()
            .pipe(achelp.subsample_achilles_data, n_genes=4, n_cell_lines=25)
            .pipe(achelp.set_achilles_categorical_columns)
        )
        assert dphelp.nunique(dm.data["hugo_symbol"]) == 4
        assert dphelp.nunique(dm.data["depmap_id"]) == 25

        kras_alleles = ["A", "X", "T"]
        kras_cellline_map = pd.DataFrame({"depmap_id": dm.data["depmap_id"]})
        kras_cellline_map = kras_cellline_map.drop_duplicates().reset_index(drop=True)
        kras_muts = random.choices(kras_alleles, k=kras_cellline_map.shape[0])
        kras_cellline_map["kras_mutation"] = kras_muts

        for k in ["E", "F"]:
            cl = random.choices(kras_cellline_map["depmap_id"].values, k=2)
            kras_cellline_map.loc[
                kras_cellline_map.depmap_id.isin(cl), "kras_mutation"
            ] = k

        dm.data = dm.data.drop(columns="kras_mutation").merge(
            kras_cellline_map, on="depmap_id"
        )

        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=dm,
            kras_mutation_minimum=0,
        )
        sp_four.build_model()
        assert sp_four.model is not None
        if sp_four.noncentered_param:
            a = sp_four.model["μ_g_offset"]
        else:
            a = sp_four.model["μ_g"]
        n_expected_kras_alleles = 5
        assert a.dshape == (n_expected_kras_alleles,)

        sp_four = SpecletFour(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=dm,
            kras_mutation_minimum=3,
        )
        sp_four.build_model()
        assert sp_four.model is not None
        if sp_four.noncentered_param:
            a = sp_four.model["μ_g_offset"]
        else:
            a = sp_four.model["μ_g"]
        n_expected_kras_alleles = 4
        assert a.dshape == (n_expected_kras_alleles,)
