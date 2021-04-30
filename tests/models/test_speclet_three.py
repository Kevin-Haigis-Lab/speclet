import random
from pathlib import Path

import pandas as pd
import pymc3 as pm
import pytest

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.managers.model_data_managers import CrcDataManager
from src.models.speclet_three import SpecletThree


class TestSpecletThree:
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
        sp_two = SpecletThree("test-model", root_cache_dir=tmp_path, debug=True)
        assert sp_two.model is None

    def test_build_model(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_two = SpecletThree(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp_two.model is None
        sp_two.build_model()
        assert sp_two.model is not None

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_two = SpecletThree(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp_two.model is None
        sp_two.build_model()
        assert sp_two.model is not None
        assert sp_two.observed_var_name is not None
        assert sp_two.mcmc_results is None
        _ = sp_two.mcmc_sample_model(
            mcmc_draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp_two.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_two = SpecletThree(
            "test-model", root_cache_dir=tmp_path, debug=True, data_manager=data_manager
        )
        assert sp_two.model is None
        sp_two.build_model()
        assert sp_two.model is not None
        assert sp_two.observed_var_name is not None
        assert sp_two.advi_results is None
        _ = sp_two.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp_two.advi_results is not None

    def test_optional_kras_cov(self, tmp_path: Path, data_manager: CrcDataManager):
        sp_two = SpecletThree(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=data_manager,
            kras_cov=False,
        )

        assert sp_two.model is None
        sp_two.build_model()
        assert isinstance(sp_two.model, pm.Model)
        assert "a" not in list(sp_two.model.named_vars.keys())

        sp_two.kras_cov = True
        assert sp_two.model is None
        sp_two.build_model()
        assert isinstance(sp_two.model, pm.Model)
        assert "a" in list(sp_two.model.named_vars.keys())

        sp_two = SpecletThree(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            kras_cov=True,
            data_manager=data_manager,
        )
        assert sp_two.model is None
        sp_two.build_model()
        assert isinstance(sp_two.model, pm.Model)
        assert "a" in list(sp_two.model.named_vars.keys())

    @pytest.mark.DEV
    def test_kras_indexing(self, tmp_path: Path):
        dm = CrcDataManager(debug=True)
        dm.data = (
            dm.get_data()
            .pipe(achelp.subsample_achilles_data, n_genes=4, n_cell_lines=50)
            .pipe(achelp.set_achilles_categorical_columns)
        )
        assert dphelp.nunique(dm.data["hugo_symbol"]) <= 4
        assert dphelp.nunique(dm.data["depmap_id"]) >= 20

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

        sp_two = SpecletThree(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=dm,
            kras_cov=True,
            kras_mutation_minimum=0,
        )
        sp_two.build_model()
        assert sp_two.model is not None
        a = sp_two.model["a"]
        n_genes = dphelp.nunique(dm.data["hugo_symbol"])
        n_expected_kras_alleles = 5
        assert a.dshape == (n_genes, n_expected_kras_alleles)

        sp_two = SpecletThree(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=dm,
            kras_cov=True,
            kras_mutation_minimum=3,
        )
        sp_two.build_model()
        assert sp_two.model is not None
        a = sp_two.model["a"]
        n_genes = dphelp.nunique(dm.data["hugo_symbol"])
        n_expected_kras_alleles = 4
        assert a.dshape == (n_genes, n_expected_kras_alleles)
