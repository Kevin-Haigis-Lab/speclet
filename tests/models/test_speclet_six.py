from pathlib import Path
from string import ascii_letters
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.data_processing import achilles as achelp
from src.managers.model_data_managers import CrcDataManager
from src.misc import test_helpers as th
from src.modeling import pymc3_helpers as pmhelp
from src.models import speclet_six
from src.models.speclet_six import SpecletSix, SpecletSixConfiguration

chars = [str(i) for i in range(10)] + list(ascii_letters)


def monkey_get_data_path(*args: Any, **kwargs: Any) -> Path:
    return Path("tests", "depmap_test_data.csv")


def make_column_tiled(
    df: pd.DataFrame, df_map: pd.DataFrame, col: str, val: List[str]
) -> pd.DataFrame:
    new_assignemnts = np.tile(val, int(len(df_map) // 2 + 1))
    df_map[col] = new_assignemnts[0 : len(df_map)]
    df_mod = (
        df.drop(columns=[col])
        .merge(df_map, left_index=False, right_index=False)
        .pipe(achelp.set_achilles_categorical_columns)
    )
    return df_mod


def make_data_multiple_lineages(dm: CrcDataManager):
    data = dm.get_data()
    lineage_map = achelp.make_cell_line_to_lineage_mapping_df(data)
    new_lineages = ["lineage_A", "lineage_B"]
    dm.data = make_column_tiled(data, lineage_map, "lineage", new_lineages)


def make_data_multiple_screens(dm: CrcDataManager):
    data = dm.get_data()
    batch_map = achelp.data_batch_indices(data).batch_to_screen_map
    new_screens = ["screen_A", "screen_B"]
    dm.data = make_column_tiled(data, batch_map, "screen", new_screens)


@st.composite
def copynumber_dataframe(draw, group_name: str) -> pd.DataFrame:
    groups = draw(st.lists(st.text(alphabet=chars), min_size=1))
    values = [
        draw(
            st.lists(
                st.floats(
                    min_value=-1000.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=1,
            )
        )
        for _ in groups
    ]
    return (
        pd.DataFrame({group_name: groups, "copy_number": values})
        .explode("copy_number")
        .astype({"copy_number": float})
        .reset_index(drop=True)
    )


class TestSpecletSix:
    @given(copynumber_dataframe(group_name="depmap_id"))
    def test_centered_copynumber_by_cellline(self, df: pd.DataFrame):
        mod_df = speclet_six.centered_copynumber_by_cellline(df.copy())
        for cell_line in mod_df["depmap_id"].unique():
            avg = mod_df.query(f"depmap_id == '{cell_line}'")[
                "copy_number_cellline"
            ].mean()
            assert avg == pytest.approx(0.0, abs=0.001)

    @given(copynumber_dataframe(group_name="hugo_symbol"))
    def test_centered_copynumber_by_gene(self, df: pd.DataFrame):
        mod_df = speclet_six.centered_copynumber_by_gene(df.copy())
        for gene in mod_df["hugo_symbol"].unique():
            avg = mod_df.query(f"hugo_symbol == '{gene}'")["copy_number_gene"].mean()
            assert avg == pytest.approx(0.0, abs=0.001)

    @given(st.lists(st.booleans(), max_size=100))
    def test_converting_is_mutated_column(self, is_mutated: List[bool]):
        df = pd.DataFrame({"is_mutated": is_mutated})
        mod_df = speclet_six.convert_is_mutated_to_numeric(df)
        assert mod_df["is_mutated"].dtype == np.int64

    def test_instantiation(self, tmp_path: Path):
        sp6 = SpecletSix("TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert sp6.model is None

    def test_build_model(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config=st.builds(SpecletSixConfiguration))
    def test_changing_configuration_resets_model(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletSixConfiguration,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def mock_build_model(*args, **kwargs) -> Tuple[str, str]:
            return "my-test-model", "another-string"

        monkeypatch.setattr(SpecletSix, "model_specification", mock_build_model)
        sp6 = SpecletSix(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        th.assert_changing_configuration_resets_model(
            sp6, new_config=config, default_config=SpecletSixConfiguration()
        )

    def test_model_with_multiple_cell_line_lineages(
        self, tmp_path: Path, mock_crc_dm: CrcDataManager
    ):
        make_data_multiple_lineages(mock_crc_dm)
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
        )
        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        for expected_v in ["μ_μ_d", "σ_μ_d", "μ_d", "σ_σ_d"]:
            assert expected_v in model_vars

    def test_model_with_multiple_screens(
        self, tmp_path: Path, mock_crc_dm: CrcDataManager
    ):
        d = mock_crc_dm.get_data().copy()
        d["screen"] = "screen_A"
        d = achelp.set_achilles_categorical_columns(d)
        mock_crc_dm.data = d

        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
        )

        multi_screen_vars = ["μ_μ_j", "σ_μ_j", "σ_σ_j"]

        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        print(model_vars)
        for var in multi_screen_vars:
            assert var not in model_vars

        make_data_multiple_screens(dm=mock_crc_dm)
        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        for var in multi_screen_vars:
            assert var in model_vars

    @pytest.mark.parametrize(
        "arg_name, expected_vars",
        [
            ("cell_line_cna_cov", {"μ_k", "σ_k", "k"}),
            ("gene_cna_cov", {"μ_n", "σ_n", "n"}),
            ("rna_cov", {"μ_q", "σ_q", "q"}),
            ("mutation_cov", {"μ_m", "σ_m", "m"}),
        ],
    )
    @pytest.mark.parametrize("arg_value", [True, False])
    def test_model_with_optional_cellline_cn_covariate(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        arg_name: str,
        arg_value: bool,
        expected_vars: Set[str],
    ):
        config = SpecletSixConfiguration(**{arg_name: arg_value})
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
            config=config,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        model_vars = pmhelp.get_variable_names(sp6.model, rm_log=True)
        for v in expected_vars:
            assert (v in model_vars) == arg_value

    @pytest.mark.slow
    def test_mcmc_sampling(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_multiple_lineages(
        self, tmp_path: Path, mock_crc_dm: CrcDataManager
    ):
        make_data_multiple_lineages(mock_crc_dm)
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_mcmc_sampling_with_optional_covariates(
        self, tmp_path: Path, mock_crc_dm: CrcDataManager
    ):
        sp6 = SpecletSix(
            "TEST-MODEL",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
            config=SpecletSixConfiguration(
                **{
                    "cell_line_cna_cov": True,
                    "gene_cna_cov": True,
                    "rna_cov": True,
                    "mutation_cov": True,
                }
            ),
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.mcmc_results is None
        _ = sp6.mcmc_sample_model(
            draws=10,
            tune=10,
            chains=2,
            cores=2,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.mcmc_results is not None

    @pytest.mark.slow
    def test_advi_sampling(self, tmp_path: Path, mock_crc_dm: CrcDataManager):
        sp6 = SpecletSix(
            "TEST-MODEL", root_cache_dir=tmp_path, debug=True, data_manager=mock_crc_dm
        )
        assert sp6.model is None
        sp6.build_model()
        assert sp6.model is not None
        assert sp6.observed_var_name is not None
        assert sp6.advi_results is None
        _ = sp6.advi_sample_model(
            n_iterations=100,
            draws=10,
            prior_pred_samples=10,
            post_pred_samples=10,
            random_seed=1,
        )
        assert sp6.advi_results is not None

    @settings(
        settings.get_profile("slow-adaptive"),
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(config=st.builds(SpecletSixConfiguration))
    def test_model_parameterizations(
        self,
        tmp_path: Path,
        mock_crc_dm: CrcDataManager,
        config: SpecletSixConfiguration,
    ):
        sp6 = SpecletSix(
            "test-model",
            root_cache_dir=tmp_path,
            debug=True,
            data_manager=mock_crc_dm,
            config=config,
        )

        optional_param_to_name: Dict[str, str] = {
            "k": "cell_line_cna_cov",
            "n": "gene_cna_cov",
            "q": "rna_cov",
            "m": "mutation_cov",
        }

        def pre_check_callback(param_name: str, *args: Any, **kwargs: Any) -> bool:
            if len(param_name) > 1:
                return True
            if (
                param_name in optional_param_to_name.keys()
                and not config.dict()[optional_param_to_name[param_name]]
            ):
                return True
            return False

        th.assert_model_reparameterization(
            sp6, config=config, pre_check_callback=pre_check_callback
        )
