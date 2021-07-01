#!/usr/bin/env python3

from pathlib import Path
from typing import Callable

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from src.command_line_interfaces import cli_helpers
from src.models.ceres_mimic import CeresMimic
from src.models.speclet_five import SpecletFive
from src.models.speclet_four import SpecletFour
from src.models.speclet_seven import SpecletSeven
from src.models.speclet_six import SpecletSix
from src.models.speclet_two import SpecletTwo
from src.project_enums import ModelFitMethod

#### ---- Models ---- ####


def test_clean_model_names():
    assert cli_helpers.clean_model_names("model_name") == "model_name"
    assert cli_helpers.clean_model_names("model name") == "model-name"
    assert cli_helpers.clean_model_names("model named Jerry") == "model-named-Jerry"


def test_get_model_class():

    m2 = cli_helpers.get_model_class(cli_helpers.ModelOption.crc_ceres_mimic)
    assert m2 == CeresMimic


@given(name=st.text(min_size=1))
def test_extract_fit_method(name: str):
    assume(name.lower() not in [n.value.lower() for n in ModelFitMethod])
    for fit_method in ["ADVI", "advi"]:
        name_mod = name + "_" + fit_method
        assert cli_helpers.extract_fit_method(name_mod) == ModelFitMethod.ADVI

    for fit_method in ["MCMC", "mcmc"]:
        name_mod = name + "_" + "mcmc"
        assert cli_helpers.extract_fit_method(name_mod) == ModelFitMethod.MCMC


#### ---- Modifying models ---- ####


class TestCeresMimicModifications:
    @pytest.fixture
    def ceres_model(self, tmp_path: Path) -> CeresMimic:
        return CeresMimic(name="TEST-MODEL", root_cache_dir=Path(tmp_path), debug=True)

    model_names = [
        "model",
        "ceres-model",
        "CERES-model",
        "pymc3-ceres",
        "pymc3 ceres",
    ]

    functions_to_test = [
        cli_helpers.modify_ceres_model_by_name,
        cli_helpers.modify_model_by_name,
    ]

    @pytest.mark.parametrize("model_name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    def test_modify_ceres_model_by_name_nochange(
        self, ceres_model: CeresMimic, model_name: str, fxn: Callable
    ):
        fxn(ceres_model, model_name)
        assert not ceres_model.copynumber_cov
        assert not ceres_model.sgrna_intercept_cov

    @pytest.mark.parametrize("model_name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    def test_modify_ceres_model_by_name_sgrna(
        self, ceres_model: CeresMimic, model_name: str, fxn: Callable
    ):
        model_name += "_sgrnaint"
        fxn(ceres_model, model_name)
        assert not ceres_model.copynumber_cov
        assert ceres_model.sgrna_intercept_cov

    @pytest.mark.parametrize("model_name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    def test_modify_ceres_model_by_name_copynumber(
        self, ceres_model: CeresMimic, model_name: str, fxn: Callable
    ):
        model_name += "_copynumber"
        fxn(ceres_model, model_name)
        assert ceres_model.copynumber_cov
        assert not ceres_model.sgrna_intercept_cov

    @pytest.mark.parametrize("model_name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    def test_modify_ceres_model_by_name_copynumber_sgrna(
        self, ceres_model: CeresMimic, model_name: str, fxn: Callable
    ):
        model_name += "_copynumber-sgrnaint"
        fxn(ceres_model, model_name)
        assert ceres_model.copynumber_cov
        assert ceres_model.sgrna_intercept_cov


class TestSpecletTwoModifications:
    @pytest.fixture
    def sp2_model(self, tmp_path: Path) -> SpecletTwo:
        return SpecletTwo(name="TEST-MODEL", root_cache_dir=Path(tmp_path), debug=True)

    model_names = [
        "model",
        "sp2-model",
        "SpecletTwo-model",
        "pymc3-Sp2",
        "pymc3 Speclet2",
        "pymc3 SpecletTwo",
        "SpecletTwo-kras",
        "SpecletTwo-cna",
        "SpecletTwo-gene",
        "SpecletTwo-mutation",
    ]

    @pytest.mark.parametrize("name", model_names)
    def test_modify_sp2_model_by_name_nochange(self, sp2_model: SpecletTwo, name: str):
        cli_helpers.modify_model_by_name(sp2_model, name)
        assert True  # to force code to run


class TestSpecletFourModifications:
    @pytest.fixture
    def sp4_model(self, tmp_path: Path) -> SpecletFour:
        return SpecletFour(name="TEST-MODEL", root_cache_dir=Path(tmp_path), debug=True)

    model_names = [
        "model",
        "sp4-model",
        "SpecletFour-model",
        "pymc3-Sp4",
        "pymc3 Speclet2",
        "pymc3 SpecletFour",
    ]

    functions_to_test = [
        cli_helpers.modify_specletfour_model_by_name,
        cli_helpers.modify_model_by_name,
    ]

    @pytest.mark.parametrize("name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    def test_modify_sp4_model_by_name_nochange(
        self, sp4_model: SpecletFour, name: str, fxn: Callable
    ):
        fxn(sp4_model, name)
        assert not sp4_model.copy_number_cov

    @pytest.mark.parametrize("name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    def test_modify_sp4_model_by_name_copynumber(
        self, sp4_model: SpecletFour, name: str, fxn: Callable
    ):
        name += "cn-cov"
        fxn(sp4_model, name)
        assert sp4_model.copy_number_cov


class TestSpecletFiveModifications:
    @pytest.fixture
    def sp5_model(self, tmp_path: Path) -> SpecletFive:
        return SpecletFive(name="TEST-MODEL", root_cache_dir=Path(tmp_path), debug=True)

    model_names = [
        "model",
        "sp5-model",
        "SpecletFive-model",
        "pymc3-Sp5",
        "pymc3 Speclet2",
        "pymc3 SpecletFive",
        "SpecletFive-kras",
        "SpecletFive-cna",
        "SpecletFive-gene",
        "SpecletFive-mutation",
    ]

    @pytest.mark.parametrize("name", model_names)
    def test_modify_sp5_model_by_name_nochange(self, sp5_model: SpecletFive, name: str):
        cli_helpers.modify_model_by_name(sp5_model, name)
        assert True  # to force code to run


class TestSpecletSixModifications:

    model_names = [
        "model",
        "sp6-model",
        "SpecletSix-model",
        "pymc3-Sp6",
        "pymc3 Speclet2",
        "pymc3 SpecletSix",
    ]

    functions_to_test = [
        cli_helpers.modify_specletsix_model_by_name,
        cli_helpers.modify_model_by_name,
    ]

    @settings(deadline=60000.0)  # 1 minute deadline
    @pytest.mark.parametrize("name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    @given(bool_ary=hnp.arrays(bool, shape=4))
    def test_modify_sp6_model_by_name(self, name: str, fxn: Callable, bool_ary):
        sp6_model = SpecletSix(
            name="TEST-MODEL", root_cache_dir=Path("temp/test-models"), debug=True
        )
        suffixes = ["cellcna", "genecna", "rna", "mutation"]
        for b, s in zip(bool_ary.flatten(), suffixes):
            if b:
                name += s

        fxn(sp6_model, name)
        assert sp6_model.cell_line_cna_cov == bool_ary[0]
        assert sp6_model.gene_cna_cov == bool_ary[1]
        assert sp6_model.rna_cov == bool_ary[2]
        assert sp6_model.mutation_cov == bool_ary[3]


class TestSpecletSevenModifications:

    model_names = [
        "model",
        "sp7-model",
        "SpecletSeven-model",
        "pymc3-Sp7",
        "pymc3 Speclet2",
        "pymc3 SpecletSeven",
    ]

    functions_to_test = [
        cli_helpers.modify_specletseven_model_by_name,
        cli_helpers.modify_model_by_name,
    ]

    @pytest.mark.parametrize("name", model_names)
    @pytest.mark.parametrize("fxn", functions_to_test)
    @given(bool_ary=hnp.arrays(bool, shape=1))
    def test_modify_sp7_model_by_name(self, name: str, fxn: Callable, bool_ary):
        sp7_model = SpecletSeven(
            name="TEST-MODEL", root_cache_dir=Path("temp/test-models"), debug=True
        )
        suffixes = ["noncentered"]
        for b, s in zip(bool_ary.flatten(), suffixes):
            if b:
                name += s

        fxn(sp7_model, name)
        assert sp7_model.noncentered_param == bool_ary[0]
