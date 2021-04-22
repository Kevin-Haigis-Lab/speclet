#!/usr/bin/env python3

import abc
import time
from itertools import product
from pathlib import Path
from string import ascii_lowercase as letters
from string import ascii_uppercase as LETTERS
from typing import Dict, Type, Union

import numpy as np
import pandas as pd
import pymc3 as pm
import pytest
import theano.tensor as tt

import src.modeling.simulation_based_calibration_helpers as sbc
from src.data_processing import common as dphelp
from src.data_processing.achilles import zscale_cna_by_group
from src.modeling import pymc3_sampling_api as pmapi
from src.modeling.sampling_metadata_models import SamplingArguments
from src.models.crc_ceres_mimic import CrcCeresMimic
from src.models.crc_model import CrcModel
from src.models.crc_model_one import CrcModelOne
from src.models.speclet_model import SpecletModel
from src.models.speclet_one import SpecletOne

#### ---- Mock data ---- ####


def make_mock_sgrna(of_length: int = 20) -> str:
    return "".join(np.random.choice(list(letters), of_length, replace=True))


@pytest.fixture
def mock_data() -> pd.DataFrame:
    genes = np.random.choice(list(LETTERS), 10, replace=False)
    sgrna_to_gene_map: Dict[str, str] = {}
    for gene in genes:
        for _ in range(np.random.randint(3, 6)):
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

    df["gene_cn"] = np.abs(np.random.normal(2, 0.1, len(df)))
    df["log2_cn"] = np.log2(df.gene_cn + 1)
    df = zscale_cna_by_group(
        df,
        gene_cn_col="log2_cn",
        new_col="z_log2_cn",
        groupby_cols=["depmap_id"],
        cn_max=np.log2(10),
    )
    df["lfc"] = np.random.randn(len(df))
    return df


#### ---- Test CrcModel ---- ####


class TestCrcModel:
    def test_inheritance(self, tmp_path: Path):
        model = CrcModel(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert isinstance(model, SpecletModel)

    def test_batch_size(self, tmp_path: Path):
        model = CrcModel(name="TEST-MODEL", root_cache_dir=tmp_path)
        not_debug_batch_size = model.get_batch_size()
        model.debug = True
        debug_batch_size = model.get_batch_size()
        assert debug_batch_size < not_debug_batch_size

    def test_data_paths(self, tmp_path: Path):
        model = CrcModel(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert model.get_data_path().exists and model.get_data_path().is_file()

    def test_get_data(self, tmp_path: Path):
        model = CrcModel(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert model.data is None
        data = model.get_data()
        assert model.data is not None
        assert model.data.shape[0] > model.data.shape[1]
        assert model.data.shape[0] == data.shape[0]
        assert model.data.shape[1] == data.shape[1]


#### ---- Test CrcModelOne ---- ####

AnyModel = Type[Union[CrcModelOne, CrcCeresMimic]]


class CrcModelSubclassesTests:

    Model: AnyModel

    @abc.abstractmethod
    def check_trace_shape(
        self,
        trace: pm.backends.base.MultiTrace,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        raise Exception("The `check_trace_shape()` method needs to be implemented.")

    @abc.abstractmethod
    def check_approx_fit(self, approx: pm.Approximation, n_fit: int):
        raise Exception("The `check_approx_fit()` method needs to be implemented.")

    def compare_two_results(
        self, trace_1: pm.backends.base.MultiTrace, trace_2: pm.backends.base.MultiTrace
    ):
        assert set(trace_1.varnames) == set(trace_2.varnames)
        for p in trace_1.varnames:
            np.testing.assert_array_equal(trace_1[p], trace_2[p])

    def model_init_callback(self, model: AnyModel):
        pass

    def check_mcmc_results(
        self,
        res: pmapi.MCMCSamplingResults,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        self.check_trace_shape(res.trace, n_draws=n_draws, n_chains=n_chains, data=data)

    def check_advi_results(
        self,
        res: pmapi.ApproximationSamplingResults,
        n_draws: int,
        n_fit: int,
        data: pd.DataFrame,
    ):
        self.check_approx_fit(approx=res.approximation, n_fit=n_fit)
        self.check_trace_shape(res.trace, n_draws=n_draws, n_chains=1, data=data)

    def test_init(self, tmp_path: Path):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        assert issubclass(self.Model, CrcModel)
        assert isinstance(model, CrcModel)

    @pytest.mark.slow
    def test_build_model(self, tmp_path: Path):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        assert model.data is None
        assert model.model is None
        assert model.shared_vars is None
        model.build_model()
        assert model.data is not None
        assert isinstance(model.data, pd.DataFrame)
        assert model.model is not None
        assert isinstance(model.model, pm.Model)
        assert model.shared_vars is not None
        for key, value in model.shared_vars.items():
            assert isinstance(key, str)
            assert isinstance(value, tt.sharedvar.TensorSharedVariable)

    @pytest.fixture
    def sampling_args(self) -> SamplingArguments:
        return SamplingArguments(
            name="MOCK_MODEL", cores=2, ignore_cache=True, debug=True, random_seed=123
        )

    def test_error_for_sampling_without_building(
        self, sampling_args: SamplingArguments, tmp_path: Path
    ):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        with pytest.raises(AttributeError):
            _ = model.mcmc_sample_model(sampling_args)
        with pytest.raises(AttributeError):
            _ = model.advi_sample_model(sampling_args)

    @pytest.mark.slow
    def test_manual_mcmc_sampling(self, mock_data: pd.DataFrame, tmp_path: Path):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.data = mock_data  # inject mock data
        model.build_model()

        n_chains = 2
        n_draws = 100

        with model.model:
            trace = pm.sample(
                draws=n_draws,
                tune=100,
                cores=n_chains,
                chains=n_chains,
                return_inferencedata=False,
            )

        self.check_trace_shape(
            trace=trace, n_draws=n_draws, n_chains=n_chains, data=mock_data
        )

    @pytest.mark.slow
    def test_mcmc_sampling_method(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments, tmp_path: Path
    ):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.data = mock_data  # inject mock data
        model.build_model()

        n_chains = 2
        n_draws = 100

        mcmc_results = model.mcmc_sample_model(
            mcmc_draws=n_draws, tune=100, chains=n_chains, sampling_args=sampling_args
        )

        self.check_mcmc_results(
            mcmc_results, n_draws=n_draws, n_chains=n_chains, data=mock_data
        )

    @pytest.mark.slow
    def test_manual_advi_sampling(self, mock_data: pd.DataFrame, tmp_path: Path):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.data = mock_data  # inject mock data
        model.build_model()

        n_fit: int = np.random.randint(100, 200)
        n_draws: int = np.random.randint(100, 200)

        with model.model:
            approx = pm.fit(n=n_fit)
            trace = approx.sample(draws=n_draws)

        self.check_approx_fit(approx, n_fit=n_fit)
        self.check_trace_shape(trace=trace, n_draws=n_draws, n_chains=1, data=mock_data)

    @pytest.mark.slow
    def test_advi_sampling_method(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments, tmp_path: Path
    ):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.data = mock_data  # inject mock data
        model.build_model()

        n_fit = 1000
        n_draws = 100

        advi_results = model.advi_sample_model(
            n_iterations=n_fit, draws=n_draws, sampling_args=sampling_args
        )
        self.check_advi_results(
            advi_results, n_draws=n_draws, n_fit=n_fit, data=mock_data
        )

    @pytest.mark.slow
    def test_not_rerun_mcmc_sampling(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments, tmp_path: Path
    ):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.data = mock_data  # inject mock data
        model.build_model()
        results_1 = model.mcmc_sample_model(
            mcmc_draws=100, tune=100, chains=2, sampling_args=sampling_args
        )
        a = time.time()
        results_2 = model.mcmc_sample_model(sampling_args=sampling_args)
        b = time.time()

        assert b - a < 2  # should be very quick
        self.compare_two_results(results_1.trace, results_2.trace)

    @pytest.mark.slow
    def test_not_rerun_advi_sampling(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments, tmp_path: Path
    ):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.data = mock_data  # inject mock data
        model.build_model()
        results_1 = model.advi_sample_model(
            n_iterations=1000, draws=100, sampling_args=sampling_args
        )
        a = time.time()
        results_2 = model.advi_sample_model(sampling_args=sampling_args)
        b = time.time()

        assert b - a < 2  # should be very quick

        np.testing.assert_array_equal(
            results_1.approximation.hist, results_2.approximation.hist
        )
        self.compare_two_results(results_1.trace, results_2.trace)

    def test_sbc_standard(self, tmp_path: Path):
        model = self.Model(name="TEST-MODEL", root_cache_dir=tmp_path, debug=True)
        self.model_init_callback(model)
        model.run_simulation_based_calibration(results_path=tmp_path, size="small")
        fm = sbc.SBCFileManager(dir=tmp_path)
        assert fm.all_data_exists()
        res = fm.get_sbc_results()
        assert isinstance(res, sbc.SBCResults)


class TestCrcModelOne(CrcModelSubclassesTests):

    Model = CrcModelOne

    def check_trace_shape(
        self,
        trace: pm.backends.base.MultiTrace,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        assert isinstance(trace, pm.backends.base.MultiTrace)
        total_samples = n_draws * n_chains
        assert trace["μ_g"].shape == (total_samples,)
        assert trace["μ_α"].shape == (total_samples, dphelp.nunique(data.hugo_symbol))
        assert trace["α_s"].shape == (total_samples, dphelp.nunique(data.sgrna))
        assert trace["β_l"].shape == (total_samples, dphelp.nunique(data.depmap_id))

    def check_approx_fit(self, approx: pm.Approximation, n_fit: int):
        assert isinstance(approx, pm.Approximation)
        assert len(approx.hist) <= n_fit


class TestCrcCeresMimic(CrcModelSubclassesTests):
    Model = CrcCeresMimic

    def check_trace_shape(
        self,
        trace: pm.backends.base.MultiTrace,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        assert isinstance(trace, pm.backends.base.MultiTrace)
        total_samples = n_draws * n_chains
        assert trace["μ_h"].shape == (total_samples,)
        assert trace["h"].shape == (total_samples, dphelp.nunique(data.hugo_symbol))
        assert trace["d"].shape == (
            total_samples,
            dphelp.nunique(data.hugo_symbol),
            dphelp.nunique(data.depmap_id),
        )
        assert trace["σ_a"].shape == (total_samples, 2)

    def check_approx_fit(self, approx: pm.Approximation, n_fit: int):
        assert isinstance(approx, pm.Approximation)
        assert len(approx.hist) <= n_fit

    def compare_two_results(
        self, trace_1: pm.backends.base.MultiTrace, trace_2: pm.backends.base.MultiTrace
    ):
        super().compare_two_results(trace_1, trace_2)
        for optional_param in ["β", "o"]:
            assert optional_param not in trace_1.varnames


class TestCrcCeresMimicCopyNumber(CrcModelSubclassesTests):
    Model = CrcCeresMimic

    def model_init_callback(self, model: AnyModel):
        assert isinstance(model, CrcCeresMimic)
        model.copynumber_cov = True

    def check_trace_shape(
        self,
        trace: pm.backends.base.MultiTrace,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        TestCrcCeresMimic().check_trace_shape(trace, n_draws, n_chains, data)
        assert trace["β"].shape == (n_draws * n_chains, dphelp.nunique(data.depmap_id))

    def compare_two_results(
        self, trace_1: pm.backends.base.MultiTrace, trace_2: pm.backends.base.MultiTrace
    ):
        super().compare_two_results(trace_1, trace_2)
        for new_param in ["μ_β", "σ_β", "β"]:
            assert new_param in trace_1.varnames

    def check_approx_fit(self, approx: pm.Approximation, n_fit: int):
        TestCrcCeresMimic().check_approx_fit(approx, n_fit)

    def test_gene_covariate_setter(self, tmp_path: Path):
        ceres_model = CrcCeresMimic(
            name="TEST-MODEL", root_cache_dir=tmp_path, debug=True
        )
        assert not ceres_model.copynumber_cov
        assert ceres_model.model is None

        ceres_model.build_model()
        assert ceres_model.model is not None
        assert isinstance(ceres_model.model, pm.Model)
        assert "β" not in [param.name for param in ceres_model.model.free_RVs]

        ceres_model.copynumber_cov = True
        assert ceres_model.copynumber_cov
        assert ceres_model.model is None

        ceres_model.build_model()
        assert ceres_model.model is not None
        assert isinstance(ceres_model.model, pm.Model)
        assert "β" in [param.name for param in ceres_model.model.free_RVs]


class TestCrcCeresMimicSgrna(CrcModelSubclassesTests):
    Model = CrcCeresMimic

    def model_init_callback(self, model: AnyModel):
        assert isinstance(model, CrcCeresMimic)
        model.sgrna_intercept_cov = True

    def check_trace_shape(
        self,
        trace: pm.backends.base.MultiTrace,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        TestCrcCeresMimic().check_trace_shape(trace, n_draws, n_chains, data)
        assert trace["o"].shape == (
            n_draws * n_chains,
            dphelp.nunique(data.sgrna),
        )

    def compare_two_results(
        self, trace_1: pm.backends.base.MultiTrace, trace_2: pm.backends.base.MultiTrace
    ):
        super().compare_two_results(trace_1, trace_2)
        for new_param in ["o", "σ_o", "μ_o"]:
            assert new_param in trace_1.varnames

    def check_approx_fit(self, approx: pm.Approximation, n_fit: int):
        TestCrcCeresMimic().check_approx_fit(approx, n_fit)

    def test_gene_covariate_setter(self, tmp_path: Path):
        ceres_model = CrcCeresMimic(
            name="TEST-MODEL", root_cache_dir=tmp_path, debug=True
        )
        assert not ceres_model.sgrna_intercept_cov
        assert ceres_model.model is None

        ceres_model.build_model()
        assert ceres_model.model is not None
        assert isinstance(ceres_model.model, pm.Model)
        assert "o" not in [param.name for param in ceres_model.model.free_RVs]

        ceres_model.sgrna_intercept_cov = True
        assert ceres_model.sgrna_intercept_cov
        assert ceres_model.model is None

        ceres_model.build_model()
        assert ceres_model.model is not None
        assert isinstance(ceres_model.model, pm.Model)
        assert "o" in [param.name for param in ceres_model.model.free_RVs]


@pytest.mark.DEV
class TestSpecletOne(CrcModelSubclassesTests):

    Model = SpecletOne

    def check_trace_shape(
        self,
        trace: pm.backends.base.MultiTrace,
        n_draws: int,
        n_chains: int,
        data: pd.DataFrame,
    ):
        assert isinstance(trace, pm.backends.base.MultiTrace)

    def check_approx_fit(self, approx: pm.Approximation, n_fit: int):
        assert isinstance(approx, pm.Approximation)
        assert len(approx.hist) <= n_fit
