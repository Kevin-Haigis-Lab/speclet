#!/usr/bin/env python3

import time
from itertools import product
from pathlib import Path
from string import ascii_lowercase as letters
from string import ascii_uppercase as LETTERS
from typing import Dict

import numpy as np
import pandas as pd
import pretty_errors
import pymc3 as pm
import pytest
import theano.tensor as tt

from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.modeling.sampling_pymc3_models import SamplingArguments
from src.models import crc_models, speclet_model

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


#### ---- Test CrcModel ---- ####


class TestCrcModel:
    def test_inheritance(self, tmp_path: Path):
        model = crc_models.CrcModel(cache_dir=tmp_path, debug=True)
        assert isinstance(model.get_cache_file_names(), speclet_model.ModelCachePaths)
        assert model.cache_dir == tmp_path

    def test_batch_size(self):
        model = crc_models.CrcModel()
        not_debug_batch_size = model.get_batch_size()
        model.debug = True
        debug_batch_size = model.get_batch_size()
        assert debug_batch_size < not_debug_batch_size

    def test_data_paths(self):
        model = crc_models.CrcModel(debug=True)
        assert model.get_data_path().exists and model.get_data_path().is_file()

    def test_get_data(self):
        model = crc_models.CrcModel(debug=True)
        assert model.data is None
        data = model.get_data()
        assert model.data is not None
        assert model.data.shape[0] > model.data.shape[1]
        assert model.data.shape[0] == data.shape[0]
        assert model.data.shape[1] == data.shape[1]


#### ---- Test CrcModelOne ---- ####


class TestCRCModel1:
    def test_init(self):
        model = crc_models.CrcModelOne()
        assert isinstance(model, crc_models.CrcModelOne)
        assert model.cache_dir is None
        assert model.debug == False

    @pytest.mark.slow
    def test_build_model(self):
        model = crc_models.CrcModelOne(debug=True)
        assert model.data is None
        assert model.model is None
        assert model.shared_vars is None
        model.build_model()
        assert model.data is not None
        assert isinstance(model.data, pd.DataFrame)
        assert model.model is not None
        assert isinstance(model.model, pm.Model)
        assert model.shared_vars is not None
        assert len(model.shared_vars.keys()) == 5
        for key, value in model.shared_vars.items():
            assert isinstance(key, str)
            assert isinstance(value, tt.sharedvar.TensorSharedVariable)

    @pytest.fixture
    def sampling_args(self) -> SamplingArguments:
        return SamplingArguments(
            name="MOCK_MODEL", cores=2, ignore_cache=True, debug=True, random_seed=123
        )

    def test_error_for_sampling_without_building(
        self, sampling_args: SamplingArguments
    ):
        model = crc_models.CrcModelOne()
        with pytest.raises(AttributeError):
            _ = model.mcmc_sample_model(sampling_args)
        with pytest.raises(AttributeError):
            _ = model.advi_sample_model(sampling_args)

    @pytest.mark.slow
    def test_manual_mcmc_sampling(self, mock_data: pd.DataFrame):
        model = crc_models.CrcModelOne()
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
    def test_mcmc_sampling_method(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments
    ):
        model = crc_models.CrcModelOne()
        model.data = mock_data  # inject mock data
        model.build_model()

        n_chains = 3
        n_draws = 1000

        mcmc_results = model.mcmc_sample_model(sampling_args=sampling_args)

        trace = mcmc_results.trace
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
    def test_manual_advi_sampling(self, mock_data: pd.DataFrame):
        model = crc_models.CrcModelOne()
        model.data = mock_data  # inject mock data
        model.build_model()

        n_fit: int = np.random.randint(100, 200)
        n_draws: int = np.random.randint(100, 200)

        with model.model:
            approx = pm.fit(n=n_fit)
            trace = approx.sample(draws=n_draws)

        assert isinstance(approx, pm.Approximation)
        assert len(approx.hist) <= n_fit
        assert isinstance(trace, pm.backends.base.MultiTrace)
        assert trace["μ_g"].shape == (n_draws,)
        assert trace["μ_α"].shape == (n_draws, dphelp.nunique(mock_data.hugo_symbol))
        assert trace["α_s"].shape == (n_draws, dphelp.nunique(mock_data.sgrna.values))
        assert trace["β_l"].shape == (n_draws, dphelp.nunique(mock_data.depmap_id))

    @pytest.mark.slow
    def test_advi_sampling_method(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments
    ):
        model = crc_models.CrcModelOne()
        model.data = mock_data  # inject mock data
        model.build_model()

        n_fit = 100000
        n_draws = 1000

        advi_results = model.advi_sample_model(sampling_args=sampling_args)
        approx = advi_results.approximation
        trace = advi_results.trace

        assert isinstance(approx, pm.Approximation)
        assert len(approx.hist) <= n_fit
        assert isinstance(trace, pm.backends.base.MultiTrace)
        assert trace["μ_g"].shape == (n_draws,)
        assert trace["μ_α"].shape == (n_draws, dphelp.nunique(mock_data.hugo_symbol))
        assert trace["α_s"].shape == (n_draws, dphelp.nunique(mock_data.sgrna.values))
        assert trace["β_l"].shape == (n_draws, dphelp.nunique(mock_data.depmap_id))

    @pytest.mark.DEV
    def test_not_rerun_mcmc_sampling(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments
    ):
        model = crc_models.CrcModelOne()
        model.data = mock_data  # inject mock data
        model.build_model()
        results_1 = model.mcmc_sample_model(sampling_args=sampling_args)
        a = time.time()
        results_2 = model.mcmc_sample_model(sampling_args=sampling_args)
        b = time.time()

        assert b - a < 2  # should be very quick

        for p in ["μ_g", "μ_α", "α_s", "β_l"]:
            np.testing.assert_array_equal(results_1.trace[p], results_2.trace[p])

    @pytest.mark.DEV
    def test_not_rerun_advi_sampling(
        self, mock_data: pd.DataFrame, sampling_args: SamplingArguments
    ):
        model = crc_models.CrcModelOne()
        model.data = mock_data  # inject mock data
        model.build_model()
        results_1 = model.advi_sample_model(sampling_args=sampling_args)
        a = time.time()
        results_2 = model.advi_sample_model(sampling_args=sampling_args)
        b = time.time()

        assert b - a < 2  # should be very quick

        np.testing.assert_array_equal(
            results_1.approximation.hist, results_2.approximation.hist
        )

        for p in ["μ_g", "μ_α", "α_s", "β_l"]:
            np.testing.assert_array_equal(
                results_1.prior_predictive[p], results_2.prior_predictive[p]
            )
