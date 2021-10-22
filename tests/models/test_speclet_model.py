from pathlib import Path
from time import time
from typing import Any, Final, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import pytest

from src.exceptions import CacheDoesNotExistError
from src.managers.data_managers import CrisprScreenDataManager
from src.misc.test_helpers import do_nothing
from src.modeling import simulation_based_calibration_helpers as sbc
from src.modeling.pymc3_sampling_api import ApproximationSamplingResults
from src.models import speclet_model
from src.project_enums import MockDataSize, ModelFitMethod

TEST_DATA: Final[Path] = Path("tests", "depmap_test_data.csv")


@pytest.fixture
def mock_crispr_screen_dm() -> CrisprScreenDataManager:
    return CrisprScreenDataManager(TEST_DATA)


class MinimalSpecletModel(speclet_model.SpecletModel):
    def model_specification(self) -> tuple[pm.Model, str]:
        return "my-mock-model", "mock-observed_var"


def make_x_and_y_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["x"] = np.random.normal(0, 2, df.shape[0])
    df["y"] = np.random.normal(0, 0.2, df.shape[0]) + 1.5 * (-0.5 * df["x"].values)
    return df


class MockSpecletModelClass(speclet_model.SpecletModel):
    def __init__(self, name: str, root_cache_dir: Path) -> None:
        _data_manager = CrisprScreenDataManager(
            TEST_DATA, transformations=[make_x_and_y_cols]
        )
        super().__init__(name, _data_manager, root_cache_dir=root_cache_dir)

    def model_specification(self) -> tuple[pm.Model, str]:
        data = self.data_manager.get_data()
        with pm.Model() as model:
            b = pm.Normal("b", 0, 10)
            a = pm.Normal("a", 0, 10)
            sigma = pm.HalfNormal("sigma", 10)
            y = pm.Normal(  # noqa: F841
                "y",
                a + b * data["x"].values,
                sigma,
                observed=data["y"].values,
            )
        return model, "y"


def assert_posterior_shape(res: az.InferenceData, shape: tuple[int, int]) -> None:
    for p in ["a", "b", "sigma"]:
        assert res.posterior[p].shape == shape


class TestSpecletModel:
    @pytest.fixture(scope="function")
    def mock_sp_model(self, tmp_path: Path) -> MockSpecletModelClass:
        return MockSpecletModelClass(
            name="test-model",
            root_cache_dir=tmp_path,
        )

    def test_mcmc_sample_model_fails_without_overriding(
        self, tmp_path: Path, mock_crispr_screen_dm: CrisprScreenDataManager
    ) -> None:
        sp = MinimalSpecletModel(
            "test-model", data_manager=mock_crispr_screen_dm, root_cache_dir=tmp_path
        )
        with pytest.raises(AttributeError, match="Cannot sample: model is 'None'"):
            sp.mcmc_sample_model()

    @pytest.mark.slow
    def test_build_model(self, mock_sp_model: MockSpecletModelClass) -> None:
        assert mock_sp_model.model is None
        mock_sp_model.build_model()
        assert mock_sp_model.model is not None
        assert isinstance(mock_sp_model.model, pm.Model)

    @pytest.mark.slow
    def test_mcmc_sample_model(self, mock_sp_model: MockSpecletModelClass) -> None:
        mock_sp_model.build_model()
        mcmc_res = mock_sp_model.mcmc_sample_model(
            prior_pred_samples=25,
            random_seed=1,
            sample_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 2,
                "cores": 1,
            },
        )
        assert isinstance(mcmc_res, az.InferenceData)
        assert_posterior_shape(mcmc_res, (2, 50))

        assert mock_sp_model.mcmc_results is not None

        tic = time()
        mcmc_res_2 = mock_sp_model.mcmc_sample_model(
            prior_pred_samples=25,
            random_seed=1,
            sample_kwargs={
                "draws": 50,
                "tune": 50,
                "chains": 2,
                "cores": 1,
            },
        )
        toc = time()

        assert mcmc_res is mcmc_res_2
        assert toc - tic < 1
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(
                mcmc_res.posterior[p], mcmc_res_2.posterior[p]
            )

    def test_advi_sample_model_fails_without_model(
        self, tmp_path: Path, mock_crispr_screen_dm: CrisprScreenDataManager
    ) -> None:
        sp = MinimalSpecletModel(
            "test-model", data_manager=mock_crispr_screen_dm, root_cache_dir=tmp_path
        )
        with pytest.raises(AttributeError, match="model is 'None'"):
            sp.advi_sample_model()

    @pytest.mark.slow
    def test_advi_sample_model(self, mock_sp_model: MockSpecletModelClass) -> None:
        mock_sp_model.build_model()
        advi_res = mock_sp_model.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=25,
            random_seed=1,
        )
        assert_posterior_shape(advi_res.inference_data, (1, 100))

        assert mock_sp_model.advi_results is not None

        tic = time()
        advi_res_2 = mock_sp_model.advi_sample_model(
            n_iterations=100,
            draws=100,
            prior_pred_samples=25,
            random_seed=1,
        )
        toc = time()

        assert advi_res is advi_res_2
        assert toc - tic < 1
        for p in ["a", "b", "sigma"]:
            np.testing.assert_array_equal(
                advi_res.inference_data.posterior[p],
                advi_res_2.inference_data.posterior[p],
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("fit_method", [ModelFitMethod.ADVI, ModelFitMethod.MCMC])
    def test_run_simulation_based_calibration(
        self,
        tmp_path: Path,
        fit_method: ModelFitMethod,
        monkeypatch: pytest.MonkeyPatch,
        centered_eight: az.InferenceData,
    ) -> None:
        def mock_mcmc(*args: Any, **kwargs: Any) -> az.InferenceData:
            return centered_eight.copy()

        def mock_advi(*args: Any, **kwargs: Any) -> ApproximationSamplingResults:
            return ApproximationSamplingResults(centered_eight.copy(), [1, 2, 3])

        monkeypatch.setattr(
            speclet_model.SpecletModel, "update_observed_data", do_nothing
        )
        monkeypatch.setattr(speclet_model.SpecletModel, "mcmc_sample_model", mock_mcmc)
        monkeypatch.setattr(speclet_model.SpecletModel, "advi_sample_model", mock_advi)
        sp = MockSpecletModelClass("test-model", root_cache_dir=tmp_path)
        assert sp.model is None

        fit_kwargs: dict[str, Union[float, int]]

        if fit_method == ModelFitMethod.ADVI:
            fit_kwargs = {"n_iterations": 100, "draws": 10}
        else:
            fit_kwargs = {
                "draws": 10,
                "tune": 10,
                "chains": 2,
                "cores": 1,
                "target_accept": 0.8,
            }

        fit_kwargs["prior_pred_samples"] = 10

        sp.run_simulation_based_calibration(
            results_path=tmp_path,
            fit_method=fit_method,
            size=MockDataSize.SMALL,
            fit_kwargs=fit_kwargs,
        )
        assert sp.model is not None
        assert (tmp_path / "inference-data.netcdf").exists()

    def _touch_sbc_results_files(self, sbc_fm: sbc.SBCFileManager) -> None:
        for f in (
            sbc_fm.inference_data_path,
            sbc_fm.priors_path_get,
            sbc_fm.posterior_summary_path,
        ):
            f.touch()

    def test_get_sbc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_crispr_screen_dm: CrisprScreenDataManager,
        iris: pd.DataFrame,
        centered_eight: az.InferenceData,
        centered_eight_post: pd.DataFrame,
    ) -> None:
        # setup SBC results
        sbc_dir = tmp_path / "sbc-results-dir"
        if not sbc_dir.exists():
            sbc_dir.mkdir()
        sbc_fm = sbc.SBCFileManager(sbc_dir)
        sbc_fm.save_sbc_data(iris)
        self._touch_sbc_results_files(sbc_fm)

        def mock_get_sbc_results(*args: Any, **kwargs: Any) -> sbc.SBCResults:
            return sbc.SBCResults(
                priors={},
                inference_obj=centered_eight,
                posterior_summary=centered_eight_post,
            )

        def just_return(x: Any, df: pd.DataFrame) -> pd.DataFrame:
            return df

        monkeypatch.setattr(sbc.SBCFileManager, "get_sbc_results", mock_get_sbc_results)
        monkeypatch.setattr(speclet_model.SpecletModel, "build_model", do_nothing)
        monkeypatch.setattr(mock_crispr_screen_dm, "apply_transformations", just_return)

        sp = MinimalSpecletModel(
            "testing-get-sbc",
            mock_crispr_screen_dm,
            root_cache_dir=tmp_path,
        )
        sim_df, sbc_res, sim_sbc_fm = sp.get_sbc(sbc_dir)
        assert isinstance(sim_df, pd.DataFrame)
        assert sim_df.shape == iris.shape
        assert isinstance(sbc_res, sbc.SBCResults)
        assert isinstance(sim_sbc_fm, sbc.SBCFileManager)
        assert sim_sbc_fm.dir == sbc_fm.dir

    def test_get_sbc_errors(
        self, tmp_path: Path, mock_crispr_screen_dm: CrisprScreenDataManager
    ) -> None:
        sp_model = MinimalSpecletModel(
            "testing-model", mock_crispr_screen_dm, root_cache_dir=tmp_path
        )
        sbc_dir = tmp_path / "sbc-results"
        if not sbc_dir.exists():
            sbc_dir.mkdir()
        sbc_fm = sbc.SBCFileManager(sbc_dir)

        with pytest.raises(CacheDoesNotExistError):
            _ = sp_model.get_sbc(sbc_dir)

        sbc_fm.sbc_data_path.touch()
        with pytest.raises(CacheDoesNotExistError):
            _ = sp_model.get_sbc(sbc_dir)

        sbc_fm.sbc_data_path.unlink()
        self._touch_sbc_results_files(sbc_fm)
        with pytest.raises(CacheDoesNotExistError):
            _ = sp_model.get_sbc(sbc_dir)
