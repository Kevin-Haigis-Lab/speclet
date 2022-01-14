from itertools import product

import arviz as az
import numpy as np
import pandas as pd
import pytest
from numpy.random import standard_normal

from speclet.analysis import arviz_analysis as azanal


@pytest.fixture(scope="module")
def mock_data() -> pd.DataFrame:
    x = np.random.uniform(0, 100, 300)
    y = 1 + 2 * x + np.random.normal(0, 0.1, len(x))
    return pd.DataFrame({"x": x, "y": y})


class TestSummarizePosteriorPredictions:
    def test_columnnames(self) -> None:
        ppc_df = azanal.summarize_posterior_predictions(standard_normal((100, 200)))
        expected_columns = ["pred_mean", "pred_hdi_low", "pred_hdi_high"]
        for col in expected_columns:
            assert col in ppc_df.columns

    def test_hdi_parameter(self, mock_data: pd.DataFrame) -> None:
        post_shape = (mock_data.shape[0], 100)
        ppc_df_low = azanal.summarize_posterior_predictions(
            standard_normal(post_shape), hdi_prob=0.5
        )
        ppc_df_high = azanal.summarize_posterior_predictions(
            standard_normal(post_shape), hdi_prob=0.99
        )

        np.testing.assert_array_less(
            ppc_df_high["pred_hdi_low"], ppc_df_low["pred_hdi_low"]
        )
        np.testing.assert_array_less(
            ppc_df_low["pred_hdi_high"], ppc_df_high["pred_hdi_high"]
        )

    def test_data_merging(self, mock_data: pd.DataFrame) -> None:
        ppc_df = azanal.summarize_posterior_predictions(
            standard_normal((mock_data.shape[0], 100)), merge_with=mock_data
        )
        expected_columns = ["x", "y", "pred_mean"]
        for col in expected_columns:
            assert col in ppc_df.columns

    def test_calc_error(self, mock_data: pd.DataFrame) -> None:
        y_pred = standard_normal((mock_data.shape[0], 100))
        ppc_df = azanal.summarize_posterior_predictions(y_pred)
        assert "error" not in ppc_df.columns

        ppc_df = azanal.summarize_posterior_predictions(y_pred, calc_error=True)
        assert "error" not in ppc_df.columns

        ppc_df = azanal.summarize_posterior_predictions(
            y_pred, calc_error=True, observed_y="y_pred"
        )
        assert "error" not in ppc_df.columns

        ppc_df = azanal.summarize_posterior_predictions(y_pred, merge_with=mock_data)
        assert "error" not in ppc_df.columns

        ppc_df = azanal.summarize_posterior_predictions(
            y_pred, merge_with=mock_data, calc_error=True
        )
        assert "error" not in ppc_df.columns

        with pytest.raises(TypeError):
            ppc_df = azanal.summarize_posterior_predictions(
                y_pred,
                merge_with=mock_data,
                calc_error=True,
                observed_y="Not a real column",
            )

        ppc_df = azanal.summarize_posterior_predictions(
            y_pred,
            merge_with=mock_data,
            calc_error=True,
            observed_y="y",
        )
        assert "error" in ppc_df.columns


def test_extract_matrix_variable_indices() -> None:

    n_i = 3
    n_j = 4
    i = list(range(n_i))
    i_groups = np.array([f"i_{x}" for x in i])
    j = list(range(n_j))
    j_groups = np.array([f"j_{x}" for x in j])
    var = [f"[{i},{j}]" for i, j in product(i, j)]
    post_summary = pd.DataFrame({"index": var})

    summary = azanal.extract_matrix_variable_indices(
        post_summary,
        col="index",
        idx1=i_groups,
        idx2=j_groups,
        idx1name="i",
        idx2name="j",
    )

    np.testing.assert_equal(
        summary["i"].values.astype(str),
        np.repeat(i_groups, int(len(summary) / len(i_groups))).astype(str),
    )

    np.testing.assert_equal(
        summary["j"].values.astype(str),
        np.tile(j_groups, int(len(summary) / len(j_groups))).astype(str),
    )


def test_get_hdi_colnames_from_az_summary(centered_eight_post: pd.DataFrame) -> None:
    hdi_cols = azanal.get_hdi_colnames_from_az_summary(centered_eight_post)
    assert hdi_cols == ("hdi_3%", "hdi_97%")


@pytest.mark.parametrize(
    "az_obj_name", ["centered_eight", "non_centered_eight", "radon", "rugby"]
)
def test_describe_mcmc(az_obj_name: str) -> None:
    az_obj = az.load_arviz_data(az_obj_name)
    assert isinstance(az_obj, az.InferenceData)
    mcmc_desc = azanal.describe_mcmc(az_obj, plot=False)
    assert mcmc_desc is not None
