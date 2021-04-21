#!/usr/bin/env python3

"""Builders for CRC CERES Mimic model."""

from pathlib import Path
from typing import Dict, Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import theano
from theano.tensor.sharedvar import TensorSharedVariable as TTShared

import src.modeling.simulation_based_calibration_helpers as sbc
from src.data_processing import achilles as achelp
from src.modeling import pymc3_sampling_api as pmapi
from src.modeling.sampling_metadata_models import SamplingArguments
from src.models.crc_model import CrcModel
from src.models.protocols import SelfSufficientModel

#### ---- CRC CERES Mimic ---- ####


class CrcCeresMimic(CrcModel, SelfSufficientModel):
    """CRC CERES Mimic.

    This model is just the part of the CERES model that includes the sgRNA
    "activity score" (q) and the per-gene (h) and per-gene-per-cell-line (d) covariates.
    In addition, I have included a parameter for pDNA batch.
    """

    shared_vars: Optional[Dict[str, TTShared]] = None
    model: Optional[pm.Model] = None
    advi_results: Optional[pmapi.ApproximationSamplingResults] = None
    mcmc_results: Optional[pmapi.MCMCSamplingResults] = None

    ReplacementsDict = Dict[TTShared, Union[pm.Minibatch, np.ndarray]]

    _copynumber_cov: bool
    _sgrna_intercept_cov: bool

    def __init__(
        self,
        name: str,
        root_cache_dir: Optional[Path] = None,
        debug: bool = False,
        copynumber_cov: bool = False,
        sgrna_intercept_cov: bool = False,
    ):
        """Create a CrcCeresMimic object.

        Args:
            name (str): A unique identifier for this instance of CrcCeresMimic. (Used
              for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None.
            debug (bool, optional): Are you in debug mode? Defaults to False.
            copynumber_cov (bool, optional): Should the gene copy number covariate be
              included in the model? Default to False.
            sgrna_intercept_cov (bool, optional): Should a varying intercept for
              `sgRNA|gene` be included in the model? Default to False.
        """
        super().__init__(
            name="ceres-mimic-1_" + name, root_cache_dir=root_cache_dir, debug=debug
        )
        self._copynumber_cov = copynumber_cov
        self._sgrna_intercept_cov = sgrna_intercept_cov

    def _reset_model_afer_change_to_optional_covariate(self) -> None:
        """Reset some attributes after the PyMC3 model has changed."""
        self.model = None
        self.advi_results = None
        self.mcmc_results = None

    @property
    def copynumber_cov(self) -> bool:
        """Get the current value of `copynumber_cov` attribute.

        Returns:
            bool: Whether or not the copy number covariate is included in the model.
        """
        return self._copynumber_cov

    @copynumber_cov.setter
    def copynumber_cov(self, new_value: bool):
        """Set the value for the `copynumber_cov` attribute.

        If the value changes, then the `model` attribute and model results attributes
        `advi_results` and `mcmc_results` are all reset to None.

        Args:
            new_value (bool): Whether or not the copy number covariate should be
              included in the model.
        """
        if new_value != self._copynumber_cov:
            self._reset_model_afer_change_to_optional_covariate()
            self._copynumber_cov = new_value

    @property
    def sgrna_intercept_cov(self) -> bool:
        """Get the current value of `sgrna_intercept_cov` attribute.

        Returns:
            bool: Whether or not the `sgRNA|gene` varying intercept covariate is
              included in the model.
        """
        return self._sgrna_intercept_cov

    @sgrna_intercept_cov.setter
    def sgrna_intercept_cov(self, new_value: bool):
        """Set the value for the `sgrna_intercept_cov` attribute.

        If the value changes, then the `model` attribute and model results attributes
        `advi_results` and `mcmc_results` are all reset to None.

        Args:
            new_value (bool): Whether or not the `sgRNA|gene` varying intercept
              covariate is included in the model.
        """
        if new_value != self._sgrna_intercept_cov:
            self._reset_model_afer_change_to_optional_covariate()
            self._sgrna_intercept_cov = new_value

    def _get_indices_collection(self, data: pd.DataFrame) -> achelp.CommonIndices:
        return achelp.common_indices(data)

    def build_model(self, data: Optional[pd.DataFrame] = None) -> None:
        """Build CRC CERES Mimic One.

        Args:
            data (Optional[pd.DataFrame], optional): Data to used to build the model
              around. If None (default), then Achilles data is read in. Defaults to
              None.

        Returns:
            [type]: None
        """
        if data is None:
            data = self.get_data()

        total_size = data.shape[0]
        indices_collection = self._get_indices_collection(data)

        # Shared Theano variables
        sgrna_idx_shared = theano.shared(indices_collection.sgrna_idx)
        sgrna_to_gene_idx_shared = theano.shared(indices_collection.sgrna_to_gene_idx)
        gene_idx_shared = theano.shared(indices_collection.gene_idx)
        cellline_idx_shared = theano.shared(indices_collection.cellline_idx)
        batch_idx_shared = theano.shared(indices_collection.batch_idx)
        lfc_shared = theano.shared(data.lfc.values)
        copynumber_shared = theano.shared(data.z_log2_cn.values)

        with pm.Model() as model:

            # Hyper-priors
            σ_a = pm.HalfNormal("σ_a", np.array([0.1, 2]), shape=2)
            a = pm.Exponential("a", σ_a, shape=(indices_collection.n_genes, 2))

            μ_h = pm.Normal("μ_h", 0, 0.2)
            σ_h = pm.HalfNormal("σ_h", 1)

            μ_d = pm.Normal("μ_d", 0, 0.2)
            σ_d = pm.HalfNormal("σ_d", 1)

            μ_η = pm.Normal("μ_η", 0, 0.1)
            σ_η = pm.HalfNormal("σ_η", 0.1)

            # Main parameter priors
            q = pm.Beta(
                "q",
                alpha=a[sgrna_to_gene_idx_shared, 0],
                beta=a[sgrna_to_gene_idx_shared, 1],
                shape=indices_collection.n_sgrnas,
            )
            h = pm.Normal("h", μ_h, σ_h, shape=indices_collection.n_genes)
            d = pm.Normal(
                "d",
                μ_d,
                σ_d,
                shape=(indices_collection.n_genes, indices_collection.n_celllines),
            )
            η = pm.Normal("η", μ_η, σ_η, shape=indices_collection.n_batches)

            gene_comp = h[gene_idx_shared] + d[gene_idx_shared, cellline_idx_shared]

            if self.copynumber_cov:
                # Add varying slope with copy number.
                μ_β = pm.Normal("μ_β", 0, 0.1)
                σ_β = pm.HalfNormal("σ_β", 0.5)
                β = pm.Normal("β", μ_β, σ_β, shape=indices_collection.n_celllines)
                gene_comp += β[cellline_idx_shared] * copynumber_shared

            μ = pm.Deterministic(
                "μ", q[sgrna_idx_shared] * gene_comp + η[batch_idx_shared]
            )

            if self.sgrna_intercept_cov:
                # Hyper priors for sgRNA|gene varying intercept.
                μ_og = pm.Normal("μ_og", 0, 0.1)
                σ_og = pm.HalfNormal("σ_og", 0.2)
                # Priors for sgRNA|gene varying intercept.
                μ_o = pm.Normal("μ_o", μ_og, σ_og, shape=indices_collection.n_genes)
                σ_o = pm.HalfNormal("σ_o", 0.2)
                # sgRNA|gene varying intercept.
                o = pm.Normal(
                    "o",
                    μ_o[sgrna_to_gene_idx_shared],
                    σ_o,
                    shape=indices_collection.n_sgrnas,
                )
                μ = μ + o[sgrna_idx_shared]

            σ = pm.HalfNormal("σ", 2)

            # Likelihood
            lfc = pm.Normal(  # noqa: F841
                "lfc", μ, σ, observed=lfc_shared, total_size=total_size
            )

        shared_vars = {
            "sgrna_idx_shared": sgrna_idx_shared,
            "sgrna_to_gene_idx_shared": sgrna_to_gene_idx_shared,
            "gene_idx_shared": gene_idx_shared,
            "cellline_idx_shared": cellline_idx_shared,
            "batch_idx_shared": batch_idx_shared,
            "lfc_shared": lfc_shared,
            "copynumber_shared": copynumber_shared,
        }

        self.model = model
        self.shared_vars = shared_vars
        return None

    def _get_replacement_parameters(self) -> ReplacementsDict:
        if self.data is None:
            raise AttributeError(
                "Cannot create replacement parameters before data has been loaded."
            )
        if self.shared_vars is None:
            raise AttributeError(
                "No shared variables - cannot create replacement parameters.."
            )

        batch_size = self.get_batch_size()
        indices = self._get_indices_collection(self.data)

        sgrna_idx_batch = pm.Minibatch(indices.sgrna_idx, batch_size=batch_size)
        gene_idx_batch = pm.Minibatch(indices.gene_idx, batch_size=batch_size)
        cellline_idx_batch = pm.Minibatch(indices.cellline_idx, batch_size=batch_size)
        batch_idx_batch = pm.Minibatch(indices.batch_idx, batch_size=batch_size)
        lfc_data_batch = pm.Minibatch(self.data.lfc.values, batch_size=batch_size)
        copynumber_data_batch = pm.Minibatch(
            self.data.z_log2_cn.values, batch_size=batch_size
        )

        return {
            self.shared_vars["sgrna_idx_shared"]: sgrna_idx_batch,
            self.shared_vars["gene_idx_shared"]: gene_idx_batch,
            self.shared_vars["cellline_idx_shared"]: cellline_idx_batch,
            self.shared_vars["batch_idx_shared"]: batch_idx_batch,
            self.shared_vars["lfc_shared"]: lfc_data_batch,
            self.shared_vars["copynumber_shared"]: copynumber_data_batch,
        }

    def mcmc_sample_model(
        self,
        sampling_args: SamplingArguments,
        mcmc_draws: int = 10000,
        tune: int = 1000,
        chains: int = 3,
    ) -> pmapi.MCMCSamplingResults:
        """Fit the model with MCMC.

        Args:
            sampling_args (SamplingArguments): Arguments for the sampling procedure.
        """
        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )
        if self.shared_vars is None:
            raise AttributeError("Cannot sample: cannot find shared variables.")

        if self.mcmc_results is not None:
            return self.mcmc_results

        if not sampling_args.ignore_cache and self.mcmc_cache_exists():
            self.mcmc_results = self.get_mcmc_cache(model=self.model)
            return self.mcmc_results

        self.mcmc_results = pmapi.pymc3_sampling_procedure(
            model=self.model,
            mcmc_draws=mcmc_draws,
            tune=tune,
            chains=chains,
            cores=sampling_args.cores,
            random_seed=sampling_args.random_seed,
        )
        self.write_mcmc_cache(self.mcmc_results)
        return self.mcmc_results

    def advi_sample_model(
        self,
        sampling_args: SamplingArguments,
        n_iterations: int = 100000,
        draws: int = 1000,
    ) -> pmapi.ApproximationSamplingResults:
        """Fit the model with ADVI.

        Args:
            sampling_args (SamplingArguments): Arguments for the sampling procedure.

        Returns:
            ApproximationSamplingResults: The results of fitting the model with ADVI.
        """
        if self.model is None:
            raise AttributeError(
                "Cannot sample: model is 'None'. "
                + "Make sure to run `model.build_model()` first."
            )
        if self.shared_vars is None:
            raise AttributeError("Cannot sample: cannot find shared variables.")

        replacements = self._get_replacement_parameters()

        if self.advi_results is not None:
            return self.advi_results

        if not sampling_args.ignore_cache and self.advi_cache_exists():
            self.advi_results = self.get_advi_cache()
            return self.advi_results

        self.advi_results = pmapi.pymc3_advi_approximation_procedure(
            model=self.model,
            n_iterations=n_iterations,
            draws=draws,
            callbacks=[
                pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
            ],
            random_seed=sampling_args.random_seed,
            fit_kwargs={"more_replacements": replacements},
        )
        self.write_advi_cache(self.advi_results)
        return self.advi_results

    def run_simulation_based_calibration(
        self, results_path: Path, random_seed: Optional[int] = None, size: str = "large"
    ) -> None:
        """Run a round of simulation-based calibration.

        Args:
            results_path (Path): Where to store the results.
            random_seed (Optional[int], optional): Random seed (for reproducibility).
              Defaults to None.
            size (str, optional): Size of the data set to mock. Defaults to "large".
        """
        if size == "large":
            mock_data = sbc.generate_mock_achilles_data(
                n_genes=100, n_sgrnas_per_gene=5, n_cell_lines=20, n_batches=3
            )
        elif size == "small":
            mock_data = sbc.generate_mock_achilles_data(
                n_genes=10, n_sgrnas_per_gene=3, n_cell_lines=5, n_batches=2
            )
        else:
            raise Exception("Unknown value for `size` parameter.")

        self.build_model(data=mock_data)
        assert self.model is not None
        with self.model:
            priors = pm.sample_prior_predictive(samples=1, random_seed=random_seed)

        mock_data["lfc"] = priors.get("lfc").flatten()
        self.data = mock_data

        sampling_args = SamplingArguments(
            name=f"sbc-seed{random_seed}",
            cores=1,
            sample=True,
            ignore_cache=False,
            debug=False,
            random_seed=random_seed,
        )

        res = self.advi_sample_model(sampling_args)
        posterior_summary = az.summary(res.trace, fmt="wide", hdi_prob=0.89)
        assert isinstance(posterior_summary, pd.DataFrame)
        az_results = pmapi.convert_samples_to_arviz(model=self.model, res=res)
        results_manager = sbc.SBCFileManager(dir=results_path)
        results_manager.save_sbc_results(
            priors=priors,
            inference_obj=az_results,
            posterior_summary=posterior_summary,
        )
