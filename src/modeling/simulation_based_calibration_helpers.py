"""Helpers for organizing simualtaion-based calibrations."""

from pathlib import Path
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pandas as pd


class SBCResults:
    """Results from a single round of SBC."""

    priors: Dict[str, Any]
    inference_obj: az.InferenceData
    posterior_summary: pd.DataFrame

    def __init__(
        self,
        priors: Dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ):
        """Create an instance of SBCResults.

        Args:
            priors (Dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        self.priors = priors
        self.inference_obj = inference_obj
        self.posterior_summary = posterior_summary


class SBCFileManager:
    """Manages the results from a round of simulation-based calibration."""

    dir: Path
    inference_data_path: Path
    priors_path_set: Path
    priors_path_get: Path
    posterior_summary_path: Path

    sbc_results: Optional[SBCResults] = None

    def __init__(self, dir: Path):
        """Create a SBCFileManager.

        Args:
            dir (Path): The directory where the data is stored.
        """
        self.dir = dir
        self.inference_data_path = dir / "inference-data.netcdf"
        self.priors_path_set = dir / "priors"
        self.priors_path_get = dir / "priors.npz"
        self.posterior_summary_path = dir / "posterior-summary.csv"

    def save_sbc_results(
        self,
        priors: Dict[str, Any],
        inference_obj: az.InferenceData,
        posterior_summary: pd.DataFrame,
    ) -> None:
        """Save the results from a round of SBC.

        Args:
            priors (Dict[str, Any]): Priors representing the 'true' values.
            inference_obj (az.InferenceData): Fitting results.
            posterior_summary (pd.DataFrame): A summary of the posteriors.
        """
        inference_obj.to_netcdf(self.inference_data_path.as_posix())
        np.savez(self.priors_path_set.as_posix(), **priors)
        posterior_summary.to_csv(
            self.posterior_summary_path.as_posix(), index_label="parameter"
        )

    def _tidy_numy_files(self, files: Any) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        for k in files.files:
            d[k] = files[k]
        return d

    def get_sbc_results(self, re_read: bool = False) -> SBCResults:
        """Retrieve results of a round of SBC.

        Args:
            re_read (bool, optional): Should the results be re-read from file? Defaults to False.

        Returns:
            SBCResults: The results from the round of SBC.
        """
        if self.sbc_results is not None and not re_read:
            return self.sbc_results

        inference_obj = az.from_netcdf(self.inference_data_path)
        priors_files = np.load(self.priors_path_get.as_posix())
        priors = self._tidy_numy_files(priors_files)
        posterior_summary = pd.read_csv(
            self.posterior_summary_path, index_col="parameter"
        )

        self.sbc_results = SBCResults(
            priors=priors,
            inference_obj=inference_obj,
            posterior_summary=posterior_summary,
        )
        return self.sbc_results

    def all_data_exists(self) -> bool:
        """Confirm that all data exists.

        Returns:
            bool: True if all of the data exists, else false.
        """
        for p in [
            self.priors_path_get,
            self.posterior_summary_path,
            self.inference_data_path,
        ]:
            if not p.exists():
                return False
        return True
