#!/usr/bin/env python3

"""Managers of model data."""

import abc
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import src.modeling.simulation_based_calibration_helpers as sbc
from src.data_processing import achilles as achelp
from src.io import data_io


class DataManager(abc.ABC):
    """Abstract base class for the data managers."""

    debug: bool
    data: Optional[pd.DataFrame] = None

    @abc.abstractmethod
    def __init__(self, debug: bool = False) -> None:
        """Initialize the data manager.

        Args:
            debug (bool, optional): Should the debugging data be used? Defaults to
              False.
        """
        pass

    @abc.abstractmethod
    def get_data_path(self) -> Path:
        """Location of the data.

        Returns:
            Path: Path to data.
        """
        pass

    @abc.abstractmethod
    def get_batch_size(self) -> int:
        """Batch size for ADVI depending on debug mode and data set size.

        Returns:
            int: The batch size for fitting with ADVI.
        """
        pass

    @abc.abstractmethod
    def get_data(self) -> pd.DataFrame:
        """Return the intended data object.

        Make sure to account for debug.

        Returns:
            pd.DataFrame: The data frame for modeling.
        """
        pass

    @abc.abstractmethod
    def generate_mock_data(
        self, size: Union[sbc.MockDataSizes, str], random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data to be used for testing or SBC.

        Args:
            size (Union[MockDataSizes, str]): Size of the final mock dataset.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            pd.DataFrame: Mock data.
        """
        pass


class CrcDataManager(DataManager):
    """Manager for CRC modeling data."""

    debug: bool
    data: Optional[pd.DataFrame] = None

    def __init__(self, debug: bool = False):
        """Create a CRC data manager.

        Args:
            debug (bool, optional): Are you in debug mode? Defaults to False.
        """
        self.debug = debug

    def get_data_path(self) -> Path:
        """Get the path for the data set to use.

        Returns:
            Path: Path to the data.
        """
        if self.debug:
            return data_io.data_path(to=data_io.DataFile.crc_subsample)
        return data_io.data_path(to=data_io.DataFile.crc_data)

    def get_batch_size(self) -> int:
        """Decide on the minibatch size for modeling CRC data.

        Returns:
            int: Batch size.
        """
        if self.debug:
            return 1000
        else:
            return 10000

    def _get_sgrnas_that_map_to_multiple_genes(self, df: pd.DataFrame) -> np.ndarray:
        return (
            achelp.make_sgrna_to_gene_mapping_df(df)
            .groupby(["sgrna"])["hugo_symbol"]
            .count()
            .reset_index()
            .query("hugo_symbol > 1")["sgrna"]
            .unique()
        )

    def _drop_sgrnas_that_map_to_multiple_genes(self, df: pd.DataFrame) -> pd.DataFrame:
        sgrnas_to_remove = self._get_sgrnas_that_map_to_multiple_genes(df)
        df_new = df.copy()[~df["sgrna"].isin(sgrnas_to_remove)]
        return df_new

    def _drop_missing_copynumber(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()[~df["gene_cn"].isna()]
        return df_new

    def _load_data(self) -> pd.DataFrame:
        """Load CRC data."""
        df = achelp.read_achilles_data(self.get_data_path(), low_memory=False)
        df = self._drop_sgrnas_that_map_to_multiple_genes(df)
        df = self._drop_missing_copynumber(df)
        df = achelp.set_achilles_categorical_columns(df)
        return df

    def get_data(self) -> pd.DataFrame:
        """Get the data for modeling.

        If the data is not already loaded, it is first read from disk.
        """
        if self.data is None:
            self.data = self._load_data()
        return self.data

    def generate_mock_data(
        self, size: Union[sbc.MockDataSizes, str], random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data to be used for testing or SBC.

        Args:
            size (Union[MockDataSizes, str]): Size of the final mock dataset.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            pd.DataFrame: Mock data.
        """
        if isinstance(size, str):
            size = sbc.MockDataSizes(size)

        if size == sbc.MockDataSizes.small:
            self.data = sbc.generate_mock_achilles_data(
                n_genes=10, n_sgrnas_per_gene=3, n_cell_lines=5, n_batches=2
            )
        else:
            self.data = sbc.generate_mock_achilles_data(
                n_genes=100, n_sgrnas_per_gene=5, n_cell_lines=20, n_batches=3
            )
        return self.data


class MockDataManager(DataManager):
    """A data manager with mock data (primarily for testing)."""

    def __init__(self, debug: bool = False) -> None:
        """Initialize a MockDataManager.

        This DataManager makes a small data set for testing and demo purpose.

        Args:
            debug (bool, optional): Should the debugging data be used? Defaults to
              False.
        """
        self.debug = debug

    def get_data_path(self) -> Path:
        """Location of the data.

        Returns:
            Path: Path to data.
        """
        return Path("/dev/null")

    def get_batch_size(self) -> int:
        """Batch size for ADVI depending on debug mode and data set size.

        Returns:
            int: The batch size for fitting with ADVI.
        """
        return 10 if self.debug else 20

    def get_data(self) -> pd.DataFrame:
        """Return the intended data object.

        Make sure to account for debug.

        Returns:
            pd.DataFrame: The data frame for modeling.
        """
        n_data_points = 50 if self.debug else 100
        x = np.random.uniform(-1, 1, n_data_points)
        y = -1 + 2 * x + (np.random.randn(n_data_points) / 2.0)
        return pd.DataFrame({"x": x, "y": y})

    def generate_mock_data(
        self, size: Union[sbc.MockDataSizes, str], random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data to be used for testing or SBC.

        Args:
            size (Union[MockDataSizes, str]): Size of the final mock dataset.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            pd.DataFrame: Mock data.
        """
        if isinstance(size, str):
            size = sbc.MockDataSizes(size)

        if size == sbc.MockDataSizes.small:
            self.debug = True
        else:
            self.debug = False
        return self.get_data()
