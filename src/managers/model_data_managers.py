#!/usr/bin/env python3

"""Managers of model data."""

import abc
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

import src.modeling.simulation_based_calibration_helpers as sbc
from src.data_processing import achilles as achelp
from src.io import data_io
from src.loggers import logger

DataFrameTransformations = List[Callable[[pd.DataFrame], pd.DataFrame]]


class DataManager(abc.ABC):
    """Abstract base class for the data managers."""

    debug: bool
    transformations: DataFrameTransformations
    _data: Optional[pd.DataFrame] = None

    @abc.abstractmethod
    def __init__(
        self,
        debug: bool = False,
        transformations: Optional[DataFrameTransformations] = None,
    ) -> None:
        """Initialize the data manager.

        Args:
            debug (bool, optional): Should the debugging data be used? Defaults to
              False.
            transformations (Optional[DataFrameTransformations], optional): List of
              callable functions or classes for transforming the data. Defaults to None
              (an empty list).
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
    def set_data(self, new_data: Optional[pd.DataFrame]) -> None:
        """Set the data object (can be `None`).

        Args:
            new_data (Optional[pd.DataFrame]): New data object.
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

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored transformations to the data set.

        Args:
            df (pd.DataFrame): The data on which to operate.

        Returns:
            pd.DataFrame: The transformed data set.
        """
        if len(self.transformations) == 0:
            return df

        for transform in self.transformations:
            df = transform(df)

        return df

    def add_transformations(
        self, new_trans: List[Callable[[pd.DataFrame], pd.DataFrame]]
    ):
        """Add new data transformations.

        Args:
            new_trans (List[Callable[[pd.DataFrame], pd.DataFrame]]): A list of
            callables to be used to transform the data. Each transformation must take a
            pandas DataFrame and return a pandas DataFrame.
        """
        self.transformations += new_trans

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """The data object.

        Returns:
            Optional[pd.DataFrame]: If the data has been loaded, the pandas DataFrame,
            else `None`.
        """
        return self._data

    @data.setter
    def data(self, new_data: Optional[pd.DataFrame]) -> None:
        """Set the data object (can be `None`).

        Calls `self.set_data()` so that the subclass can have specific behavior when
        setting the data.

        Args:
            new_data (Optional[pd.DataFrame]): The new data object.
        """
        self.set_data(new_data=new_data)


class CrcDataManager(DataManager):
    """Manager for CRC modeling data."""

    def __init__(
        self,
        debug: bool = False,
        transformations: Optional[DataFrameTransformations] = None,
    ) -> None:
        """Create a CRC data manager.

        Args:
            debug (bool, optional): Are you in debug mode? Defaults to False.
            transformations (Optional[DataFrameTransformations], optional): List of
              callable functions or classes for transforming the data. Defaults to None
              (an empty list).
        """
        self.debug = debug
        if transformations is None:
            self.transformations = []
        else:
            self.transformations = transformations

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
        logger.warning(
            f"Dropping {len(sgrnas_to_remove)} sgRNA that map to multiple genes."
        )
        df_new = df.copy()[~df["sgrna"].isin(sgrnas_to_remove)]
        return df_new

    def _drop_missing_copynumber(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()[~df["copy_number"].isna()]
        size_diff = df.shape[0] - df_new.shape[0]
        logger.warning(f"Dropping {size_diff} data points with missing copy number.")
        return df_new

    def _load_data(self) -> pd.DataFrame:
        """Load CRC data."""
        logger.debug("Reading data from file.")
        df = achelp.read_achilles_data(self.get_data_path(), low_memory=False)
        df = self._drop_sgrnas_that_map_to_multiple_genes(df)
        df = self._drop_missing_copynumber(df)
        df = achelp.set_achilles_categorical_columns(df)
        return df

    def get_data(self) -> pd.DataFrame:
        """Get the data for modeling.

        If the data is not already loaded, it is first read from disk.
        """
        logger.debug("Retrieving data.")
        if self._data is None:
            self._data: pd.DataFrame = self._load_data().pipe(
                self.apply_transformations
            )
        return self._data

    def set_data(self, new_data: Optional[pd.DataFrame]) -> None:
        """Set the new data set and apply the transofrmations automatically.

        Args:
            new_data (Optional[pd.DataFrame]): New data set.
        """
        if new_data is None:
            self._data = None
        else:
            self._data = self.apply_transformations(new_data)

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
        logger.info(f"Generating mock data of size '{size}'.")
        if isinstance(size, str):
            size = sbc.MockDataSizes(size)

        if size == sbc.MockDataSizes.small:
            self.data = sbc.generate_mock_achilles_data(
                n_genes=10,
                n_sgrnas_per_gene=3,
                n_cell_lines=5,
                n_lineages=2,
                n_batches=2,
                n_screens=1,
                n_kras_types=2,
            )
        elif size == sbc.MockDataSizes.medium:
            self.data = sbc.generate_mock_achilles_data(
                n_genes=25,
                n_sgrnas_per_gene=5,
                n_cell_lines=12,
                n_lineages=2,
                n_batches=3,
                n_screens=2,
                n_kras_types=3,
            )
        else:
            self.data = sbc.generate_mock_achilles_data(
                n_genes=100,
                n_sgrnas_per_gene=5,
                n_cell_lines=20,
                n_lineages=3,
                n_batches=4,
                n_screens=2,
                n_kras_types=3,
            )
        return self.data


class MockDataManager(DataManager):
    """A data manager with mock data (primarily for testing)."""

    def __init__(
        self,
        debug: bool = False,
        transformations: Optional[DataFrameTransformations] = None,
    ) -> None:
        """Initialize a MockDataManager.

        This DataManager makes a small data set for testing and demo purpose.

        Args:
            debug (bool, optional): Should the debugging data be used? Defaults to
              False.
            transformations (Optional[DataFrameTransformations], optional): List of
              callable functions or classes for transforming the data. Defaults to None
              (an empty list).
        """
        self.debug = debug
        if transformations is None:
            self.transformations = []
        else:
            self.transformations = transformations

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
        if self.data is None:
            logger.info("Creating data for mock data manager.")
            n_data_points = 50 if self.debug else 100
            x = np.random.uniform(-1, 1, n_data_points)
            y = -1 + 2 * x + (np.random.randn(n_data_points) / 2.0)
            self.data = pd.DataFrame({"x": x, "y": y})
        return self.data

    def set_data(self, new_data: Optional[pd.DataFrame]) -> None:
        """Set the new data set and apply the transofrmations automatically.

        Args:
            new_data (Optional[pd.DataFrame]): New data set.
        """
        if new_data is None:
            self._data = None
        else:
            self._data = self.apply_transformations(new_data)

    def generate_mock_data(
        self, size: Union[sbc.MockDataSizes, str], random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data to be used for testing or SBC.

        Args:
            size (Union[MockDataSizes, str]): Size of the final mock dataset.
            random_seed (Optional[int], optional): Not used. Defaults to None.

        Returns:
            pd.DataFrame: Mock data.
        """
        logger.debug("Generating mock data.")
        logger.info("This method just calls `self.get_data()` in the MockDataManager.")
        if isinstance(size, str):
            size = sbc.MockDataSizes(size)

        if size == sbc.MockDataSizes.small:
            self.debug = True
        else:
            self.debug = False
        return self.get_data()
