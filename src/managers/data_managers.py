"""Data management classes."""

from pathlib import Path
from typing import Any, Callable, Final, Optional, TypeVar, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.data_processing import achilles as achelp
from src.io.data_io import DataFile, data_path
from src.loggers import logger
from src.modeling import mock_data
from src.project_enums import MockDataSize, assert_never

Data = TypeVar("Data", pd.DataFrame, dd.DataFrame)
DataFrameTransformation = Callable[[Data], Data]

common_crispr_screen_transformations: Final[list[DataFrameTransformation]] = [
    achelp.drop_sgrnas_that_map_to_multiple_genes,
    achelp.set_achilles_categorical_columns,
]


def _get_crispr_screen_data_coltypes() -> dict[str, str]:
    col_types = {}
    for c in [
        "sgrna",
        "hugo_symbol",
        "p_dna_batch",
        "replicate_id",
        "depmap_id",
        "sex",
    ]:
        col_types[c] = "object"
    for c in ["counts_final", "counts_final_total"]:
        col_types[c] = "int"
    return col_types


class DataNotLoadedException(BaseException):
    """Data not loaded exception."""

    pass


class CrisprScreenDataManager:
    """Manage CRISPR screen data."""

    _data: Optional[pd.DataFrame]
    data_source: Path
    _transformations: list[DataFrameTransformation]
    columns: Optional[set[str]]
    use_dask: bool

    def __init__(
        self,
        data_source: Union[Path, DataFile],
        transformations: Optional[list[DataFrameTransformation]] = None,
        columns: Optional[set[str]] = None,
        use_dask: bool = False,
    ) -> None:
        """Create a CRISPR screen data manager.

        Args:
            data_source (Union[Path, DataFile]): CSV file with data.
            transformations (Optional[list[DataFrameTransformation]], optional): List of
              functions that take, mutate, and return a data frame (pandas or dask).
              Defaults to None.
            columns (Optional[set[str]], optional): Columns to keep from the data.
              Defaults to None.
            use_dask (bool, optional): Use a dask backend? Defaults to False.
        """
        self._data = None
        self.columns = columns
        self._transformations = [] if transformations is None else transformations
        self.use_dask = use_dask

        if isinstance(data_source, DataFile):
            self.data_source = data_path(data_source)
        else:
            self.data_source = data_source

        assert self.data_source.name.endswith("csv")
        assert self.data_source.is_file()
        assert self.data_source.exists()

    def _read_data(self, read_kwargs: dict[str, Any]) -> pd.DataFrame:
        read_csv = dd.read_csv if self.use_dask else pd.read_csv
        df: Union[pd.DataFrame, dd.DataFrame]
        col_types = _get_crispr_screen_data_coltypes()
        if self.columns is None:
            df = read_csv(self.data_source, dtype=col_types, **read_kwargs)
        else:
            df = read_csv(
                self.data_source, usecols=self.columns, dtype=col_types, **read_kwargs
            )

        df = self._apply_transformations(df)

        if self.use_dask:
            df = df.compute()

        self._data = df
        return self._data

    def get_data(self, read_kwargs: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """Get CRISPR screen data.

        If the data has already been loaded, it is returned without re-reading from
        file.

        Args:
            read_kwargs (Optional[dict[str, Any]], optional): Key-word arguments for the
            CSV-parsing function. Defaults to None.

        Returns:
            pd.DataFrame: CRISPR screen data.
        """
        if self._data is not None:
            logger.info("Getting data - already loaded.")
            return self._data

        logger.info("Getting data - reading from file.")
        if read_kwargs is None:
            read_kwargs = {}
        return self._read_data(read_kwargs=read_kwargs)

    def set_data(self, data: pd.DataFrame, apply_transformations: bool = False) -> None:
        """Set the CRISPR screen data.

        Args:
            data (pd.DataFrame): New data.
            apply_transformations (bool, optional): Should the transformations be
              applied? Defaults to False.

        Returns:
            None
        """
        logger.info("Setting data.")
        if apply_transformations:
            data = self._apply_transformations(data)
        self._data = data
        return None

    def clear_data(self) -> None:
        """Clear the CRISPR screen data."""
        logger.info("Clearing data.")
        self._data = None

    def data_is_loaded(self) -> bool:
        """Check if the data has been loaded."""
        return self._data is not None

    def add_transformation(self, fxn: DataFrameTransformation) -> None:
        """Add a new transformation.

        The new transformation is added to the end of the current list.

        TODO (@jhrcook): Allow adding a list of funtions.

        Args:
            fxn (DataFrameTransformation): Data transforming function.

        Returns:
            None
        """
        logger.info("Adding new transformation.")
        self._transformations.append(fxn)
        return None

    def insert_transformation(self, fxn: DataFrameTransformation, at: int) -> None:
        """Insert a new transformation at a specified index.

        Args:
            fxn (DataFrameTransformation): Data transforming function.
            at (int): Insertion index.

        Returns:
            None
        """
        logger.info(f"Inserting transformation at index {at}.")
        self._transformations.insert(at, fxn)
        return None

    def get_transformations(self) -> list[DataFrameTransformation]:
        """Get (a copty of) the list of transformations.

        Returns:
            list[DataFrameTransformation]: Copy of the list of transformation.
        """
        return self._transformations.copy()

    def clear_transformations(self) -> None:
        """Clear the list of transformations."""
        logger.info("Clearing transformations.")
        self._transformations = []

    def set_transformations(
        self, new_transformations: list[DataFrameTransformation], apply: bool = False
    ) -> None:
        """Set the list of transformations.

        Args:
            new_transformations (list[DataFrameTransformation]): New list of data
              transforming functions.
            apply (bool, optional): Should the new list be applied to the data? Defaults
              to False.

        Returns:
            None
        """
        logger.info("Setting transformations.")
        self._transformations = new_transformations
        if apply:
            self.apply_transformations()
        return None

    def _apply_transformations(self, data: Data) -> Data:
        logger.info("Applying transofrmations to data.")
        for fxn in self._transformations:
            data = fxn(data)
        return data

    def apply_transformations(self) -> None:
        """Apply the transformations to the data."""
        logger.info("Applying transformations.")
        if self._data is None:
            raise DataNotLoadedException("Data not loaded")
        self._data = self._apply_transformations(self._data)

    def generate_mock_data(
        self, size: Union[MockDataSize, str], random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate mock data to be used for testing or SBC.

        Args:
            size (Union[MockDataSize, str]): Size of the final mock dataset.
            random_seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            pd.DataFrame: Mock data.
        """
        logger.info(f"Generating mock data of size '{size}'.")
        if random_seed is not None:
            np.random.seed(random_seed)

        if isinstance(size, str):
            size = MockDataSize(size)

        if size is MockDataSize.SMALL:
            self._data = mock_data.generate_mock_achilles_data(
                n_genes=10,
                n_sgrnas_per_gene=3,
                n_cell_lines=5,
                n_lineages=2,
                n_batches=2,
                n_screens=1,
            )
        elif size is MockDataSize.MEDIUM:
            self._data = mock_data.generate_mock_achilles_data(
                n_genes=25,
                n_sgrnas_per_gene=5,
                n_cell_lines=12,
                n_lineages=2,
                n_batches=3,
                n_screens=2,
            )
        elif size is MockDataSize.LARGE:
            self._data = mock_data.generate_mock_achilles_data(
                n_genes=100,
                n_sgrnas_per_gene=5,
                n_cell_lines=20,
                n_lineages=3,
                n_batches=4,
                n_screens=2,
            )
        else:
            assert_never(size)
        assert self._data is not None
        return self._data
