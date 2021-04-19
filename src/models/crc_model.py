#!/usr/bin/env python3

"""Builders for CRC PyMC3 models."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_processing import achilles as achelp
from src.io import data_io
from src.models.speclet_model import SpecletModel


class CrcModel(SpecletModel):
    """Base model for CRC modeling.

    Args:
        SpecletModel ([type]): Subclassed from a SpecletModel.
    """

    debug: bool
    data: Optional[pd.DataFrame] = None

    def __init__(
        self, name: str, root_cache_dir: Optional[Path] = None, debug: bool = False
    ):
        """Create a CrcModel object.

        Args:
            name (str): A unique identifier for this instance of CrcModel.
              (Used for cache management.)
            root_cache_dir (Optional[Path], optional): The directory for caching
              sampling/fitting results. Defaults to None to use the speclet default.
            debug (bool, optional): Are you in debug mode? Defaults to False.
        """
        super().__init__(name="crc_" + name, root_cache_dir=root_cache_dir)
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
