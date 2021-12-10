from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from speclet import io
from speclet.managers.data_managers import CrisprScreenDataManager as CrisprDM
from speclet.managers.data_managers import (
    DataFileDoesNotExist,
    DataFileIsNotAFile,
    UnsupportedDataFileType,
)


def test_fails_on_nonexistent_datafile(tmp_path: Path) -> None:
    fake_file = tmp_path / "does-not-exists-datafile.csv"
    with pytest.raises(DataFileDoesNotExist):
        CrisprDM(fake_file)  # as a Path object
    with pytest.raises(DataFileDoesNotExist):
        CrisprDM(str(fake_file))  # as a string


def test_fails_on_notafile_datafile(tmp_path: Path) -> None:
    fake_file = tmp_path / "does-not-exists-datafile.csv"
    fake_file.mkdir()
    with pytest.raises(DataFileIsNotAFile):
        CrisprDM(fake_file)  # as a Path object
    with pytest.raises(DataFileIsNotAFile):
        CrisprDM(str(fake_file))  # as a string


@pytest.mark.parametrize("suffix", [".cssv", ".txt", "csv", "_csv"])
def test_fails_on_noncsv_datafile(tmp_path: Path, suffix: str) -> None:
    fake_file = tmp_path / ("fake-datafile" + suffix)
    fake_file.touch()
    with pytest.raises(UnsupportedDataFileType):
        CrisprDM(fake_file)  # as a Path object
    with pytest.raises(UnsupportedDataFileType):
        CrisprDM(str(fake_file))  # as a string


def test_init_crispr_dm_with_data_file() -> None:
    _ = CrisprDM(io.DataFile.DEPMAP_TEST_DATA)
    return None


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".pkl"])
def test_init_crispr_dm_with_path(tmp_path: Path, suffix: str) -> None:
    f = tmp_path / ("fake-data" + suffix)
    f.touch()
    _ = CrisprDM(f)
    _ = CrisprDM(str(f))


def _head(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy().head(5)
    assert isinstance(_df, pd.DataFrame)
    return _df


def test_get_data() -> None:
    dm = CrisprDM(io.DataFile.DEPMAP_TEST_DATA)
    data = dm.get_data()
    assert data.shape[0] > 0
    assert data is dm.get_data()


def test_get_data_with_transforms() -> None:
    dm = CrisprDM(io.DataFile.DEPMAP_TEST_DATA, transformations=[_head])
    data = dm.get_data()
    assert data.shape[0] == 5
    assert data is dm.get_data()


def test_fail_validation_completely_different(iris: pd.DataFrame) -> None:
    dm = CrisprDM(io.DataFile.DEPMAP_TEST_DATA)
    with pytest.raises(SchemaError):
        dm.apply_validation(iris)
    with pytest.raises(SchemaError):
        dm.set_data(iris)


def test_fail_validation_close() -> None:
    dm = CrisprDM(io.DataFile.DEPMAP_TEST_DATA)
    data = dm.get_data()
    mod_data = data.copy().drop(columns=["sgrna"])
    with pytest.raises(SchemaError):
        dm.apply_validation(mod_data)
    with pytest.raises(SchemaError):
        dm.set_data(mod_data)
