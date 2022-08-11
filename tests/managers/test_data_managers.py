from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError

from speclet import io
from speclet.managers.data_managers import CancerGeneDataManager as CancerGeneDM
from speclet.managers.data_managers import CrisprScreenDataManager as CrisprDM
from speclet.managers.data_managers import (
    DataFileDoesNotExist,
    DataFileIsNotAFile,
    LineageSubtypeGeneMap,
    UnsupportedDataFileType,
)

# --- CrisprScreenDataManager ---


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
    iris["sgrna_target_chr"] = ["1"] * len(iris)
    iris["depmap_id"] = ["a"] * len(iris)
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


def test_fail_validation_nonunique_mapping_of_sgrna_to_gene() -> None:
    dm = CrisprDM(io.DataFile.DEPMAP_TEST_DATA)
    data = dm.get_data()
    # Just shuffle gene assignments.
    mod_data = data.copy()
    mod_data["hugo_symbol"] = mod_data.hugo_symbol.sample()
    with pytest.raises(SchemaError):
        dm.apply_validation(mod_data)

    # Change a single gene assignment.
    mod_data = data.copy()
    genes = mod_data.hugo_symbol.to_list()
    swap_gene = set(genes).difference(genes[0])
    mod_data["hugo_symbol"] = [swap_gene] + genes[1:]
    with pytest.raises(SchemaError):
        dm.apply_validation(mod_data)


def test_fail_validation_nonunique_mapping_of_celllines_to_lineage() -> None:
    dm = CrisprDM(io.DataFile.DEPMAP_TEST_DATA)
    data = dm.get_data()
    # Just shuffle gene assignments.
    mod_data = data.copy()
    mod_data["lineage"] = mod_data.lineage.sample()
    with pytest.raises(SchemaError):
        dm.apply_validation(mod_data)

    # Change a single gene assignment.
    mod_data = data.copy()
    lineages = mod_data.lineage.to_list()
    swap_lineage = set(lineages).difference(lineages[0])
    mod_data["lineage"] = [swap_lineage] + lineages[1:]
    with pytest.raises(SchemaError):
        dm.apply_validation(mod_data)


# --- CancerGeneDataManager ---


def test_read_bailey_2018_gene_map() -> None:
    dm = CancerGeneDM()
    gene_map = dm.bailey_2018_cancer_genes()
    assert "bile_duct" in gene_map
    assert "colorectal" in gene_map


def test_read_cosmic_gene_map() -> None:
    dm = CancerGeneDM()
    gene_map = dm.cosmic_cancer_genes()
    assert "bile_duct" in gene_map
    assert "colorectal" in gene_map


@pytest.mark.parametrize(
    "gene_map",
    [CancerGeneDM().bailey_2018_cancer_genes(), CancerGeneDM().cosmic_cancer_genes()],
)
def test_reduce_gene_map(gene_map: LineageSubtypeGeneMap) -> None:
    dm = CancerGeneDM()
    reduced_gene_map = dm.reduce_to_lineage(gene_map)
    assert len(reduced_gene_map) == len(gene_map)
    assert set(reduced_gene_map.keys()) == set(gene_map.keys())
    assert isinstance(reduced_gene_map["colorectal"], set)
