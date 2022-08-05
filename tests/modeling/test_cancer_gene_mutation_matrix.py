from itertools import product

import janitor  # noqa: F401
import numpy as np
import pandas as pd
import pytest

from speclet.modeling.cancer_gene_mutation_matrix import (
    convert_to_dataframe,
    extract_mutation_matrix_and_cancer_genes,
    make_cancer_gene_mutation_matrix,
)


def random_fake_data(
    n_cells: int = 10, n_genes: int = 50, n_cgs: int = 8, seed: int | None = None
) -> tuple[pd.DataFrame, list[str]]:
    np.random.seed(seed)
    cell_lines = [f"cell_{i}" for i in range(n_cells)]
    genes = [f"gene_{i}" for i in range(n_genes)]
    cancer_genes = list(np.random.choice(genes, n_cgs, replace=False))
    cancer_genes.sort()

    data = pd.DataFrame(
        product(cell_lines, genes), columns=["depmap_id", "hugo_symbol"]
    ).assign(
        is_mutated=lambda d: np.random.choice(
            [True, False], len(d), replace=True, p=[0.4, 0.6]
        )
    )
    data = (
        pd.concat([data for _ in range(3)])
        .reset_index(drop=True)
        .assign(rna_expr=lambda d: np.random.normal(size=len(d)))
    )
    return data, cancer_genes


@pytest.mark.parametrize("seed", [10, 100, 1000])
@pytest.mark.parametrize(["n_genes", "n_cells", "n_cgs"], [(10, 100, 8), (12, 10, 4)])
def test_standard_make_cancer_gene_mutation_matrix(
    n_genes: int, n_cells: int, n_cgs: int, seed: int
) -> None:
    data, cancer_genes = random_fake_data(
        n_cells=n_cells, n_genes=n_genes, n_cgs=n_cgs, seed=seed
    )
    cg_mut_matrix = make_cancer_gene_mutation_matrix(
        data,
        cancer_genes,
        gene_col="hugo_symbol",
        cell_line_col="depmap_id",
        mut_col="is_mutated",
    )
    assert cg_mut_matrix is not None
    assert cg_mut_matrix.shape[0] == len(data)


def test_convert_to_dataframe() -> None:
    data, cancer_genes = random_fake_data()
    cg_mut_matrix = make_cancer_gene_mutation_matrix(data, cancer_genes)
    cg_mut_df = convert_to_dataframe(cg_mut_matrix)

    assert cg_mut_matrix is not None
    assert cg_mut_df is not None
    assert cg_mut_df.shape == cg_mut_matrix.shape
    cgs = cg_mut_matrix.coords["cancer_gene"].values
    assert np.all([a == b for a, b in zip(cg_mut_df.columns, cgs)])
    assert np.all(cg_mut_df.values == cg_mut_matrix.values)


def test_convert_to_dataframe_returns_none() -> None:
    assert convert_to_dataframe(None) is None


@pytest.mark.parametrize(
    ["gene_col", "cell_col", "mut_col"],
    [
        ("arbitrary-name", "other-col", "fhkvsjk"),
        ("87934789uhifefi", "93f%CC", "?<><>?"),
        ('""', "___", "?"),
    ],
)
def test_cancer_gene_mutation_matrix_different_column_names(
    gene_col: str, cell_col: str, mut_col: str
) -> None:
    data, cancer_genes = random_fake_data(seed=10)

    original_mat = make_cancer_gene_mutation_matrix(
        data,
        cancer_genes,
        gene_col="hugo_symbol",
        cell_line_col="depmap_id",
        mut_col="is_mutated",
    )
    assert original_mat is not None
    assert original_mat.shape[0] == len(data)

    data = data.rename(
        columns={"hugo_symbol": gene_col, "depmap_id": cell_col, "is_mutated": mut_col}
    )
    new_mat = make_cancer_gene_mutation_matrix(
        data,
        cancer_genes,
        gene_col=gene_col,
        cell_line_col=cell_col,
        mut_col=mut_col,
    )
    assert new_mat is not None
    assert new_mat.shape[0] == len(data)
    assert np.all(original_mat == new_mat)
    assert np.all(original_mat.values == new_mat.values)
    assert np.all(original_mat.coords["cancer_gene"] == new_mat.coords["cancer_gene"])


def test_merge_comutated_cancer_genes() -> None:
    data, cancer_genes = random_fake_data(seed=2)

    # Set two cancer genes as perfectly comutated.
    cg1, cg2 = cancer_genes[0], cancer_genes[1]
    muts = set(data.query(f"hugo_symbol == '{cg1}' and is_mutated")["depmap_id"])
    idx = (data["hugo_symbol"] == cg2).values
    data.loc[idx, "is_mutated"] = 0
    idx = idx * data["depmap_id"].isin(muts).values
    data.loc[idx, "is_mutated"] = 1

    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes)
    assert cg_mut_mat is not None
    cgs = cg_mut_mat.coords["cancer_gene"].values.tolist()
    assert f"{cg1}|{cg2}" in cgs
    assert cg1 not in cgs
    assert cg2 not in cgs


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_no_colinearity_in_cancer_gene_mutation_matrix(seed: int) -> None:
    data, cancer_genes = random_fake_data(n_genes=10, n_cells=5, n_cgs=5, seed=seed)
    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes)
    if cg_mut_mat is None:
        return
    n_cols = cg_mut_mat.shape[1]
    for i, j in product(range(n_cols), range(n_cols)):
        if i == j:
            continue
        assert not (cg_mut_mat[:, i] == cg_mut_mat[:, j]).all()


def _num_mutations_in_genes(
    df: pd.DataFrame,
    genes: list[str],
    cell_col: str = "depmap_id",
    gene_col: str = "hugo_symbol",
    mut_col: str = "is_mutated",
) -> pd.Series:
    return (
        df.filter_column_isin(gene_col, genes)[[cell_col, gene_col, mut_col]]
        .drop_duplicates()
        .query(mut_col)
        .groupby(gene_col)[cell_col]
        .count()
    )


@pytest.mark.parametrize("top_n", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", [23, 44])
def test_top_n_cancer_genes(top_n: int, seed: int) -> None:
    data, cancer_genes = random_fake_data(n_genes=200, n_cells=10, n_cgs=20, seed=seed)
    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes, top_n_cg=top_n)
    assert cg_mut_mat is not None
    assert 0 <= len(cg_mut_mat.coords["cancer_gene"])
    cgs = cg_mut_mat.coords["cancer_gene"].values.tolist()
    n_muts_cgs = _num_mutations_in_genes(data, cgs)
    assert 0 < n_muts_cgs.nunique() <= top_n


@pytest.mark.parametrize("min_n", [0, 1, 5, 10])
@pytest.mark.parametrize(["n_genes", "n_cells", "n_cgs"], [(10, 2, 3), (5, 20, 2)])
def test_min_n_mutations(n_genes: int, n_cells: int, n_cgs: int, min_n: int) -> None:
    data, cancer_genes = random_fake_data(
        n_genes=n_genes, n_cells=n_cells, n_cgs=n_cgs, seed=3
    )
    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes, min_n=min_n)
    if cg_mut_mat is None:
        n_muts_cgs = _num_mutations_in_genes(data, cancer_genes)
        mut_in_all_or_none = (n_muts_cgs == n_cells) + (n_muts_cgs == 0)
        assert np.all((n_muts_cgs < min_n) + mut_in_all_or_none)
    else:
        cgs = cg_mut_mat.coords["cancer_gene"].values.tolist()
        n_muts_cgs = _num_mutations_in_genes(data, cgs)
        assert np.all(n_muts_cgs >= min_n)


@pytest.mark.parametrize("min_freq", [0, 0.1, 0.5, 0.8, 1.0, 1.2])
@pytest.mark.parametrize("n_cells", [2, 3, 20, 100])
def test_min_freq_mutations(min_freq: float, n_cells: int) -> None:
    data, cancer_genes = random_fake_data(
        n_genes=100, n_cells=n_cells, n_cgs=10, seed=3
    )
    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes, min_freq=min_freq)
    if cg_mut_mat is None:
        mut_freq_cgs = _num_mutations_in_genes(data, cancer_genes) / n_cells
        mut_in_all_or_none = (mut_freq_cgs == 1) + (mut_freq_cgs == 0)
        assert np.all((mut_freq_cgs < min_freq) + mut_in_all_or_none)
    else:
        cgs = cg_mut_mat.coords["cancer_gene"].values.tolist()
        mut_freq_cgs = _num_mutations_in_genes(data, cgs) / n_cells
        assert np.all(mut_freq_cgs >= min_freq)


def test_no_cancer_genes_provided() -> None:
    assert make_cancer_gene_mutation_matrix(pd.DataFrame(), []) is None


def test_does_not_drop_cancer_if_only_one() -> None:
    data, cancer_genes = random_fake_data(n_cgs=1)
    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes)
    assert cg_mut_mat is not None
    assert set(cg_mut_mat.coords["cancer_gene"].values) == set(cancer_genes)
    assert len(cg_mut_mat.coords["cancer_gene"].values) == 1


def test_extract_mutation_matrix_and_cancer_genes() -> None:
    data, cancer_genes = random_fake_data()
    cg_mut_mat = make_cancer_gene_mutation_matrix(data, cancer_genes)
    assert cg_mut_mat is not None
    mut_mat, cgs = extract_mutation_matrix_and_cancer_genes(cg_mut_mat)
    assert np.all(mut_mat == cg_mut_mat)
    assert cg_mut_mat.shape == mut_mat.shape
    assert len(cgs) == cg_mut_mat.shape[1] == mut_mat.shape[1]
    assert set(cgs) == set(cg_mut_mat.coords["cancer_gene"].values)
    assert np.all(np.array(cgs) == cg_mut_mat.coords["cancer_gene"].values)
