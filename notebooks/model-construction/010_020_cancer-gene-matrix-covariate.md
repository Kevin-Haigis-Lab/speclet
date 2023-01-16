# Designing a covariate for the mutation of cancer genes

## Setup

```python
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path
from time import time
from typing import Union

import arviz as az
import janitor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc as pm
import pymc.math as pmmath
```

```python
from speclet.analysis.arviz_analysis import extract_matrix_variable_coords
from speclet.bayesian_models.hierarchical_nb import HierarchcalNegativeBinomialModel
from speclet.data_processing.common import get_cats
from speclet.data_processing.crispr import common_indices
from speclet.io import DataFile
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.plot.plotnine_helpers import set_gg_theme
from speclet.project_configuration import read_project_configuration
```

```python
# Notebook execution timer.
notebook_tic = time()

# Plotting setup.
set_gg_theme()
%config InlineBackend.figure_format = "retina"

# Constants
RANDOM_SEED = 816
np.random.seed(RANDOM_SEED)
HDI_PROB = read_project_configuration().modeling.highest_density_interval
```

## Load data

```python
counts_data = CrisprScreenDataManager(DataFile.DEPMAP_CRC_BONE_SUBSAMPLE).get_data()
counts_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>p_dna_batch</th>
      <th>genome_alignment</th>
      <th>hugo_symbol</th>
      <th>screen</th>
      <th>multiple_hits_on_gene</th>
      <th>sgrna_target_chr</th>
      <th>sgrna_target_pos</th>
      <th>...</th>
      <th>num_mutations</th>
      <th>any_deleterious</th>
      <th>any_tcga_hotspot</th>
      <th>any_cosmic_hotspot</th>
      <th>is_mutated</th>
      <th>copy_number</th>
      <th>lineage</th>
      <th>primary_or_metastasis</th>
      <th>is_male</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CGGAGCCTCGCCATTCCCGA</td>
      <td>COLO201-311Cas9_RepA_p6_batch3</td>
      <td>-0.183298</td>
      <td>3</td>
      <td>chr9_136410332_-</td>
      <td>ENTR1</td>
      <td>broad</td>
      <td>True</td>
      <td>9</td>
      <td>136410332</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.064776</td>
      <td>colorectal</td>
      <td>metastasis</td>
      <td>True</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAATAATTAAGTATGCACAT</td>
      <td>COLO201-311Cas9_RepA_p6_batch3</td>
      <td>-1.102995</td>
      <td>3</td>
      <td>chr13_48081696_-</td>
      <td>MED4</td>
      <td>broad</td>
      <td>True</td>
      <td>13</td>
      <td>48081696</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.582238</td>
      <td>colorectal</td>
      <td>metastasis</td>
      <td>True</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACAGCTGTTTACCAAGCGA</td>
      <td>COLO201-311Cas9_RepA_p6_batch3</td>
      <td>-0.991020</td>
      <td>3</td>
      <td>chr13_48083409_-</td>
      <td>MED4</td>
      <td>broad</td>
      <td>True</td>
      <td>13</td>
      <td>48083409</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.582238</td>
      <td>colorectal</td>
      <td>metastasis</td>
      <td>True</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AATCAACCCACAGCTGCACA</td>
      <td>COLO201-311Cas9_RepA_p6_batch3</td>
      <td>0.219207</td>
      <td>3</td>
      <td>chr17_7675183_+</td>
      <td>TP53</td>
      <td>broad</td>
      <td>True</td>
      <td>17</td>
      <td>7675183</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.978003</td>
      <td>colorectal</td>
      <td>metastasis</td>
      <td>True</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACAAGGGGCGACCGTCGCCA</td>
      <td>COLO201-311Cas9_RepA_p6_batch3</td>
      <td>0.003980</td>
      <td>3</td>
      <td>chr8_103415011_-</td>
      <td>DCAF13</td>
      <td>broad</td>
      <td>True</td>
      <td>8</td>
      <td>103415011</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.234548</td>
      <td>colorectal</td>
      <td>metastasis</td>
      <td>True</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>

## Create hierarchical NB model object

```python
hnb = HierarchcalNegativeBinomialModel()
```

```python
valid_counts_data = hnb.data_processing_pipeline(counts_data)
```

## Modify the data

A fake dictionary of cancer genes per lineage.

```python
cancer_genes = {
    "colorectal": {"KRAS", "APC", "PIK3CA"},
    "bone": {"TP53", "SCRG1"},
}
```

I modified the data to insert a synthetic lethal interaction in each lineage.

- for `colorectal`, there is a **reduction** in the final count of *DCAF13* when *KRAS* is mutated
- for `bone`, there is an **increase** in the final count of *MED4* when *SCRG1* is mutated

```python
def _modify_comutation(
    df: pd.DataFrame, target: str, cancer_gene: str, lineage: str, change: float
) -> pd.DataFrame:
    genes = df.hugo_symbol.values
    lineages = df.lineage.values
    cell_lines = df.depmap_id.values

    cancer_muts = (
        df.query(
            f"hugo_symbol == '{cancer_gene}' and lineage == '{lineage}'"
        )
        .query("is_mutated")
        .depmap_id.unique()
        .astype(str)
    )

    ary = (
        (lineages == lineage)
        * (genes == target)
        * [c in cancer_muts for c in cell_lines]
    )
    print(f"number of changes: {sum(ary)}")

    ct_final = df["counts_final"].values
    ct_final[ary] = ct_final[ary] * change
    df["counts_final"] = ct_final
    return df
```

```python
comutation_changes = [
    {"target": "ADH1B", "cancer_gene": "KRAS", "lineage": "colorectal", "change": 0.2},
    {"target": "MED4", "cancer_gene": "SCRG1", "lineage": "bone", "change": 2.0},
]
for change_info in comutation_changes:
    valid_counts_data = _modify_comutation(valid_counts_data, **change_info)  # type: ignore
```

    number of changes: 3
    number of changes: 4

## Covariate design and model construction

```python
def _collect_all_cancer_genes(cancer_genes: dict[str, set[str]]) -> list[str]:
    gene_set: set[str] = set()
    for genes in cancer_genes.values():
        gene_set = gene_set.union(genes)
    gene_list = list(gene_set)
    gene_list.sort()
    return gene_list


def _collect_mutations_per_cell_line(data: pd.DataFrame) -> dict[str, set[str]]:
    mut_data = (
        data[["depmap_id", "hugo_symbol", "is_mutated"]]
        .drop_duplicates()
        .query("is_mutated")
        .reset_index(drop=True)
    )
    mutations: dict[str, set[str]] = {}
    for cl in data.depmap_id.unique():
        mutations[cl] = set(mut_data.query(f"depmap_id == '{cl}'").hugo_symbol.unique())
    return mutations


def _make_cancer_gene_mutation_matrix(
    data: pd.DataFrame,
    cancer_genes: dict[str, set[str]],
    cell_lines: list[str],
    genes: list[str],
) -> np.ndarray:
    lineages = (
        data[["depmap_id", "lineage"]]
        .drop_duplicates()
        .sort_values("depmap_id")
        .lineage.values
    )
    assert len(cell_lines) == len(lineages)
    cell_mutations = _collect_mutations_per_cell_line(data)
    mut_mat = np.zeros(shape=(len(genes), len(cell_lines)), dtype=int)
    for j, (cl, lineage) in enumerate(zip(cell_lines, lineages)):
        cell_muts = cell_mutations[cl].intersection(cancer_genes[lineage])
        if len(cell_muts) == 0:
            continue
        mut_ary = np.array([g in cell_muts for g in genes])
        mut_mat[:, j] = mut_ary

    return mut_mat


def _augmented_mutation_data(
    data: pd.DataFrame, cancer_genes: dict[str, set[str]]
) -> np.ndarray:
    mut = data["is_mutated"].values.astype(int)
    for i, (gene, lineage) in enumerate(zip(data["hugo_symbol"], data["lineage"])):
        if gene in cancer_genes[lineage]:
            mut[i] = 0
    return mut


def make_model(data: pd.DataFrame, cancer_genes: dict[str, set[str]]) -> pm.Model:
    coords = {
        "gene": get_cats(data, "hugo_symbol"),
        "lineage": get_cats(data, "lineage"),
        "cell_line": get_cats(data, "depmap_id"),
        "cancer_gene": _collect_all_cancer_genes(cancer_genes),
    }

    ct_initial = data.counts_initial_adj.values.astype(float)
    ct_final = data.counts_final.values.astype(int)
    mut = _augmented_mutation_data(data, cancer_genes)

    M = _make_cancer_gene_mutation_matrix(
        data, cancer_genes, cell_lines=coords["cell_line"], genes=coords["cancer_gene"]
    )
    M = M[None, :, :]

    indices = common_indices(data)
    c = indices.cellline_idx
    g = indices.gene_idx
    ll = indices.lineage_idx
    c_to_ll = indices.cellline_to_lineage_idx

    with pm.Model(coords=coords, rng_seeder=RANDOM_SEED) as m:
        z = pm.Normal("z", 0, 5)

        sigma_b = pm.Gamma("sigma_b", 3, 1)
        delta_b = pm.Normal("delta_b", 0, 1, dims=("cell_line"))
        b = pm.Deterministic("b", delta_b * sigma_b, dims=("cell_line"))

        sigma_d = pm.Gamma("sigma_d", 3, 1)
        delta_d = pm.Normal("delta_d", 0, 1, dims=("gene", "lineage"))
        d = pm.Deterministic("d", delta_d * sigma_d, dims=("gene", "lineage"))

        sigma_m = pm.Gamma("sigma_m", 3, 1)
        delta_m = pm.Normal("delta_m", 0, 1, dims=("gene", "lineage"))
        m_var = pm.Deterministic("m", delta_m * sigma_m, dims=("gene", "lineage"))

        w = pm.Normal("w", 0, 1, dims=("gene", "cancer_gene", "lineage"))

        eta = pm.Deterministic(
            "eta",
            z
            + b[c]
            + d[g, ll]
            + (w[:, :, c_to_ll] * M).sum(axis=1)[g, c]
            + mut * m_var[g, ll],
        )
        mu = pm.Deterministic("mu", pmmath.exp(eta))

        alpha = pm.Gamma("alpha", 2.0, 0.5)
        y = pm.NegativeBinomial("ct_final", mu * ct_initial, alpha, observed=ct_final)

    return m
```

```python
pm.model_to_graphviz(make_model(valid_counts_data, cancer_genes=cancer_genes))
```

![svg](010_020_cancer-gene-matrix-covariate_files/010_020_cancer-gene-matrix-covariate_19_0.svg)

## Sample posterior

```python
with make_model(valid_counts_data, cancer_genes=cancer_genes):
    trace = pm.sample(draws=500, tune=200, chains=2, cores=2)
    _ = pm.sample_posterior_predictive(trace, extend_inferencedata=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/pymc/aesaraf.py:996: UserWarning: The parameter 'updates' of aesara.function() expects an OrderedDict, got <class 'dict'>. Using a standard dictionary here results in non-deterministic behavior. You should use an OrderedDict if you are using Python 2.7 (collections.OrderedDict for older python), or use a list of (shared, update) pairs. Do not just convert your dictionary to this type before the call as the conversion will still be non-deterministic.
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [z, sigma_b, delta_b, sigma_d, delta_d, sigma_m, delta_m, w, alpha]

<style>
    /*Turns off some styling*/
    progress {
        /*gets rid of default border in Firefox and Opera.*/
        border: none;
        /*Needs to be in here for Safari polyfill so background images work as expected.*/
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>

<div>
  <progress value='1400' class='' max='1400' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1400/1400 01:17<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 200 tune and 500 draw iterations (400 + 1_000 draws total) took 91 seconds.
    The acceptance probability does not match the target. It is 0.8929, but should be close to 0.8. Try to increase the number of tuning steps.
    We recommend running at least 4 chains for robust computation of convergence diagnostics

<style>
    /*Turns off some styling*/
    progress {
        /*gets rid of default border in Firefox and Opera.*/
        border: none;
        /*Needs to be in here for Safari polyfill so background images work as expected.*/
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>

<div>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:00<00:00]
</div>

## Posterior analysis

```python
var_names = ["z", "sigma_b", "b", "sigma_d", "d", "sigma_m", "m", "w", "alpha"]
az.plot_trace(trace, var_names=var_names)
plt.tight_layout();
```

![png](010_020_cancer-gene-matrix-covariate_files/010_020_cancer-gene-matrix-covariate_23_0.png)

```python
w_post = (
    az.summary(trace, var_names=["w"], kind="stats")
    .reset_index(drop=False)
    .rename(columns={"index": "param"})
    .assign(
        coords=lambda d: [
            x.replace("w[", "").replace("]", "").split(",") for x in d.param
        ]
    )
    .assign(
        hugo_symbol=lambda d: [x[0].strip() for x in d.coords],
        cancer_gene=lambda d: [x[1].strip() for x in d.coords],
        lineage=lambda d: [x[2].strip() for x in d.coords],
    )
)
```

The effect was captured by `w`.
Funny enough, I think the interaction between *MED4* and *KRAS* in colorectal is real; I only modified the values in bone.

```python
(
    gg.ggplot(
        w_post.query("hugo_symbol == 'MED4' or hugo_symbol == 'ADH1B'"),
        gg.aes(x="cancer_gene", y="lineage"),
    )
    + gg.facet_wrap("~hugo_symbol", ncol=1)
    + gg.geom_point(gg.aes(color="mean", size="1/sd"))
    + gg.scale_color_gradient2(low="#2c7bb6", mid="#ffffbf", high="#d7191c")
    + gg.theme(figure_size=(4, 4))
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

![png](010_020_cancer-gene-matrix-covariate_files/010_020_cancer-gene-matrix-covariate_26_1.png)

    <ggplot: (323938567)>

```python
(
    gg.ggplot(w_post, gg.aes(x="hugo_symbol", y="cancer_gene", fill="mean"))
    + gg.facet_wrap("~lineage", ncol=1)
    + gg.geom_tile(gg.aes(alpha="1/sd"), color=None)
    + gg.scale_x_discrete(expand=(0, 0.5))
    + gg.scale_y_discrete(expand=(0, 0.5))
    + gg.theme(figure_size=(10, 3), axis_text_x=gg.element_text(size=6, angle=90))
    + gg.labs(
        x="target gene",
        y="cancer-related gene",
        fill="$\hat{w}$",
        alpha="$\sigma(w)^{-1}$",
    )
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

![png](010_020_cancer-gene-matrix-covariate_files/010_020_cancer-gene-matrix-covariate_27_1.png)

    <ggplot: (324051508)>

Check the posteriors of $m$ and $w$ for the cancer genes when they are the target gene.
The mutation covariate $m$ has been augmented to skip the gene when it is a cancer gene.

```python
m_post = (
    az.summary(trace, var_names=["m"], kind="stats")
    .reset_index(drop=False)
    .rename(columns={"index": "param"})
    .pipe(
        extract_matrix_variable_coords,
        col="param",
        idx1name="hugo_symbol",
        idx2name="lineage",
    )
)
```

```python
plot_df = pd.concat(
    [
        w_post.query("hugo_symbol == cancer_gene").assign(var="w"),
        m_post.filter_column_isin(
            "hugo_symbol", _collect_all_cancer_genes(cancer_genes)
        ).assign(var="m"),
    ]
)

(
    gg.ggplot(plot_df, gg.aes(x="hugo_symbol", y="mean"))
    + gg.facet_wrap("~lineage", nrow=1)
    + gg.geom_point(gg.aes(color="var"), position=gg.position_dodge(width=0.5))
    + gg.scale_color_brewer(type="qual", palette="Dark2")
    + gg.theme(figure_size=(8, 4))
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/plotnine/utils.py:371: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

![png](010_020_cancer-gene-matrix-covariate_files/010_020_cancer-gene-matrix-covariate_30_1.png)

    <ggplot: (324535125)>

Just to make sure that the extreme positive distribution in `d` is not one of the genes being tested here.

```python
az.summary(trace, var_names="d", kind="stats").sort_values(
    "mean", ascending=False
).head(10)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d[TP53, bone]</th>
      <td>0.456</td>
      <td>0.128</td>
      <td>0.212</td>
      <td>0.695</td>
    </tr>
    <tr>
      <th>d[SAMD11, bone]</th>
      <td>0.269</td>
      <td>0.123</td>
      <td>0.048</td>
      <td>0.502</td>
    </tr>
    <tr>
      <th>d[COX7A1, bone]</th>
      <td>0.242</td>
      <td>0.124</td>
      <td>0.037</td>
      <td>0.489</td>
    </tr>
    <tr>
      <th>d[PWWP2B, bone]</th>
      <td>0.234</td>
      <td>0.127</td>
      <td>-0.010</td>
      <td>0.459</td>
    </tr>
    <tr>
      <th>d[CLYBL, bone]</th>
      <td>0.216</td>
      <td>0.128</td>
      <td>-0.028</td>
      <td>0.454</td>
    </tr>
    <tr>
      <th>d[KCNK17, bone]</th>
      <td>0.216</td>
      <td>0.142</td>
      <td>-0.065</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>d[FAM19A3, bone]</th>
      <td>0.211</td>
      <td>0.134</td>
      <td>-0.010</td>
      <td>0.483</td>
    </tr>
    <tr>
      <th>d[GPR157, bone]</th>
      <td>0.209</td>
      <td>0.131</td>
      <td>-0.021</td>
      <td>0.473</td>
    </tr>
    <tr>
      <th>d[FAM151B, bone]</th>
      <td>0.173</td>
      <td>0.132</td>
      <td>-0.063</td>
      <td>0.415</td>
    </tr>
    <tr>
      <th>d[ZFP82, bone]</th>
      <td>0.160</td>
      <td>0.141</td>
      <td>-0.067</td>
      <td>0.457</td>
    </tr>
  </tbody>
</table>
</div>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 3.30 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-03-28

    Python implementation: CPython
    Python version       : 3.9.9
    IPython version      : 8.1.1

    Compiler    : Clang 11.1.0
    OS          : Darwin
    Release     : 21.4.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: JHCookMac

    Git branch: oncogene-cov

    pandas    : 1.4.1
    plotnine  : 0.8.0
    arviz     : 0.12.0
    matplotlib: 3.5.1
    janitor   : 0.22.0
    pymc      : 4.0.0b5
    numpy     : 1.22.3
