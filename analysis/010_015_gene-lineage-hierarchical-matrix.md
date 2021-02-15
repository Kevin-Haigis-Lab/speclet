```python
import re
import string
import warnings
from pathlib import Path
from time import time

import arviz as az
import common_data_processing as dphelp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import pymc3_helpers as pmhelp
import seaborn as sns
from theano import tensor as tt

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_minimal())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 245
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("pymc3_model_cache")
```

# Modeling with a hierarchical 2D covariate with additional varying levels

This notebook is to experiment with creating a hierarchical covariate with 2 dimensions: gene $\times$ cell line.
Each value in the matrix will come from a model of two varying covariables, one for the gene and one for the cell line.
The cell line covariate will then come from a distribution of the lineage as another level in the model.

## Data preparation

I will need a fair amount of data to fit a model with so many parameters, but I will still start with a subsample of the full data set.

```python
NUM_GENES = 20
NUM_LINEAGES = 5

data_path = Path("../modeling_data")
starting_dataset_path = data_path / "depmap_modeling_dataframe.csv"
final_dataset_path = (
    data_path / "subsample_covariate-matrix_depmap-modeling-dataframe.csv"
)

if final_dataset_path.exists():
    # Read in data set that has already been created.
    print("Reading in existing file.")
    modeling_data = pd.read_csv(final_dataset_path)
else:
    # Read in full data set and subsample some genes and lineages.
    print(
        f"Subsampling {NUM_GENES} genes and {NUM_LINEAGES} lineages from full dataset."
    )
    full_modeling_data = pd.read_csv(starting_dataset_path, low_memory=False)

    GENES = np.random.choice(full_modeling_data.hugo_symbol.unique(), NUM_GENES)
    LINEAGES = np.random.choice(full_modeling_data.lineage.unique(), NUM_LINEAGES)

    modeling_data = full_modeling_data[full_modeling_data.hugo_symbol.isin(GENES)]
    modeling_data = modeling_data[modeling_data.lineage.isin(LINEAGES)]
    modeling_data = modeling_data.reset_index(drop=True)

    modeling_data.to_csv(final_dataset_path)
    del full_modeling_data
```

    Reading in existing file.

```python
print("Genes:")
print(modeling_data.hugo_symbol.unique().tolist())

print("-" * 60)
print("Lineages:")
print(modeling_data.lineage.unique().tolist())

print("-" * 60)
print(f"number of cell lines: {modeling_data.depmap_id.nunique()}")
```

    Genes:
    ['C18orf54', 'ARG1', 'DHDH', 'RGPD6', 'H3F3A', 'STRC', 'GIPR', 'FAM206A', 'YPEL5', 'GTF2E1', 'AAED1', 'LGI3', 'ZNF175', 'ARHGAP26', 'IL17B', 'HERC1', 'PLEKHH3', 'NACC2', 'PRKAR1A', 'CT47A7']
    ------------------------------------------------------------
    Lineages:
    ['blood', 'upper_aerodigestive', 'colorectal', 'liver']
    ------------------------------------------------------------
    number of cell lines: 132

```python
modeling_data = modeling_data.sort_values(
    ["hugo_symbol", "sgrna", "lineage", "depmap_id"]
).reset_index(drop=True)

for col in [
    "sgrna",
    "pdna_batch",
    "depmap_id",
    "lineage",
    "kras_mutation",
    "hugo_symbol",
]:
    modeling_data = dphelp.make_cat(modeling_data, col, ordered=True, sort_cats=True)
```

```python
modeling_data.shape
```

    (22960, 28)

```python
modeling_data.head(5)
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
      <th>Unnamed: 0</th>
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>pdna_batch</th>
      <th>passes_qc</th>
      <th>depmap_id</th>
      <th>primary_or_metastasis</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>...</th>
      <th>log2_gene_cn_p1</th>
      <th>gene_cn</th>
      <th>n_muts</th>
      <th>any_deleterious</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6249</td>
      <td>CACCCGCACGAACACCACCA</td>
      <td>hel-311cas9_repa_p4_batch3</td>
      <td>0.619938</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000004</td>
      <td>NaN</td>
      <td>blood</td>
      <td>AML</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6331</td>
      <td>CACCCGCACGAACACCACCA</td>
      <td>hel-311cas9_repb_p4_batch3</td>
      <td>0.780068</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000004</td>
      <td>NaN</td>
      <td>blood</td>
      <td>AML</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6413</td>
      <td>CACCCGCACGAACACCACCA</td>
      <td>hel9217-311cas9_repa_p6_batch3</td>
      <td>0.176848</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000005</td>
      <td>NaN</td>
      <td>blood</td>
      <td>AML</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6495</td>
      <td>CACCCGCACGAACACCACCA</td>
      <td>hel9217-311cas9_repb_p6_batch3</td>
      <td>0.295670</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000005</td>
      <td>NaN</td>
      <td>blood</td>
      <td>AML</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12809</td>
      <td>CACCCGCACGAACACCACCA</td>
      <td>mv4;11-311cas9_repa_p6_batch2</td>
      <td>0.564909</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000045</td>
      <td>Primary</td>
      <td>blood</td>
      <td>AML</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>

## Data visualization

```python
plot_data = (
    modeling_data[["lineage", "depmap_id"]]
    .drop_duplicates()
    .groupby("lineage")
    .count()
    .reset_index()
)

(
    gg.ggplot(plot_data, gg.aes("lineage", "depmap_id"))
    + gg.geom_col(gg.aes(fill="depmap_id"))
    + gg.geom_text(gg.aes(label="depmap_id"), va="bottom")
    + gg.scale_fill_gradient(guide=None)
    + gg.scale_y_continuous(expand=(0, 0, 0.05, 0))
    + gg.labs(x="lineage", y="number of cell lines")
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_9_0.png)

    <ggplot: (8754894691816)>

```python
plot_data = (
    modeling_data[["lineage", "hugo_symbol", "lfc"]]
    .groupby(["lineage", "hugo_symbol"])
    .mean()
    .reset_index()
)

(
    gg.ggplot(plot_data, gg.aes(x="lineage", y="hugo_symbol", fill="lfc"))
    + gg.geom_tile()
    + gg.scale_x_discrete(expand=(0.1, 0.1))
    + gg.labs(x="lineage", y="gene", fill="avg. LFC")
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_10_0.png)

    <ggplot: (8754885703554)>

```python
(
    gg.ggplot(modeling_data, gg.aes(x="hugo_symbol", y="lfc"))
    + gg.geom_boxplot(
        gg.aes(color="lineage", fill="lineage"),
        alpha=0.4,
        outlier_alpha=0.4,
        outlier_size=0.4,
    )
    + gg.theme(
        figure_size=(10, 5),
        axis_text_x=gg.element_text(angle=90),
        legend_position=(0.22, 0.25),
    )
    + gg.labs(x="gene", y="LFC", fill="lineage")
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_11_0.png)

    <ggplot: (8754885644141)>

## Modeling

```python
gene_idx, num_genes = dphelp.get_indices_and_count(modeling_data, "hugo_symbol")

cellline_idx, num_celllines = dphelp.get_indices_and_count(modeling_data, "depmap_id")

cellline_to_lineage_map = (
    modeling_data[["lineage", "depmap_id"]].drop_duplicates().reset_index(drop=True)
)
cellline_to_lineage_idx, num_lineages = dphelp.get_indices_and_count(
    cellline_to_lineage_map, "lineage"
)
```

```python
with pm.Model() as m1:

    # Indexing arrays
    gene_idx_shared = pm.Data("gene_idx", gene_idx)
    cellline_idx_shared = pm.Data("cellline_idx", cellline_idx)
    cellline_to_lineage_idx_shared = pm.Data(
        "cellline_to_lineage_idx", cellline_to_lineage_idx
    )

    # Data
    lfc_shared = pm.Data("lfc", modeling_data.lfc.to_numpy())

    # Model parameters
    μ_μ_μ_γ = pm.Normal("μ_μ_μ_γ", 3)
    σ_μ_μ_γ = pm.HalfNormal("σ_μ_μ_γ", 3)

    μ_μ_γ = pm.Normal("μ_μ_γ", μ_μ_μ_γ, σ_μ_μ_γ, shape=(num_genes, 1))
    σ_σ_μ_γ = pm.HalfNormal("σ_σ_μ_γ", 3)
    σ_μ_γ = pm.HalfNormal("σ_μ_γ", σ_σ_μ_γ, shape=(num_genes, 1))

    μ_γ = pm.Normal("μ_γ", μ_μ_γ, σ_μ_γ, shape=(num_genes, num_lineages))
    σ_σ_γ = pm.HalfNormal("σ_σ_γ", 3)
    σ_γ = pm.HalfNormal("σ_γ", σ_σ_γ, shape=num_lineages)

    γ = pm.Normal(
        "γ",
        μ_γ[:, cellline_to_lineage_idx_shared],
        σ_γ[cellline_to_lineage_idx_shared],
        shape=(num_genes, num_celllines),
    )

    μ = pm.Deterministic("μ", γ[gene_idx_shared, cellline_idx_shared])
    σ = pm.HalfNormal("σ", 3)

    # Likelihood
    y = pm.Normal("y", μ, σ, observed=lfc_shared)
```

```python
pm.model_to_graphviz(m1)
```

![svg](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_15_0.svg)

TODO:

- do prior predictive checks to modify prior distributions to help with sampling
- there are some divergences in the sampling, but it is not extreme

```python
m1_cache_dir = pymc3_cache_dir / "gene-lineage-model_m1"

m1_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m1,
    num_mcmc=3000,
    tune=4000,
    chains=4,
    cores=4,
    random_seed=RANDOM_SEED,
    cache_dir=m1_cache_dir,
    force=False,
    sample_kwargs={"init": "advi", "n_init": 200000, "target_accept": 0.9},
)

m1_az = pmhelp.samples_to_arviz(model=m1, res=m1_sampling_results)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using advi...

<div>
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
  <progress value='285631' class='' max='500000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  57.13% [285631/500000 13:53<10:25 Average Loss = 17,363]
</div>

    Interrupted at 285,676 [57%]: Average Loss = 19,225
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, γ, σ_γ, σ_σ_γ, μ_γ, σ_μ_γ, σ_σ_μ_γ, μ_μ_γ, σ_μ_μ_γ, μ_μ_μ_γ]

<div>
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
  <progress value='15933' class='' max='28000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  56.90% [15933/28000 23:26<17:45 Sampling 4 chains, 52 divergences]
</div>

    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-28-7308b85c789e> in <module>
          1 m1_cache_dir = pymc3_cache_dir / "gene-lineage-model_m1"
          2
    ----> 3 m1_sampling_results = pmhelp.pymc3_sampling_procedure(
          4     model=m1,
          5     num_mcmc=3000,


    /n/data2/dfci/cancerbio/haigis/Cook/speclet/analysis/pymc3_helpers.py in pymc3_sampling_procedure(model, num_mcmc, tune, chains, cores, prior_check_samples, ppc_samples, random_seed, cache_dir, force, sample_kwargs)
         99                 prior_check_samples, random_seed=random_seed
        100             )
    --> 101             trace = pm.sample(
        102                 draws=num_mcmc,
        103                 tune=tune,


    ~/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/sampling.py in sample(draws, step, init, n_init, start, trace, chain_idx, chains, cores, tune, progressbar, model, random_seed, discard_tuned_samples, compute_convergence_checks, callback, return_inferencedata, idata_kwargs, mp_ctx, pickle_backend, **kwargs)
        590     # ideally via the "tune" statistic, but not all samplers record it!
        591     if "tune" in trace.stat_names:
    --> 592         stat = trace.get_sampler_stats("tune", chains=0)
        593         # when CompoundStep is used, the stat is 2 dimensional!
        594         if len(stat.shape) == 2:


    ~/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/backends/base.py in get_sampler_stats(self, stat_name, burn, thin, combine, chains, squeeze)
        520             chains = [chains]
        521
    --> 522         results = [self._straces[chain].get_sampler_stats(stat_name, None, burn, thin)
        523                    for chain in chains]
        524         return _squeeze_cat(results, combine, squeeze)


    ~/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/backends/base.py in <listcomp>(.0)
        520             chains = [chains]
        521
    --> 522         results = [self._straces[chain].get_sampler_stats(stat_name, None, burn, thin)
        523                    for chain in chains]
        524         return _squeeze_cat(results, combine, squeeze)


    KeyError: 0

```python
az.summary(m1_az, var_names=["σ_μ_μ_γ", "μ_μ_μ_γ"], hdi_prob=0.89)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>σ_μ_μ_γ</th>
      <td>0.540</td>
      <td>0.097</td>
      <td>0.394</td>
      <td>0.681</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>10071.0</td>
      <td>8590.0</td>
      <td>12290.0</td>
      <td>6218.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>μ_μ_μ_γ</th>
      <td>-0.135</td>
      <td>0.125</td>
      <td>-0.328</td>
      <td>0.067</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>8478.0</td>
      <td>7397.0</td>
      <td>8896.0</td>
      <td>5171.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
az.plot_trace(m1_az, var_names=["μ_μ_μ_γ", "σ_μ_μ_γ"], combined=True)
plt.show()
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_19_0.png)

```python
gamma_post_summary = az.summary(m1_az, var_names="γ", hdi_prob=0.89, kind="stats")
```

```python
gamma_post_summary.head()
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>γ[0,0]</th>
      <td>0.191</td>
      <td>0.144</td>
      <td>-0.038</td>
      <td>0.422</td>
    </tr>
    <tr>
      <th>γ[0,1]</th>
      <td>0.072</td>
      <td>0.145</td>
      <td>-0.156</td>
      <td>0.310</td>
    </tr>
    <tr>
      <th>γ[0,2]</th>
      <td>-0.067</td>
      <td>0.142</td>
      <td>-0.298</td>
      <td>0.156</td>
    </tr>
    <tr>
      <th>γ[0,3]</th>
      <td>-0.252</td>
      <td>0.122</td>
      <td>-0.436</td>
      <td>-0.045</td>
    </tr>
    <tr>
      <th>γ[0,4]</th>
      <td>0.128</td>
      <td>0.146</td>
      <td>-0.103</td>
      <td>0.358</td>
    </tr>
  </tbody>
</table>
</div>

```python
def extract_matrix_variable_indices(
    d: pd.DataFrame,
    col: str,
    idx1: np.ndarray,
    idx2: np.ndarray,
    idx1name: str,
    idx2name: str,
) -> pd.DataFrame:
    indices_list = [
        [int(x) for x in re.findall("[0-9]+", s)] for s in d[[col]].to_numpy().flatten()
    ]
    indices_array = np.asarray(indices_list)
    d[idx1name] = idx1[indices_array[:, 0]]
    d[idx2name] = idx2[indices_array[:, 1]]
    return d


gamma_post_summary = (
    gamma_post_summary.reset_index()
    .rename(columns={"index": "variable"})
    .pipe(
        extract_matrix_variable_indices,
        col="variable",
        idx1=modeling_data.hugo_symbol.unique(),
        idx2=modeling_data.depmap_id.unique(),
        idx1name="hugo_symbol",
        idx2name="depmap_id",
    )
)
```

```python
gamma_post_summary = gamma_post_summary.merge(
    cellline_to_lineage_map, on="depmap_id", left_index=False, right_index=False
)
```

```python
gamma_post_summary.head()
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
      <th>variable</th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>hugo_symbol</th>
      <th>depmap_id</th>
      <th>lineage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>γ[0,0]</td>
      <td>0.191</td>
      <td>0.144</td>
      <td>-0.038</td>
      <td>0.422</td>
      <td>AAED1</td>
      <td>ACH-000004</td>
      <td>blood</td>
    </tr>
    <tr>
      <th>1</th>
      <td>γ[1,0]</td>
      <td>0.247</td>
      <td>0.143</td>
      <td>0.020</td>
      <td>0.477</td>
      <td>ARG1</td>
      <td>ACH-000004</td>
      <td>blood</td>
    </tr>
    <tr>
      <th>2</th>
      <td>γ[2,0]</td>
      <td>0.121</td>
      <td>0.142</td>
      <td>-0.110</td>
      <td>0.340</td>
      <td>ARHGAP26</td>
      <td>ACH-000004</td>
      <td>blood</td>
    </tr>
    <tr>
      <th>3</th>
      <td>γ[3,0]</td>
      <td>0.084</td>
      <td>0.147</td>
      <td>-0.165</td>
      <td>0.306</td>
      <td>C18orf54</td>
      <td>ACH-000004</td>
      <td>blood</td>
    </tr>
    <tr>
      <th>4</th>
      <td>γ[4,0]</td>
      <td>0.001</td>
      <td>0.133</td>
      <td>-0.210</td>
      <td>0.213</td>
      <td>CT47A7</td>
      <td>ACH-000004</td>
      <td>blood</td>
    </tr>
  </tbody>
</table>
</div>

```python
(
    gg.ggplot(gamma_post_summary, gg.aes(x="depmap_id", y="hugo_symbol"))
    + gg.geom_tile(gg.aes(fill="mean"))
    + gg.theme(axis_text_x=gg.element_blank())
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_25_0.png)

    <ggplot: (8770121041577)>

```python
(
    gg.ggplot(gamma_post_summary, gg.aes(x="hugo_symbol", y="mean"))
    + gg.geom_boxplot(
        gg.aes(color="lineage", fill="lineage"),
        alpha=0.4,
        outlier_alpha=0.4,
        outlier_size=0.4,
    )
    + gg.theme(
        figure_size=(10, 5),
        axis_text_x=gg.element_text(angle=90),
        legend_position=(0.22, 0.25),
    )
    + gg.labs(
        x="gene",
        y="posterior mean",
        fill="lineage",
        title="Posterior distributions for γ",
    )
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_26_0.png)

    <ggplot: (8770120674392)>

```python

```

```python

```

```python

```

## Modeling

This is a model with the following features:

- the primary level is $\mu = \alpha_s + \beta_c + \gamma_{gc}$
- $\alpha_s$ is a varying intercept by sgRNA which comes from a distribution for the target gene: $\mathcal{N}(\mu_\alpha, \sigma_\alpha)$
- $\beta_c$ is a varying intercept by cell line which comes from a distribution for the lineage: $\mathcal{N}(\mu_\beta, \sigma_\beta)$
- $\gamma_{gc}$ is a  2-dimensional varying intercept - one for the gene and one for the cell line
    - the mean of the distribution for $\gamma_{gc}$ is $\mu_\gamma = \eta_g + \theta_c$ so it is linked the gene and cell line
    - the standard deviation for this distribution $\sigma_\gamma$ varying by gene as the lineage specificity can vary to differing degrees per gene
    - $\eta_g$ comes from a distribution of genes and $\theta_c$ comes from a distribution of cell lines which has an additional level for the lineages of the cell lines

---

$
lfc \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_s + \beta_c + \gamma_{gc} \\
\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha}, \sigma_{\alpha}) \\
\qquad \mu_{\alpha} \sim \mathcal{N}(\mu_{\mu_{\alpha}}, \sigma_{\mu_{\alpha}}) \\
\qquad \quad \mu_{\mu_{\alpha}} \sim \mathcal{N}(0, 3) \\
\qquad \quad \sigma_{\mu_{\alpha}} \sim \text{HalfNormal}(3) \\
\qquad \sigma_{\alpha} \sim \text{HalfNormal}(\sigma_{\sigma_{\alpha}}) \\
\qquad \quad \sigma_{\sigma_{\alpha}} \sim \text{HalfNormal}(3) \\
\
\quad \beta_c \sim \mathcal{N}(\mu_{\beta}, \sigma_{\beta}) \\
\qquad \mu_{\beta} \sim \mathcal{N}(\mu_{\mu_{\beta}}, \sigma_{\mu_{\beta}}) \\
\qquad \quad \mu_{\mu_{\beta}} \sim \mathcal{N}(0, 3) \\
\qquad \quad \sigma_{\mu_{\beta}} \sim \text{HalfNormal}(3) \\
\qquad \sigma_{\beta} \sim \text{HalfNormal}(\sigma_{\sigma_{\beta}}) \\
\qquad \quad \sigma_{\sigma_{\beta}} \sim \text{HalfNormal}(3) \\
\
\quad \gamma_{gc} \sim \mathcal{N}(\mu_{\gamma}, \sigma_{\gamma}) \\
\qquad \mu_{\gamma} = \eta_g + \theta_c \\
\qquad \quad \eta_g \sim \mathcal{N}(\mu_{\eta}, \sigma_{\eta}) \\
\qquad \qquad \mu_{\eta} \sim \mathcal{N}(0, 3) \\
\qquad \qquad \sigma_{\eta} \sim \text{HalfNormal}(0, 3) \\
\
\qquad \quad \theta_c \sim \mathcal{N}(\mu_{\theta}, \sigma_{\theta}) \\
\qquad \qquad \mu_{\theta} \sim \mathcal{N}(\mu_{\mu_\theta}, \sigma_{\mu_\theta}) \\
\qquad \qquad \quad \mu_{\mu_\theta} \sim \mathcal{N}(0, 3) \\
\qquad \qquad \quad \sigma_{\mu_\theta} \sim \text{HalfNormal}(3) \\
\qquad \qquad \sigma_{\theta} \sim \text{HalfNormal}(\sigma_{\sigma_\theta}) \\
\qquad \qquad \quad \sigma_{\sigma_\theta} \sim \text{HalfNormal}(3) \\
\
\qquad \sigma_{\gamma} \sim \text{HalfNormal}(\sigma_{\sigma_{\gamma}}) \\
\qquad \quad \sigma_{\sigma_{\gamma}} \sim \text{HalfNormal}(1) \\
\
\sigma \sim \text{HalfNormal}(3)
$

```python
# Index sgRNAs and genes for full data.
sgrna_idx, num_sgrnas = dphelp.get_indices_and_count(modeling_data, "sgrna")
gene_idx, num_genes = dphelp.get_indices_and_count(modeling_data, "hugo_symbol")

# Index genes for sgRNAs.
sgrna_gene_mapping_df = (
    modeling_data[["hugo_symbol", "sgrna"]].drop_duplicates().reset_index(drop=True)
)
sgrna_gene_idx = dphelp.get_indices(sgrna_gene_mapping_df, "hugo_symbol")

# Index of genes for γ.
γ_gene_idx = dphelp.get_indices(
    modeling_data[["hugo_symbol"]].drop_duplicates().reset_index(drop=True),
    "hugo_symbol",
)


# Index cell lines for full data set.
cellline_idx, num_celllines = dphelp.get_indices_and_count(modeling_data, "depmap_id")
_, num_lineages = dphelp.get_indices_and_count(modeling_data, "lineage")

# Index lineages for cell lines.
cellline_lineage_mapping_df = (
    modeling_data[["lineage", "depmap_id"]].drop_duplicates().reset_index(drop=True)
)
cellline_lineage_idx = dphelp.get_indices(cellline_lineage_mapping_df, "lineage")
```

```python
with pm.Model() as m8:

    # Indexing arrays
    sgrna_idx_shared = pm.Data("sgrna_idx", sgrna_idx)
    gene_idx_shared = pm.Data("gene_idx", gene_idx)
    sgrna_gene_idx_shared = pm.Data("sgrna_gene_idx", sgrna_gene_idx)
    cellline_idx_shared = pm.Data("cellline_idx", cellline_idx)
    cellline_lineage_idx_shared = pm.Data("cellline_lineage_idx", cellline_lineage_idx)

    # Data
    lfc_shared = pm.Data("lfc", modeling_data.lfc.to_numpy())

    μ_μ_α = pm.Normal("μ_μ_α", 0, 3)
    σ_μ_α = pm.HalfNormal("σ_μ_α", 3)
    σ_σ_α = pm.HalfNormal("σ_σ_α", 3)
    μ_α = pm.Normal("μ_α", μ_μ_α, σ_μ_α, shape=num_genes)
    σ_α = pm.HalfNormal("σ_α", σ_σ_α, shape=num_genes)

    μ_μ_β = pm.Normal("μ_μ_β", 0, 3)
    σ_μ_β = pm.HalfNormal("σ_μ_β", 3)
    σ_σ_β = pm.HalfNormal("σ_σ_β", 3)
    μ_β = pm.Normal("μ_β", μ_μ_β, σ_μ_β, shape=num_lineages)
    σ_β = pm.HalfNormal("σ_β", σ_σ_β, shape=num_lineages)

    σ_σ_γ = pm.HalfNormal("σ_σ_γ", 1)
    #     μ_γ = pm.Deterministic("μ_γ", η[gene_idx_shared] + θ[cellline_idx_shared])
    μ_γ = pm.Normal("μ_γ", 0, 5)
    σ_γ = pm.HalfNormal("σ_γ", σ_σ_γ, shape=(num_genes))

    α = pm.Normal(
        "α", μ_α[sgrna_gene_idx_shared], σ_α[sgrna_gene_idx_shared], shape=num_sgrnas
    )
    β = pm.Normal(
        "β",
        μ_β[cellline_lineage_idx_shared],
        σ_β[cellline_lineage_idx_shared],
        shape=num_celllines,
    )
    γ = pm.Normal("γ", μ_γ, σ_γ, shape=(num_genes, num_celllines))

    μ = pm.Deterministic(
        "μ",
        α[sgrna_idx_shared]
        + β[cellline_idx_shared]
        + γ[gene_idx_shared, cellline_idx_shared],
    )
    σ = pm.HalfNormal("σ", 3)

    y = pm.Normal("y", μ, σ, observed=lfc_shared)
```

```python
pm.model_to_graphviz(m8)
```

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 28.69 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-01-27

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.19.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-230.o2.rc.hms.harvard.edu

    Git branch: data-subset-model

    seaborn   : 0.11.1
    plotnine  : 0.7.1
    matplotlib: 3.3.3
    re        : 2.2.1
    numpy     : 1.19.5
    pymc3     : 3.9.3
    arviz     : 0.11.0
    theano    : 1.0.5
    pandas    : 1.2.0

```python

```
