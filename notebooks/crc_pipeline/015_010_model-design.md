# Designing models for CRC cell lines

```python
import re
import string
import warnings
from pathlib import Path
from time import time

import arviz as az
import color_pal as pal
import common_data_processing as dphelp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import pymc3_analysis as pmanal
import pymc3_sampling_api
import seaborn as sns
import theano
from pymc3_models import crc_models

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic() + gg.theme(strip_background=gg.element_blank()))
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 914
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("pymc3_model_cache")
```

## Data

```python
data = dphelp.read_achilles_data(
    Path("..", "modeling_data", "depmap_CRC_data_subsample.csv")
)
data.head()
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
      <th>pdna_batch</th>
      <th>passes_qc</th>
      <th>depmap_id</th>
      <th>primary_or_metastasis</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>kras_mutation</th>
      <th>...</th>
      <th>any_deleterious</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>log2_cn</th>
      <th>z_log2_cn</th>
      <th>is_mutated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.029491</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.861144</td>
      <td>1.386218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.426017</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.861144</td>
      <td>1.386218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>0.008626</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.234394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.280821</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.234394</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.239815</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.234394</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>

## Model Experimentation

```python
total_size = len(data.lfc.values)
sgrna_idx, n_sgrnas = dphelp.get_indices_and_count(data, "sgrna")
sgrna_to_gene_map = (
    data[["sgrna", "hugo_symbol"]]
    .drop_duplicates()
    .reset_index(drop=True)
    .sort_values("sgrna")
    .reset_index(drop=True)
)
sgrna_to_gene_idx, n_genes = dphelp.get_indices_and_count(
    sgrna_to_gene_map, "hugo_symbol"
)
cellline_idx, n_celllines = dphelp.get_indices_and_count(data, "depmap_id")
batch_idx, n_batches = dphelp.get_indices_and_count(data, "pdna_batch")
```

```python
sgrna_idx_shared = theano.shared(sgrna_idx)
sgrna_to_gene_idx_shared = theano.shared(sgrna_to_gene_idx)
cellline_idx_shared = theano.shared(cellline_idx)
batch_idx_shared = theano.shared(batch_idx)
lfc_shared = theano.shared(data.lfc.values)
```

```python
with pm.Model() as model:

    mu_g = pm.Normal("mu_g", data.lfc.values.mean(), 1)
    sigma_g = pm.HalfNormal("sigma_g", 2)
    sigma_sigma_alpha = pm.HalfNormal("sigma_sigma_alpha", 1)

    mu_alpha = pm.Normal("mu_alpha", mu_g, sigma_g, shape=n_genes)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma_sigma_alpha, shape=n_genes)
    mu_beta = pm.Normal("mu_beta", 0, 0.2)
    sigma_beta = pm.HalfNormal("sigma_beta", 1)
    mu_eta = pm.Normal("mu_eta", 0, 0.2)
    sigma_eta = pm.HalfNormal("sigma_eta", 1)

    alpha_s = pm.Normal(
        "alpha_s",
        mu_alpha[sgrna_to_gene_idx_shared],
        sigma_alpha[sgrna_to_gene_idx_shared],
        shape=n_sgrnas,
    )
    beta_l = pm.Normal("beta_l", mu_beta, sigma_beta, shape=n_celllines)
    eta_b = pm.Normal("eta_b", mu_eta, sigma_eta, shape=n_batches)

    mu = pm.Deterministic(
        "mu",
        alpha_s[sgrna_idx_shared]
        + beta_l[cellline_idx_shared]
        + eta_b[batch_idx_shared],
    )
    sigma = pm.HalfNormal("sigma", 2)

    lfc = pm.Normal("lfc", mu, sigma, observed=lfc_shared, total_size=total_size)
```

```python
pm.model_to_graphviz(model)
```

![svg](015_010_model-design_files/015_010_model-design_8_0.svg)

```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=RANDOM_SEED)
```

```python
pmanal.plot_all_priors(prior_pred, subplots=(5, 3), figsize=(10, 8), samples=500);
```

![png](015_010_model-design_files/015_010_model-design_10_0.png)

```python
batch_size = 1000

sgnra_idx_batch = pm.Minibatch(sgrna_idx, batch_size=batch_size)
cellline_idx_batch = pm.Minibatch(cellline_idx, batch_size=batch_size)
batch_idx_batch = pm.Minibatch(batch_idx, batch_size=batch_size)
lfc_data_batch = pm.Minibatch(data.lfc.values, batch_size=batch_size)
```

```python
meanfield = pymc3_sampling_api.pymc3_advi_approximation_procedure(
    model=model,
    method="advi",
    callbacks=[
        pm.callbacks.CheckParametersConvergence(tolerance=0.01, diff="absolute")
    ],
    fit_kwargs={
        "more_replacements": {
            sgrna_idx_shared: sgnra_idx_batch,
            cellline_idx_shared: cellline_idx_batch,
            batch_idx_shared: batch_idx_batch,
            lfc_shared: lfc_data_batch,
        }
    },
)
```

    Sampling from prior distributions.
    Running ADVI approximation.

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
  <progress value='0' class='' max='100000' style='width:300px; height:20px; vertical-align: middle;'></progress>

</div>

    Convergence achieved at 27600
    Interrupted at 27,599 [27%]: Average Loss = 967.8
    Sampling from posterior.
    Posterior predicitons.

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
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 01:11<00:00]
</div>

```python
az_model = pymc3_sampling_api.samples_to_arviz(model, meanfield)
```

```python
pmanal.plot_vi_hist(meanfield["approximation"])
```

![png](015_010_model-design_files/015_010_model-design_14_0.png)

    <ggplot: (351612685)>

```python
def plot_az_summary(df: pd.DataFrame, x="index") -> gg.ggplot:
    return (
        gg.ggplot(df, gg.aes(x=x))
        + gg.geom_hline(yintercept=0, linetype="--", alpha=0.5)
        + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), size=0.2)
        + gg.geom_point(gg.aes(y="mean"), size=0.5)
        + gg.theme(axis_text_x=gg.element_text(angle=90, size=6))
        + gg.labs(x="model parameter", y="posterior")
    )
```

```python
batch_posteriors = az.summary(az_model, var_names="eta_b", hdi_prob=0.89)
plot_az_summary(batch_posteriors.reset_index(drop=False))
```

    arviz - WARNING - Shape validation failed: input_shape: (1, 1000), minimum_shape: (chains=2, draws=4)

![png](015_010_model-design_files/015_010_model-design_16_1.png)

    <ggplot: (348681807)>

```python
gene_posteriors = az.summary(
    az_model,
    var_names="mu_alpha",
    hdi_prob=0.89,
)

plot_az_summary(gene_posteriors.reset_index(drop=False)) + gg.theme(figure_size=(10, 5))
```

    arviz - WARNING - Shape validation failed: input_shape: (1, 1000), minimum_shape: (chains=2, draws=4)

![png](015_010_model-design_files/015_010_model-design_17_1.png)

    <ggplot: (349306646)>

```python
sgrna_post = (
    az.summary(
        az_model,
        var_names="alpha_s",
        hdi_prob=0.89,
        kind="stats",
    )
    .reset_index(drop=False)
    .rename(columns={"index": "param"})
)
sgrna_post["sgrna"] = sgrna_to_gene_map.sgrna
sgrna_post["hugo_symbol"] = sgrna_to_gene_map.hugo_symbol

(
    gg.ggplot(sgrna_post, gg.aes(x="param"))
    + gg.facet_wrap("hugo_symbol", scales="free", ncol=5)
    + gg.geom_hline(yintercept=0, alpha=0.5, linetype="--")
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), size=0.2)
    + gg.geom_point(gg.aes(y="mean"), size=0.6)
    + gg.theme(
        axis_text_x=gg.element_blank(),
        figure_size=(8, 20),
        subplots_adjust={"wspace": 0.5},
        axis_text_y=gg.element_text(size=6),
        axis_ticks_major_x=gg.element_blank(),
        strip_text=gg.element_text(size=7),
    )
)
```

![png](015_010_model-design_files/015_010_model-design_18_0.png)

    <ggplot: (351348363)>

```python
cells_posteriors = az.summary(
    az_model,
    var_names="beta_l",
    hdi_prob=0.89,
)

plot_az_summary(cells_posteriors.reset_index(drop=False)) + gg.theme(figure_size=(8, 3))
```

    arviz - WARNING - Shape validation failed: input_shape: (1, 1000), minimum_shape: (chains=2, draws=4)

![png](015_010_model-design_files/015_010_model-design_19_1.png)

    <ggplot: (352243444)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 5.31 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-03-02

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    Compiler    : Clang 11.0.1
    OS          : Darwin
    Release     : 20.3.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: JHCookMac

    Git branch: crc-m1

    pymc3     : 3.11.1
    sys       : 3.9.1 | packaged by conda-forge | (default, Jan 26 2021, 01:32:59)
    [Clang 11.0.1 ]
    numpy     : 1.20.1
    re        : 2.2.1
    matplotlib: 3.3.4
    plotnine  : 0.7.1
    arviz     : 0.11.1
    seaborn   : 0.11.1
    theano    : 1.0.5
    pandas    : 1.2.2
