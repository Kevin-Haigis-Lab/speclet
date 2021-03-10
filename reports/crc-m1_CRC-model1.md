# Model Report

```python
import warnings
from pathlib import Path
from time import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
from analysis import color_pal as pal
from analysis import common_data_processing as dphelp
from analysis import pymc3_analysis as pmanal
from analysis import pymc3_sampling_api as pmapi
from analysis.pymc3_models import crc_models

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("pymc3_model_cache")
```

Parameters for papermill:

- `MODEL_CACHE_DIR`: location of cache
- `MODEL`: which model was tested
- `DEBUG`: if in debug mode or not

## Setup

### Papermill parameters

```python
MODEL_CACHE_DIR = Path("analysis/pymc3_model_cache/CRC-model1")
MODEL = "crc-m1"
DEBUG = True
```

### Data

```python
if DEBUG:
    data_path = Path("..", "modeling_data", "depmap_CRC_data_subsample.csv")
else:
    data_path = Path("..", "modeling_data", "depmap_CRC_data.csv")

data = dphelp.read_achilles_data(data_path)
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

### Model

> For conversion to parameterized notebook, I may need to refactor "sampling_pymc3_models.py" to be able to pass arguments without using the CLI. For instance, I would want to be able to use the same model generation function just by passing the string `"crc-m1"`, but I wouldn't want to sample the model.

```python
indices_dict = dphelp.common_indices(data)

model, shared_vars = crc_models.model_1(
    sgrna_idx=indices_dict["sgrna_idx"],
    sgrna_to_gene_idx=indices_dict["sgrna_to_gene_idx"],
    cellline_idx=indices_dict["cellline_idx"],
    batch_idx=indices_dict["batch_idx"],
    lfc_data=data.lfc.values,
)

model_res = pmapi.read_cached_vi(MODEL_CACHE_DIR)

model_az = az.from_pymc3(
    trace=model_res["trace"],
    model=model,
    posterior_predictive=model_res["posterior_predictive"],
    prior=model_res["prior_predictive"],
)
```

    Loading cached trace and posterior sample...

```python
pm.model_to_graphviz(model)
```

![svg](crc-m1_CRC-model1_files/crc-m1_CRC-model1_11_0.svg)

## Fit diagnostics

```python
pmanal.plot_vi_hist(model_res["approximation"])
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_13_0.png)

    <ggplot: (8770026000035)>

## Model parameters

```python
def check_shape(trace: np.ndarray) -> np.ndarray:
    if len(trace.shape) == 1:
        return trace[:, None]
    return trace


def add_hdi(p: gg.ggplot, values: np.ndarray, color: str) -> gg.ggplot:
    m = np.mean(values)
    hdi = az.hdi(values, hdi_prob=0.89).flatten()
    p = (
        p
        + gg.geom_vline(xintercept=m, color=color)
        + gg.geom_vline(xintercept=hdi, color=color, linetype="--")
    )
    return p


def variable_distribution_plot(var, trace: np.ndarray, max_plot=20000) -> gg.ggplot:
    trace = check_shape(trace)

    # Sample 25% of the trace.
    d = pd.DataFrame(trace).melt().assign(variable=lambda d: d.variable.astype("str"))
    d_summaries = d.groupby(["variable"])["value"].mean().reset_index(drop=False)

    if d.shape[0] > max_plot:
        d = d.sample(n=max_plot)
    else:
        d = d.sample(frac=0.2)

    p = (
        gg.ggplot(d, gg.aes(x="value"))
        + gg.geom_density(alpha=0.1)
        + gg.geom_vline(xintercept=0, color="black", size=0.7, alpha=0.7, linetype="--")
        + gg.scale_x_continuous(expand=(0, 0))
        + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
        + gg.theme(legend_position="none", figure_size=(6.5, 3))
        + gg.labs(x="posterior", y="density", title=f"Posterior distirbution of {var}")
    )

    c = pal.sns_blue

    if len(d_summaries) > 1:
        p = p + gg.geom_rug(
            data=d_summaries, sides="b", alpha=0.5, color=c, length=0.08
        )
    else:
        p = add_hdi(p, trace.flatten(), color=c)

    return p
```

```python
vars_to_inspect = model_res["trace"].varnames
vars_to_inspect = [v for v in vars_to_inspect if not "log" in v]
vars_to_inspect.sort()

for var in vars_to_inspect:
    trace = model_res["trace"][var]
    if len(trace.shape) > 1 and trace.shape[1] == data.shape[0]:
        # Do not plot the final deterministic mean (usually "μ").
        continue
    print(variable_distribution_plot(var, model_res["trace"][var]))
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_0.png)

    <ggplot: (8770016135461)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_2.png)

    <ggplot: (8770014826900)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_4.png)

    <ggplot: (8770016351687)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_6.png)

    <ggplot: (8770015799938)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_8.png)

    <ggplot: (8770016237138)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_10.png)

    <ggplot: (8770016083339)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_12.png)

    <ggplot: (8770016256741)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_14.png)

    <ggplot: (8770016207144)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_16.png)

    <ggplot: (8770014861590)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_18.png)

    <ggplot: (8770015178500)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_20.png)

    <ggplot: (8770016205373)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_22.png)

    <ggplot: (8770014772896)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_16_24.png)

    <ggplot: (8770014998821)>

## Model predicitons

```python
predictions = model_res["posterior_predictive"]
pred_summary = pmanal.summarize_posterior_predictions(
    predictions["lfc"],
    merge_with=data[["lfc", "depmap_id", "hugo_symbol", "sgrna", "log2_cn"]],
)
pred_summary["error"] = pred_summary.lfc - pred_summary.pred_mean
pred_summary.head()
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

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
      <th>pred_mean</th>
      <th>pred_hdi_low</th>
      <th>pred_hdi_high</th>
      <th>lfc</th>
      <th>depmap_id</th>
      <th>hugo_symbol</th>
      <th>sgrna</th>
      <th>log2_cn</th>
      <th>error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.147115</td>
      <td>-0.802405</td>
      <td>0.609533</td>
      <td>0.029491</td>
      <td>ACH-000007</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.861144</td>
      <td>0.176607</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.147575</td>
      <td>-0.907226</td>
      <td>0.628692</td>
      <td>0.426017</td>
      <td>ACH-000007</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.861144</td>
      <td>0.573592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.058426</td>
      <td>-0.870740</td>
      <td>0.655768</td>
      <td>0.008626</td>
      <td>ACH-000009</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.375470</td>
      <td>0.067052</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.045138</td>
      <td>-0.762936</td>
      <td>0.699539</td>
      <td>0.280821</td>
      <td>ACH-000009</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.375470</td>
      <td>0.325959</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.066573</td>
      <td>-0.828162</td>
      <td>0.687034</td>
      <td>0.239815</td>
      <td>ACH-000009</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.375470</td>
      <td>0.306389</td>
    </tr>
  </tbody>
</table>
</div>

```python
az.plot_loo_pit(model_az, y="lfc");
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_19_0.png)

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="pred_mean"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.geom_abline(slope=1, intercept=0, size=1, alpha=0.7, color="grey")
    + gg.geom_smooth(method="glm", color=pal.sns_red, size=1, alpha=0.7, se=False)
    + gg.labs(x="observed LFC", y="prediticed LFC (posterior avg.)")
)
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_20_0.png)

    <ggplot: (8770021275094)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="observed LFC", y="predition error")
)
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_21_0.png)

    <ggplot: (8770021569711)>

```python
gene_error = (
    pred_summary.groupby(["hugo_symbol"])["error"]
    .agg([np.mean, np.std])
    .reset_index(drop=False)
    .sort_values(["mean"])
    .reset_index(drop=True)
    .assign(
        hugo_symbol=lambda d: pd.Categorical(
            d.hugo_symbol.astype(str),
            categories=d.hugo_symbol.astype(str),
            ordered=True,
        )
    )
)

n_genes = 15

(
    gg.ggplot(
        gene_error.iloc[list(range(n_genes)) + list(range(-n_genes, -1))],
        gg.aes(x="hugo_symbol", y="mean"),
    )
    + gg.geom_col()
    + gg.theme(axis_text_x=gg.element_text(angle=90))
    + gg.labs(x="gene", y="error", title="Genes with the highest average error")
)
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_22_0.png)

    <ggplot: (8770016975487)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="log2_cn", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="gene copy number (log2)", y="predition error")
)
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_23_0.png)

    <ggplot: (8770021511527)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 1.72 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    The watermark extension is already loaded. To reload it, use:
      %reload_ext watermark
    Last updated: 2021-03-10

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-235.o2.rc.hms.harvard.edu

    Git branch: crc-m1

    matplotlib: 3.3.4
    arviz     : 0.11.1
    pymc3     : 3.11.1
    plotnine  : 0.7.1
    numpy     : 1.20.1
    pandas    : 1.2.2
