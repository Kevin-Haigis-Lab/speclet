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
from analysis import sampling_pymc3_models as sampling
from analysis.context_managers import set_directory
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

- `MODEL`: which model was tested
- `MODEL_NAME`: name of the model
- `DEBUG`: if in debug mode or not

## Setup

### Papermill parameters

```python
MODEL = "crc-m1"
MODEL_NAME = "CRC-model1"
DEBUG = True
```

```python
# Parameters
MODEL = "crc-m1"
MODEL_NAME = "CRC-model1"
DEBUG = True

```

```python
model_cache_dir = sampling.make_cache_name(MODEL_NAME)
```

```python
with set_directory(".."):
    model, _, data = sampling.main(
        model=MODEL, name=MODEL_NAME, sample=False, debug=DEBUG, random_seed=RANDOM_SEED
    )
```

    (🪲 debug mode)
     CRC Model 1
    Loading data...


    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/635332a6/lib/python3.9/site-packages/pymc3/data.py:316: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/635332a6/lib/python3.9/site-packages/pymc3/data.py:316: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.


    Done
    execution time: 0.17 minutes

### Data

```python
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

### Cached model fit

```python
model_res = pmapi.read_cached_vi(model_cache_dir)

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

![svg](crc-m1_CRC-model1_files/crc-m1_CRC-model1_13_0.svg)

## Fit diagnostics

```python
pmanal.plot_vi_hist(model_res["approximation"])
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_15_0.png)

    <ggplot: (8790230441689)>

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

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_0.png)

    <ggplot: (8790129279236)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_2.png)

    <ggplot: (8790230490567)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_4.png)

    <ggplot: (8790129245940)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_6.png)

    <ggplot: (8790129103613)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_8.png)

    <ggplot: (8790129279467)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_10.png)

    <ggplot: (8790129183648)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_12.png)

    <ggplot: (8790129187874)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_14.png)

    <ggplot: (8790128065374)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_16.png)

    <ggplot: (8790127868305)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_18.png)

    <ggplot: (8790128071593)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_20.png)

    <ggplot: (8790128273634)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_22.png)

    <ggplot: (8790127919339)>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_18_24.png)

    <ggplot: (8790129279395)>

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

    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/635332a6/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

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
      <td>-0.116332</td>
      <td>-0.892644</td>
      <td>0.627259</td>
      <td>0.029491</td>
      <td>ACH-000007</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.861144</td>
      <td>0.145823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.143943</td>
      <td>-0.799058</td>
      <td>0.639015</td>
      <td>0.426017</td>
      <td>ACH-000007</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.861144</td>
      <td>0.569960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.059316</td>
      <td>-0.820641</td>
      <td>0.637699</td>
      <td>0.008626</td>
      <td>ACH-000009</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.375470</td>
      <td>0.067942</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.063522</td>
      <td>-0.838232</td>
      <td>0.638793</td>
      <td>0.280821</td>
      <td>ACH-000009</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.375470</td>
      <td>0.344344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.045586</td>
      <td>-0.821382</td>
      <td>0.715843</td>
      <td>0.239815</td>
      <td>ACH-000009</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1.375470</td>
      <td>0.285402</td>
    </tr>
  </tbody>
</table>
</div>

```python
az.plot_loo_pit(model_az, y="lfc");
```

    <AxesSubplot:>

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_21_1.png)

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

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_22_0.png)

    <ggplot: (8790129360526)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="observed LFC", y="prediction error")
)
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_23_0.png)

    <ggplot: (8790128602064)>

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

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_24_0.png)

    <ggplot: (8790128592592)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="log2_cn", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7, linetype="--")
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7, linetype="--")
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="gene copy number (log2)", y="predition error")
)
```

![png](crc-m1_CRC-model1_files/crc-m1_CRC-model1_25_0.png)

    <ggplot: (8790127999877)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 2.77 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-03-12

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.21.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-104.o2.rc.hms.harvard.edu

    Git branch: papermill

    pandas    : 1.2.3
    arviz     : 0.11.2
    numpy     : 1.20.1
    pymc3     : 3.11.1
    plotnine  : 0.7.1
    matplotlib: 3.3.4
