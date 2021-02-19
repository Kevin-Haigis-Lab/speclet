# Analyzing CRC models

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
import pymc3_sampling_api
import seaborn as sns
from pymc3_models import crc_models
from theano import tensor as tt

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic() + gg.theme(strip_background=gg.element_blank()))
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("pymc3_model_cache")
```

## Data

```python
data_dir = Path("..", "modeling_data", "depmap_CRC_data_subsample.csv")
data = dphelp.read_achilles_data(data_dir, low_memory=False)
data.shape
```

    (34760, 30)

## Model 1

```python
m1_cache_dir = pymc3_cache_dir / "CRC_test_model1"

gene_idx = dphelp.get_indices(data, "hugo_symbol")

crc_model1, gene_idx_shared, lfc_data_shared = crc_models.model_1(
    gene_idx=gene_idx, lfc_data=data.lfc.values
)
pm.model_to_graphviz(crc_model1)
```

![svg](015_020_crc-model-analysis_files/015_020_crc-model-analysis_5_0.svg)

```python
crc_model1_res = pymc3_sampling_api.read_cached_vi(m1_cache_dir)
```

    Loading cached trace and posterior sample...

```python
d = pd.DataFrame({"loss": crc_model1_res["approximation"].hist}).assign(
    step=lambda d: np.arange(d.shape[0])
)

(
    gg.ggplot(d, gg.aes(x="step", y="loss"))
    + gg.geom_line(size=0.5, alpha=0.75, color="black")
)
```

![png](015_020_crc-model-analysis_files/015_020_crc-model-analysis_7_0.png)

    <ggplot: (8731821608241)>

```python
crc_m1_az = az.from_pymc3(trace=crc_model1_res["trace"], model=crc_model1)
az.summary(crc_m1_az, var_names=["μ_α", "σ_α", "σ"], hdi_prob=0.89)
```

    arviz - WARNING - Shape validation failed: input_shape: (1, 1000), minimum_shape: (chains=2, draws=4)

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
      <th>μ_α</th>
      <td>-0.128</td>
      <td>0.036</td>
      <td>-0.186</td>
      <td>-0.072</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>996.0</td>
      <td>996.0</td>
      <td>995.0</td>
      <td>1024.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>σ_α</th>
      <td>0.319</td>
      <td>0.025</td>
      <td>0.282</td>
      <td>0.360</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1031.0</td>
      <td>1031.0</td>
      <td>1023.0</td>
      <td>905.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>0.510</td>
      <td>0.008</td>
      <td>0.495</td>
      <td>0.522</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1038.0</td>
      <td>1038.0</td>
      <td>1035.0</td>
      <td>862.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
gene_post = az.summary(crc_m1_az, var_names=["α_g"], hdi_prob=0.89, kind="stats")
gene_post["hugo_symbol"] = data.hugo_symbol.cat.categories.values
gene_post = gene_post.merge(
    data.groupby("hugo_symbol")["lfc"].agg(np.mean).reset_index(drop=False),
    on="hugo_symbol",
)

gene_ggplot_theme = {
    "axis_text_x": gg.element_text(angle=90, size=6),
    "figure_size": (10, 6),
}

(
    gg.ggplot(gene_post, gg.aes(x="hugo_symbol"))
    + gg.geom_hline(yintercept=0, alpha=0.5, linetype="--")
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), alpha=0.5)
    + gg.geom_point(gg.aes(y="mean"))
    + gg.theme(**gene_ggplot_theme)
    + gg.labs(x="gene", y=r"$\alpha_g$ posterior")
)
```

![png](015_020_crc-model-analysis_files/015_020_crc-model-analysis_9_0.png)

    <ggplot: (8731783520446)>

```python
(
    gg.ggplot(gene_post, gg.aes(x="lfc", y="mean"))
    + gg.geom_abline(
        slope=1, intercept=0, linetype="--", color=pal.sns_orange, size=1.2
    )
    + gg.geom_point(gg.aes(size="sd"), alpha=0.5, color=pal.sns_blue)
    + gg.labs(x="observed LFC", y=r"$\alpha_g$ posterior", size="std. dev.")
)
```

![png](015_020_crc-model-analysis_files/015_020_crc-model-analysis_10_0.png)

    <ggplot: (8731783517546)>

The posterior predictive check has the wrong dimensions; it is using the last minibatch for predicitions.
I need to follow the guidance of this [post](https://discourse.pymc.io/t/minibatch-advi-ppc-dimensions/5583) on the Discourse which followed the explanation in this GitHub [issue](https://github.com/pymc-devs/pymc3/issues/2190#issuecomment-311609342).
Should be a simple fix.

```python
crc_model1_res["posterior_predictive"]["lfc"].shape[1] == data.shape[0]
```

    True

```python
def ppc_dataframe(ppc: np.ndarray, hdi_prob=0.89) -> pd.DataFrame:
    hdi = az.hdi(ppc, hdi_prob=hdi_prob)
    return pd.DataFrame(
        {
            "ppc_mean": ppc.mean(axis=0),
            "ppc_hdi_low": hdi[:, 0],
            "ppc_hdi_high": hdi[:, 1],
        }
    )


m1_ppc_df = ppc_dataframe(crc_model1_res["posterior_predictive"]["lfc"])
m1_ppc_df["real_lfc"] = data.lfc.values
m1_ppc_df["hugo_symbol"] = data.hugo_symbol.values

m1_ppc_df.head()
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
      <th>ppc_mean</th>
      <th>ppc_hdi_low</th>
      <th>ppc_hdi_high</th>
      <th>real_lfc</th>
      <th>hugo_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.140320</td>
      <td>-1.001702</td>
      <td>0.618150</td>
      <td>0.029491</td>
      <td>ADAMTS13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.128909</td>
      <td>-0.896227</td>
      <td>0.687206</td>
      <td>0.426017</td>
      <td>ADAMTS13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.144695</td>
      <td>-0.948575</td>
      <td>0.670772</td>
      <td>0.008626</td>
      <td>ADAMTS13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.109424</td>
      <td>-0.900942</td>
      <td>0.655753</td>
      <td>0.280821</td>
      <td>ADAMTS13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.141217</td>
      <td>-1.015847</td>
      <td>0.581773</td>
      <td>0.239815</td>
      <td>ADAMTS13</td>
    </tr>
  </tbody>
</table>
</div>

```python
(
    gg.ggplot(m1_ppc_df, gg.aes(x="real_lfc", y="ppc_mean"))
    + gg.geom_abline(slope=1, intercept=0, color="gray", linetype="--", size=1)
    + gg.scale_color_discrete()
    + gg.geom_point(gg.aes(color="hugo_symbol"), size=0.2, alpha=0.5)
    + gg.theme(legend_position="none")
    + gg.labs(x="observed LFC", y="predicted LFC")
)
```

![png](015_020_crc-model-analysis_files/015_020_crc-model-analysis_14_0.png)

    <ggplot: (8731821607376)>

## Predict on new data

```python
pred_data = data[["hugo_symbol"]].drop_duplicates().reset_index(drop=True)

pred_gene_idx = dphelp.get_indices(pred_data, "hugo_symbol")
gene_idx_shared.set_value(pred_gene_idx)
```

```python
with crc_model1:
    pred_ppc = pm.sample_posterior_predictive(
        trace=crc_model1_res["trace"], samples=100
    )
```

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
  <progress value='100' class='' max='100' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [100/100 00:00<00:00]
</div>

```python
new_ppc = ppc_dataframe(pred_ppc["lfc"])
new_ppc["hugo_symbol"] = pred_data.hugo_symbol.values
(
    gg.ggplot(new_ppc, gg.aes(x="hugo_symbol"))
    + gg.geom_hline(yintercept=0, alpha=0.5, linetype="--")
    + gg.geom_linerange(
        gg.aes(ymin="ppc_hdi_low", ymax="ppc_hdi_high"), size=0.5, alpha=0.5
    )
    + gg.geom_point(gg.aes(y="ppc_mean"), size=1)
    + gg.theme(**gene_ggplot_theme)
    + gg.labs(x="gene", y="posterior prediction")
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

![png](015_020_crc-model-analysis_files/015_020_crc-model-analysis_18_1.png)

    <ggplot: (8731783517537)>

### To-Do:

1. Address the change above to fix an issue with PPC sampling. <span style="color:green">✔︎</span>
2. Make sure I can replace the shared data with new data for predictions on unseen data. <span style="color:green">✔︎</span>
3. Run and analyze more simple models.
4. Create a Snakemake workflow to run all in parallel and then run the analysis.

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 0.97 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-02-19

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.20.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-86.o2.rc.hms.harvard.edu

    Git branch: crc

    arviz     : 0.11.1
    numpy     : 1.20.1
    re        : 2.2.1
    pandas    : 1.2.2
    plotnine  : 0.7.1
    matplotlib: 3.3.4
    pymc3     : 3.11.1
    theano    : 1.0.5
    seaborn   : 0.11.1
