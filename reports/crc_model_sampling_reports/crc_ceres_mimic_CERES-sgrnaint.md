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
import seaborn as sns
from src.command_line_interfaces import sampling_pymc3_models_cli as sampling
from src.data_processing import common as dphelp
from src.modeling import pymc3_analysis as pmanal
from src.modeling import pymc3_sampling_api as pmapi
from src.plot.color_pal import SeabornColor

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("..", "models", "modeling_cache", "pymc3_model_cache")
```

Parameters for papermill:

- `MODEL`: which model was tested
- `MODEL_NAME`: name of the model
- `DEBUG`: if in debug mode or not

## Setup

### Papermill parameters

```python
MODEL = ""
MODEL_NAME = ""
DEBUG = True
```

```python
# Parameters
MODEL = "crc_ceres_mimic"
MODEL_NAME = "CERES-sgrnaint"
DEBUG = False
```

```python
speclet_model = sampling.sample_speclet_model(
    MODEL, name=MODEL_NAME, debug=DEBUG, random_seed=RANDOM_SEED, touch=False
)
```

    (INFO) Cache directory: /n/data2/dfci/cancerbio/haigis/Cook/speclet/models/model_cache/pymc3_model_cache/CERES-sgrnaint
    (INFO) Sampling 'crc_ceres_mimic' with custom name 'CERES-sgrnaint'
    (INFO) Including sgRNA|gene varying intercept in CERES model.
    (INFO) Running model build method.
    (INFO) Running ADVI fitting method.
    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/f4a519a1/lib/python3.9/site-packages/pymc3/data.py:316: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/f4a519a1/lib/python3.9/site-packages/pymc3/data.py:316: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    (INFO) finished; execution time: 4.42 minutes

```python
model_res = speclet_model.advi_results
```

### Data

```python
data = speclet_model.data_manager.get_data()
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
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.096711</td>
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
      <td>0.0</td>
      <td>1.407616</td>
      <td>-0.317653</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.804148</td>
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
      <td>0.0</td>
      <td>1.407616</td>
      <td>-0.317653</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>-0.091043</td>
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
      <td>0.0</td>
      <td>1.848518</td>
      <td>1.385723</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.339692</td>
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
      <td>0.0</td>
      <td>1.848518</td>
      <td>1.385723</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.211244</td>
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
      <td>0.0</td>
      <td>1.848518</td>
      <td>1.385723</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>

### Cached model fit

```python
model_az = pmapi.convert_samples_to_arviz(
    model=speclet_model.model, res=speclet_model.advi_results
)
```

```python
pm.model_to_graphviz(speclet_model.model)
```

![svg](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_13_0.svg)

## Fit diagnostics

```python
pmanal.plot_vi_hist(model_res.approximation)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_15_0.png)

    <ggplot: (2970930748709)>

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

    c = SeabornColor.blue

    if len(d_summaries) > 1:
        p = p + gg.geom_rug(
            data=d_summaries, sides="b", alpha=0.5, color=c, length=0.08
        )
    else:
        p = add_hdi(p, trace.flatten(), color=c)

    return p
```

```python
vars_to_inspect = model_res.trace.varnames
vars_to_inspect = [v for v in vars_to_inspect if not "log" in v]
vars_to_inspect.sort()

for var in vars_to_inspect:
    trace = model_res.trace[var]
    if len(trace.shape) > 1 and trace.shape[1] == data.shape[0]:
        # Do not plot the final deterministic mean (usually "μ").
        continue
    try:
        print(variable_distribution_plot(var, model_res.trace[var]))
    except Exception as err:
        print(f"Skipping variable '{var}'.")
        print(err)
```

    Skipping variable 'a'.
    Must pass 2-d input. shape=(1000, 997, 2)
    Skipping variable 'd'.
    Must pass 2-d input. shape=(1000, 997, 36)

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_1.png)

    <ggplot: (2970227383971)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_3.png)

    <ggplot: (2970943292115)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_5.png)

    <ggplot: (2970943796938)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_7.png)

    <ggplot: (2970942601906)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_9.png)

    <ggplot: (2970940170515)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_11.png)

    <ggplot: (2970943554871)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_13.png)

    <ggplot: (2970942090957)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_15.png)

    <ggplot: (2970943769640)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_17.png)

    <ggplot: (2970940892365)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_19.png)

    <ggplot: (2970943796878)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_21.png)

    <ggplot: (2970955791351)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_23.png)

    <ggplot: (2970942090987)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_25.png)

    <ggplot: (2970940179017)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_27.png)

    <ggplot: (2970945570301)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_29.png)

    <ggplot: (2970942194466)>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_18_31.png)

    <ggplot: (2970941983964)>

## Model predicitons

```python
predictions = model_res.posterior_predictive
pred_summary = pmanal.summarize_posterior_predictions(
    predictions["lfc"],
    merge_with=data,
    calc_error=True,
    observed_y="lfc",
)
pred_summary.head()
```

    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/f4a519a1/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

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
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>pdna_batch</th>
      <th>passes_qc</th>
      <th>depmap_id</th>
      <th>primary_or_metastasis</th>
      <th>...</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>log2_cn</th>
      <th>z_log2_cn</th>
      <th>is_mutated</th>
      <th>error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.279200</td>
      <td>-0.299396</td>
      <td>0.938857</td>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.096711</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.407616</td>
      <td>-0.317653</td>
      <td>0</td>
      <td>-0.182489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.264571</td>
      <td>-0.408639</td>
      <td>0.949592</td>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.804148</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.407616</td>
      <td>-0.317653</td>
      <td>0</td>
      <td>0.539577</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.242681</td>
      <td>-0.497230</td>
      <td>0.876549</td>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>-0.091043</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.848518</td>
      <td>1.385723</td>
      <td>0</td>
      <td>-0.333724</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.246428</td>
      <td>-0.466466</td>
      <td>0.909590</td>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.339692</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.848518</td>
      <td>1.385723</td>
      <td>0</td>
      <td>0.093264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.238504</td>
      <td>-0.406479</td>
      <td>0.869564</td>
      <td>ACAAACCTCTCTACACCCCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.211244</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.0</td>
      <td>1.848518</td>
      <td>1.385723</td>
      <td>0</td>
      <td>-0.027260</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>

```python
az.plot_loo_pit(model_az, y="lfc");
```

    <AxesSubplot:>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_21_1.png)

```python
model_loo = az.loo(model_az, pointwise=True)
print(model_loo)
```

    Computed from 1000 by 349184 log-likelihood matrix

             Estimate       SE
    elpd_loo -208052.11   716.33
    p_loo    32109.47        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                              Count   Pct.
    (-Inf, 0.5]   (good)     322474   92.4%
     (0.5, 0.7]   (ok)        22856    6.5%
       (0.7, 1]   (bad)        3084    0.9%
       (1, Inf)   (very bad)    770    0.2%

```python
sns.distplot(model_loo.loo_i.values);
```

    /n/data2/dfci/cancerbio/haigis/Cook/speclet/.snakemake/conda/f4a519a1/lib/python3.9/site-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).





    <AxesSubplot:ylabel='Density'>

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_23_2.png)

```python
pred_summary["loo"] = model_loo.loo_i.values
```

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="pred_mean"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.geom_abline(slope=1, intercept=0, size=1, alpha=0.7, color="grey")
    + gg.geom_smooth(method="glm", color=SeabornColor.red, size=1, alpha=0.7, se=False)
    + gg.labs(x="observed LFC", y="prediticed LFC (posterior avg.)")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_25_0.png)

    <ggplot: (2970943708941)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="loo"))
    + gg.geom_point(gg.aes(color="np.abs(error)"), alpha=0.5)
    + gg.scale_color_gradient(low="grey", high="red")
    + gg.theme()
    + gg.labs(x="observed LFC", y="LOO", color="abs(error)")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_26_0.png)

    <ggplot: (2970271502845)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="np.abs(error)", y="loo"))
    + gg.geom_point(gg.aes(color="lfc"), alpha=0.5)
    + gg.labs(x="abs(error)", y="loo", color="LFC")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_27_0.png)

    <ggplot: (2970943715780)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="observed LFC", y="prediction error")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_28_0.png)

    <ggplot: (2970946757639)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="hugo_symbol", y="loo"))
    + gg.geom_point(alpha=0.2, size=0.7)
    + gg.geom_boxplot(outlier_alpha=0, alpha=0.4)
    + gg.theme(axis_text_x=gg.element_blank(), axis_ticks_major_x=gg.element_blank())
    + gg.labs(x="genes", y="LOO")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_29_0.png)

    <ggplot: (2970946057611)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="depmap_id", y="loo"))
    + gg.geom_jitter(width=0.2, alpha=0.3, size=0.7)
    + gg.geom_boxplot(outlier_alpha=0, alpha=0.4)
    + gg.theme(
        axis_text_x=gg.element_text(angle=90, size=8),
    )
    + gg.labs(x="cell lines", y="LOO")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_30_0.png)

    <ggplot: (2970950569382)>

```python
# Remove samples without gene CN data.
ppc_df_no_missing = pred_summary.copy()[~pred_summary.gene_cn.isna()]
ppc_df_no_missing["binned_gene_cn"] = [
    np.min([round(x), 10]) for x in ppc_df_no_missing.gene_cn
]

(
    gg.ggplot(ppc_df_no_missing, gg.aes(x="factor(binned_gene_cn)", y="loo"))
    + gg.geom_jitter(size=0.6, alpha=0.5, width=0.3)
    + gg.geom_boxplot(outlier_alpha=0, alpha=0.8)
    + gg.labs(x="gene copy number (max. 10)", y="LOO")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_31_0.png)

    <ggplot: (2970946060060)>

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

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_32_0.png)

    <ggplot: (2970950569328)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="log2_cn", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7, linetype="--")
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7, linetype="--")
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="gene copy number (log2)", y="predition error")
)
```

![png](crc_ceres_mimic_CERES-sgrnaint_files/crc_ceres_mimic_CERES-sgrnaint_33_0.png)

    <ggplot: (2971337434603)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 23.92 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-04-26

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

    Hostname: compute-a-16-165.o2.rc.hms.harvard.edu

    Git branch: update-pipelines

    plotnine  : 0.7.1
    matplotlib: 3.3.4
    seaborn   : 0.11.1
    arviz     : 0.11.2
    pymc3     : 3.11.1
    pandas    : 1.2.3
    numpy     : 1.20.1
