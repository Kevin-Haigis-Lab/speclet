# Negative binomial modeling of simulated CRISPR screen data

```python
%load_ext autoreload
%autoreload 2
```

```python
import re
import string
import warnings
from pathlib import Path
from time import time
from typing import Any, Union

import arviz as az
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from scipy import stats
from theano import tensor as tt
```

```python
from src.analysis import pymc3_analysis as pmanal
from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.globals import PYMC3
from src.io import cache_io
from src.loggers import logger
from src.modeling import pymc3_sampling_api as pmapi
from src.plot.color_pal import FitMethodColors, ModelColors, SeabornColor
```

```python
notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(
    gg.theme_classic()
    + gg.theme(
        figure_size=(4, 4),
        axis_ticks_major=gg.element_blank(),
        strip_background=gg.element_blank(),
    )
)
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 1042
np.random.seed(RANDOM_SEED)
```

sources of simulated NB data:

- [PyMC3 docs: "GLM: Negative Binomial Regression"](https://docs.pymc.io/pymc-examples/examples/generalized_linear_models/GLM-negative-binomial-regression.html)

```python
def get_nb_vals(mu, alpha, size):
    """Generate negative binomially distributed samples by
    drawing a sample from a gamma distribution with mean `mu` and
    shape parameter `alpha', then drawing from a Poisson
    distribution whose rate parameter is given by the sampled
    gamma variable.

    """

    g = stats.gamma.rvs(alpha, scale=mu / alpha, size=size)
    return stats.poisson.rvs(g)
```

## Simulation 1. Single sgRNA

```python
n = 100  # number simulated data points

# Simulation parameters.
sim1: dict[str, Any] = {"β": -0.5, "α": 2.0}

# Simulated data.
sim1_data = pd.DataFrame({"initial_read_count": np.random.poisson(100, n)})
sim1_data["initial_read_count_log"] = np.log(sim1_data)
eta = sim1["β"]

mu = np.exp(eta) * sim1_data["initial_read_count"]
sim1_data["final_read_count"] = [
    get_nb_vals(mu[i], sim1["α"], size=1) for i in range(n)
]

sim1_data.head()
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
      <th>initial_read_count</th>
      <th>initial_read_count_log</th>
      <th>final_read_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>95</td>
      <td>4.553877</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>110</td>
      <td>4.700480</td>
      <td>128</td>
    </tr>
    <tr>
      <th>2</th>
      <td>122</td>
      <td>4.804021</td>
      <td>37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88</td>
      <td>4.477337</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>124</td>
      <td>4.820282</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>

```python
sns.jointplot(data=sim1_data, x="initial_read_count", y="final_read_count");
```

![png](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_9_0.png)

Simulation 1 model `sim1_model`:

$$
\begin{gather}
\beta \sim N(0, 2) \\
\eta = \beta
\end{gather}
$$

Simulation 1 model `sim1_model`:

$$
\begin{aligned}
\beta &\sim N(0, 2) \\
\eta &= \beta \\
\mu &= \exp(\eta) \times \text{initial read counts} \\
\alpha &\sim \text{HalfNormal}(0, 5) \\
y &\sim \text{NB}(\mu, \alpha)
\end{aligned}
$$

```python
with pm.Model() as sim1_model:
    β = pm.Normal("β", 0, 2)
    η = pm.Deterministic("η", β)
    μ = pm.Deterministic("μ", pm.math.exp(η) * sim1_data.initial_read_count.values)
    α = pm.HalfNormal("α", 5)
    y = pm.NegativeBinomial("y", μ, α, observed=sim1_data.final_read_count.values)
```

```python
pm.model_to_graphviz(sim1_model)
```

![svg](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_13_0.svg)

```python
with sim1_model:
    sim1_trace = pm.sample(
        tune=2000, random_seed=850, chains=4, return_inferencedata=True
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, β]

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
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 00:14<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 2_000 tune and 1_000 draw iterations (8_000 + 4_000 draws total) took 35 seconds.

```python
az.plot_trace(sim1_trace, var_names=["β", "α"]);
```

![png](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_15_0.png)

```python
az.summary(sim1_trace, var_names=["α", "β"])
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
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>α</th>
      <td>2.190</td>
      <td>0.305</td>
      <td>1.649</td>
      <td>2.751</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>3467.0</td>
      <td>2604.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β</th>
      <td>-0.529</td>
      <td>0.070</td>
      <td>-0.656</td>
      <td>-0.396</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3598.0</td>
      <td>2925.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

### Scaling the initial number of reads instead of including as exposure.

Simulation 1 model `sim2_model`:

$$
\begin{aligned}
\beta &\sim N(0, 2) \\
\eta &= \beta + \beta_s \frac{X_\text{initial reads}}{100} \\
\mu &= \exp(\eta) \\
\alpha &\sim \text{HalfNormal}(0, 5) \\
y &\sim \text{NB}(\mu, \alpha)
\end{aligned}
$$

```python
with pm.Model() as sim1_model_rescale:
    β = pm.Normal("β", 0, 2)
    β_s = pm.Normal("β_s", 0, 2)
    η = pm.Deterministic("η", β + β_s * (sim1_data.initial_read_count / 100))
    μ = pm.Deterministic("μ", pm.math.exp(η))
    α = pm.HalfNormal("α", 5)
    y = pm.NegativeBinomial("y", μ, α, observed=sim1_data.final_read_count.values)
```

```python
pm.model_to_graphviz(sim1_model_rescale)
```

![svg](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_20_0.svg)

```python
with sim1_model_rescale:
    sim1_rescale_trace = pm.sample(
        tune=2000, chains=4, random_seed=823, return_inferencedata=True
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, β_s, β]

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
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 00:48<00:00 Sampling 4 chains, 4 divergences]
</div>

    Sampling 4 chains for 2_000 tune and 1_000 draw iterations (8_000 + 4_000 draws total) took 66 seconds.
    There were 4 divergences after tuning. Increase `target_accept` or reparameterize.

```python
az.plot_trace(
    sim1_rescale_trace, var_names=["β", "α"], filter_vars="like", compact=False
);
```

![png](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_22_0.png)

```python
az.summary(sim1_rescale_trace, var_names=["α", "β"], filter_vars="like")
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
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>β</th>
      <td>2.185</td>
      <td>0.640</td>
      <td>0.984</td>
      <td>3.408</td>
      <td>0.020</td>
      <td>0.014</td>
      <td>1014.0</td>
      <td>1258.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β_s</th>
      <td>1.884</td>
      <td>0.636</td>
      <td>0.723</td>
      <td>3.136</td>
      <td>0.020</td>
      <td>0.014</td>
      <td>1018.0</td>
      <td>1253.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>α</th>
      <td>2.190</td>
      <td>0.307</td>
      <td>1.609</td>
      <td>2.767</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>1223.0</td>
      <td>813.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

The issue is definitely colinearity between \[beta\] and $\beta_s$.
In this case, since the real value for $\beta_s = 1$, this model is equivalent to `sim1_model` where the initial read count is included as an exposure.

```python
az.plot_joint(sim1_rescale_trace, var_names=["β", "β_s"]);
```

![png](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_25_0.png)

## Simulation 2. Multiple sgRNA

```python
n_per_guide = 10  # number simulated data points
n_guides = 10
n = n_per_guide * n_guides

guides = [f"sgrna_{i:02d}" for i in range(n_guides)]

# Simulation parameters.
sim2: dict[str, Any] = {"μ_β": -1.0, "σ_β": 1.0, "α": 2.0}
sim2["β"] = np.random.normal(sim2["μ_β"], sim2["σ_β"], size=n_guides)

# Simulated data.
sim2_data = pd.DataFrame(
    {
        "initial_read_count": np.random.poisson(100, n_per_guide * n_guides),
        "sgrna": np.repeat(guides, n_per_guide),
    }
)
sim2_data["sgrna"] = pd.Categorical(sim2_data["sgrna"], categories=guides, ordered=True)
eta = [sim2["β"][i] for i in sim2_data.sgrna.factorize(sort=True)[0]]

mu = np.exp(eta) * sim2_data["initial_read_count"]
sim2_data["final_read_count"] = [
    get_nb_vals(mu[i], sim2["α"], size=1) for i in range(n)
]

sim2_data.head()
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
      <th>initial_read_count</th>
      <th>sgrna</th>
      <th>final_read_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>sgrna_00</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>87</td>
      <td>sgrna_00</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>sgrna_00</td>
      <td>105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93</td>
      <td>sgrna_00</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>110</td>
      <td>sgrna_00</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>

```python
sim2["β"]
```

    array([-0.91495072,  1.04076884,  0.40329737,  0.25581239, -1.75847039,
           -0.7877694 , -2.15842514, -0.57742939, -1.2349024 , -0.54333212])

```python
(
    gg.ggplot(sim2_data, gg.aes(x="initial_read_count", y="final_read_count"))
    + gg.geom_point(gg.aes(color="sgrna"))
    + gg.scale_color_hue()
)
```

![png](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_29_0.png)

    <ggplot: (358823184)>

Simulation 2 model `sim2_model`:

$$
\begin{aligned}
\mu_\beta &\sim N(0, 2) \\
\sigma_\beta &\sim \text{HalfNormal}(2) \\
\beta_g &\sim_g N(\mu_\beta, \sigma_\beta) \\
\eta &= \beta_g[\text{sgRNA}] \\
\mu &= \exp(\eta) \times \text{initial read counts} \\
\alpha &\sim \text{HalfNormal}(0, 5) \\
y &\sim \text{NB}(\mu, \alpha)
\end{aligned}
$$

```python
sgrna_idx = sim2_data.sgrna.factorize(sort=True)[0]

with pm.Model() as sim2_model:
    μ_β = pm.Normal("μ_β", 0, 2)
    σ_β = pm.HalfNormal("σ_β", 2)
    β = pm.Normal("β", μ_β, σ_β, shape=n_guides)
    η = pm.Deterministic("η", β[sgrna_idx])
    μ = pm.Deterministic("μ", pm.math.exp(η) * sim2_data.initial_read_count.values)
    α = pm.HalfNormal("α", 5)
    y = pm.NegativeBinomial("y", μ, α, observed=sim2_data.final_read_count.values)
```

```python
pm.model_to_graphviz(sim2_model)
```

![svg](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_32_0.svg)

```python
with sim2_model:
    sim2_trace = pm.sample(
        tune=2000, random_seed=851, chains=4, return_inferencedata=True
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, β, σ_β, μ_β]

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
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 00:24<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 2_000 tune and 1_000 draw iterations (8_000 + 4_000 draws total) took 43 seconds.

```python
az.plot_trace(sim2_trace, var_names=["β", "α"], filter_vars="like");
```

![png](005_010_simulation-nb-crispr_files/005_010_simulation-nb-crispr_34_0.png)

```python
az.summary(sim2_trace, var_names=["α", "β"], filter_vars="like")
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
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ_β</th>
      <td>-0.561</td>
      <td>0.330</td>
      <td>-1.200</td>
      <td>0.040</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>4527.0</td>
      <td>2988.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[0]</th>
      <td>-0.916</td>
      <td>0.211</td>
      <td>-1.331</td>
      <td>-0.545</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>6177.0</td>
      <td>2939.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[1]</th>
      <td>1.040</td>
      <td>0.196</td>
      <td>0.663</td>
      <td>1.390</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5467.0</td>
      <td>3292.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[2]</th>
      <td>-0.026</td>
      <td>0.200</td>
      <td>-0.387</td>
      <td>0.363</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>5210.0</td>
      <td>2872.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[3]</th>
      <td>0.269</td>
      <td>0.203</td>
      <td>-0.115</td>
      <td>0.633</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5879.0</td>
      <td>2661.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[4]</th>
      <td>-1.373</td>
      <td>0.217</td>
      <td>-1.767</td>
      <td>-0.961</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5681.0</td>
      <td>2717.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[5]</th>
      <td>-0.503</td>
      <td>0.204</td>
      <td>-0.892</td>
      <td>-0.112</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>4898.0</td>
      <td>2592.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[6]</th>
      <td>-2.066</td>
      <td>0.230</td>
      <td>-2.471</td>
      <td>-1.609</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5587.0</td>
      <td>2821.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[7]</th>
      <td>-0.676</td>
      <td>0.215</td>
      <td>-1.098</td>
      <td>-0.291</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5514.0</td>
      <td>2670.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[8]</th>
      <td>-1.129</td>
      <td>0.214</td>
      <td>-1.534</td>
      <td>-0.740</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>6618.0</td>
      <td>3044.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β[9]</th>
      <td>-0.385</td>
      <td>0.200</td>
      <td>-0.739</td>
      <td>-0.006</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5248.0</td>
      <td>3231.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_β</th>
      <td>1.020</td>
      <td>0.278</td>
      <td>0.589</td>
      <td>1.555</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>4321.0</td>
      <td>2987.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>α</th>
      <td>2.417</td>
      <td>0.362</td>
      <td>1.763</td>
      <td>3.081</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>3711.0</td>
      <td>2697.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 1.92 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-09-09

    Python implementation: CPython
    Python version       : 3.9.6
    IPython version      : 7.26.0

    Compiler    : Clang 11.1.0
    OS          : Darwin
    Release     : 20.4.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: JHCookMac.local

    Git branch: nb-model

    theano    : 1.0.5
    matplotlib: 3.4.3
    numpy     : 1.21.2
    seaborn   : 0.11.2
    scipy     : 1.7.1
    pymc3     : 3.11.2
    arviz     : 0.11.2
    plotnine  : 0.8.0
    pandas    : 1.3.2
    re        : 2.2.1
