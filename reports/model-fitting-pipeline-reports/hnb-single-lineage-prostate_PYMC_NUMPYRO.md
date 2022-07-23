# Model Report


```python
import logging
from itertools import product
from time import time
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from xarray import Dataset

from speclet import model_configuration
from speclet.analysis.arviz_analysis import describe_mcmc, summarize_rhat
from speclet.bayesian_models import get_bayesian_model
from speclet.io import project_root
from speclet.loggers import set_console_handler_level
from speclet.managers.cache_manager import (
    get_cached_posterior,
    get_posterior_cache_name,
)
from speclet.project_configuration import get_bayesian_modeling_constants
from speclet.project_enums import ModelFitMethod
```

    WARNING (aesara.tensor.blas): Using NumPy C-API based implementation for BLAS functions.



```python
notebook_tic = time()
set_console_handler_level(logging.WARNING)
%config InlineBackend.figure_format = "retina"
HDI_PROB = get_bayesian_modeling_constants().hdi_prob
```

Parameters for papermill:

- `MODEL_NAME`: name of the model
- `FIT_METHOD`: method used to fit the model; either "ADVI" or "MCMC"
- `CONFIG_PATH`: path to configuration file
- `ROOT_CACHE_DIR`: path to the root caching directory

## Setup

### Papermill parameters


```python
CONFIG_PATH = ""
MODEL_NAME = ""
FIT_METHOD_STR = ""
ROOT_CACHE_DIR = ""
```


```python
# Parameters
MODEL_NAME = "hnb-single-lineage-prostate"
FIT_METHOD_STR = "PYMC_NUMPYRO"
CONFIG_PATH = "models/model-configs.yaml"
ROOT_CACHE_DIR = "models"
```


```python
FIT_METHOD = ModelFitMethod(FIT_METHOD_STR)
model_config = model_configuration.get_configuration_for_model(
    config_path=project_root() / CONFIG_PATH, name=MODEL_NAME
)
model = get_bayesian_model(model_config.model)(**model_config.model_kwargs)
trace = get_cached_posterior(
    get_posterior_cache_name(MODEL_NAME, FIT_METHOD),
    cache_dir=project_root() / ROOT_CACHE_DIR,
)
```

## Fit diagnostics


```python
if FIT_METHOD in {ModelFitMethod.PYMC_NUMPYRO, ModelFitMethod.PYMC_MCMC}:
    print("R-HAT")
    rhat_summ = summarize_rhat(trace)
    print(rhat_summ)
    print("=" * 60)
    describe_mcmc(trace)
```

    R-HAT




![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_10_1.png)



                   count      mean       std       min       25%       50%  \
    var_name
    a            71062.0  1.000988  0.000999  0.999114  1.000258  1.000786
    alpha            1.0  1.002090       NaN  1.002090  1.002090  1.002090
    b            18119.0  1.000983  0.000997  0.999221  1.000253  1.000780
    delta_a      71062.0  1.000996  0.001013  0.999130  1.000257  1.000788
    delta_b      18119.0  1.000997  0.001007  0.999138  1.000261  1.000791
    delta_mu_a   18119.0  1.000992  0.001010  0.999124  1.000259  1.000785
    eta         355308.0  1.000986  0.000995  0.999099  1.000257  1.000787
    mu_a         18119.0  1.000974  0.000988  0.999120  1.000255  1.000775
    mu_b             1.0  1.000122       NaN  1.000122  1.000122  1.000122
    mu_mu_a          1.0  1.000791       NaN  1.000791  1.000791  1.000791
    sigma_a          1.0  1.000625       NaN  1.000625  1.000625  1.000625
    sigma_b          1.0  1.000116       NaN  1.000116  1.000116  1.000116
    sigma_mu_a       1.0  1.000942       NaN  1.000942  1.000942  1.000942

                     75%       max
    var_name
    a           1.001514  1.009380
    alpha       1.002090  1.002090
    b           1.001509  1.008099
    delta_a     1.001525  1.009188
    delta_b     1.001519  1.007595
    delta_mu_a  1.001517  1.007949
    eta         1.001510  1.009903
    mu_a        1.001494  1.007848
    mu_b        1.000122  1.000122
    mu_mu_a     1.000791  1.000791
    sigma_a     1.000625  1.000625
    sigma_b     1.000116  1.000116
    sigma_mu_a  1.000942  1.000942
    ============================================================
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.892, 0.906, 0.931, 1.024
    avg. step size: 0.03, 0.029, 0.027, 0.028




![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_10_3.png)



## Model predictions


```python
np.random.seed(333)

pp: Dataset = trace.posterior_predictive["ct_final"]
n_chains, n_draws, n_data = pp.shape
n_rand = 10
draws_idx = np.random.choice(np.arange(n_draws), n_rand, replace=False)

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(8, 10), squeeze=True, sharex=False, sharey=False
)

alpha = 0.2

for c, d in product(range(n_chains), draws_idx):
    draw = pp[c, d, :].values.flatten()
    sns.kdeplot(x=draw, ax=axes[0], color="tab:blue", alpha=alpha)
    sns.kdeplot(x=np.log10(draw + 1), ax=axes[1], color="tab:blue", alpha=alpha)

avg_ppc = pp.median(axis=(0, 1))
sns.kdeplot(x=avg_ppc, ax=axes[0], color="tab:orange", alpha=0.8)
sns.kdeplot(x=np.log10(avg_ppc + 1), ax=axes[1], color="tab:orange", alpha=0.8)

obs_data = trace.observed_data["ct_final"].values.flatten()
sns.kdeplot(x=obs_data, ax=axes[0], color="black", alpha=0.8)
sns.kdeplot(x=np.log10(obs_data + 1), ax=axes[1], color="black", alpha=0.8)

axes[0].set_xlabel("ct_final")
axes[1].set_xlabel(r"$\log_{10}($ ct_final $)$")

leg_handles = [
    Line2D([0], [0], color="tab:blue", label="draw"),
    Line2D([0], [0], color="tab:orange", label="post. pred. median"),
    Line2D([0], [0], color="black", label="observed"),
]
for ax in axes:
    ax.legend(handles=leg_handles, loc="best")

plt.tight_layout()
plt.show()
```



![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_12_0.png)




```python
has_log_likelihood = "log_likelihood" in trace
```


```python
if has_log_likelihood:
    psis_loo = az.loo(trace, pointwise=True)
    psis_loo
```


```python
if has_log_likelihood:
    az.plot_khat(psis_loo)
    plt.tight_layout()
    plt.show()
```

---


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 29.74 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-23

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.71.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-182.o2.rc.hms.harvard.edu

    Git branch: simplify

    arviz     : 0.12.1
    numpy     : 1.23.0
    logging   : 0.5.1.2
    speclet   : 0.0.9000
    matplotlib: 3.5.2
    seaborn   : 0.11.2
