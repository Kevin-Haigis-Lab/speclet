# Model Report


```python
import logging
from time import time
from typing import Optional

import arviz as az
import matplotlib.pyplot as plt

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
if FIT_METHOD is ModelFitMethod.PYMC_NUMPYRO or FIT_METHOD is ModelFitMethod.PYMC_MCMC:
    print("R-HAT")
    rhat_summ = summarize_rhat(trace)
    print(rhat_summ)
    print("=" * 60)
    describe_mcmc(trace)
```

    R-HAT


    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/arviz/stats/diagnostics.py:586: RuntimeWarning: invalid value encountered in double_scalars
      (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)




![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_10_2.png)



                                count      mean       std       min       25%  \
    var_name
    alpha                         1.0  1.000909       NaN  1.000909  1.000909
    b                             5.0  1.011968  0.000525  1.011195  1.011698
    cell_line_effect         355310.0  1.010942  0.002217  1.000507  1.010703
    celllines                    10.0  1.006406  0.005879  1.000406  1.000825
    celllines_chol_cov            3.0  1.001364  0.000965  1.000661  1.000814
    celllines_chol_cov_corr       3.0  1.000688  0.000776  0.999792  1.000464
    celllines_chol_cov_stds       2.0  1.001153  0.000695  1.000661  1.000907
    d                         71062.0  1.000872  0.000923  0.999131  1.000206
    delta_celllines              10.0  1.000904  0.000674  1.000243  1.000579
    delta_d                   71062.0  1.001332  0.001235  0.999157  1.000419
    delta_genes              163071.0  1.012180  0.025624  0.999196  1.001550
    eta                      355310.0  1.000882  0.000978  0.999113  1.000170
    f                             5.0  1.000844  0.000382  1.000406  1.000694
    gene_effect              355310.0  1.000882  0.000975  0.999109  1.000174
    genes                    163071.0  1.048427  0.051842  0.999165  1.001355
    genes_chol_cov               45.0  1.240294  0.313199  1.015185  1.084821
    genes_chol_cov_corr          80.0  1.113192  0.248458  0.999898  1.016583
    genes_chol_cov_stds           9.0  1.161959  0.213158  1.002686  1.014534
    h                         18119.0  1.001062  0.001089  0.999165  1.000261
    k                         18119.0  1.000979  0.000934  0.999252  1.000329
    m                         18119.0  1.054216  0.010005  1.013207  1.048698
    mu                       355310.0  1.000849  0.000959  0.999111  1.000151
    mu_d                      18119.0  1.023953  0.006934  1.000715  1.020465
    sigma_b                       1.0  1.000661       NaN  1.000661  1.000661
    sigma_d                       1.0  1.000158       NaN  1.000158  1.000158
    sigma_f                       1.0  1.001644       NaN  1.001644  1.001644
    sigma_h                       1.0  1.006099       NaN  1.006099  1.006099
    sigma_k                       1.0  1.014534       NaN  1.014534  1.014534
    sigma_m                       1.0  1.180998       NaN  1.180998  1.180998
    sigma_mu_d                    1.0  1.024743       NaN  1.024743  1.024743
    sigma_w                       5.0  1.246252  0.256293  1.002686  1.140781
    w                         90595.0  1.071126  0.057157  0.999203  1.040264

                                  50%       75%       max
    var_name
    alpha                    1.000909  1.000909  1.000909
    b                        1.012108  1.012371  1.012468
    cell_line_effect         1.011762  1.012288  1.013058
    celllines                1.006323  1.012005  1.012468
    celllines_chol_cov       1.000967  1.001715  1.002464
    celllines_chol_cov_corr  1.001136  1.001136  1.001136
    celllines_chol_cov_stds  1.001153  1.001398  1.001644
    d                        1.000693  1.001344  1.008786
    delta_celllines          1.000628  1.001041  1.002299
    delta_d                  1.001096  1.001989  1.011374
    delta_genes              1.004240  1.012056  1.527316
    eta                      1.000681  1.001384  1.022192
    f                        1.000815  1.000855  1.001452
    gene_effect              1.000680  1.001382  1.022242
    genes                    1.042619  1.065695  1.438466
    genes_chol_cov           1.126476  1.191798  2.370640
    genes_chol_cov_corr      1.032848  1.101292  2.307072
    genes_chol_cov_stds      1.140781  1.189422  1.680165
    h                        1.000843  1.001622  1.009514
    k                        1.000809  1.001446  1.012093
    m                        1.055074  1.060881  1.087743
    mu                       1.000653  1.001343  1.022192
    mu_d                     1.024435  1.027883  1.142280
    sigma_b                  1.000661  1.000661  1.000661
    sigma_d                  1.000158  1.000158  1.000158
    sigma_f                  1.001644  1.001644  1.001644
    sigma_h                  1.006099  1.006099  1.006099
    sigma_k                  1.014534  1.014534  1.014534
    sigma_m                  1.180998  1.180998  1.180998
    sigma_mu_d               1.024743  1.024743  1.024743
    sigma_w                  1.189422  1.218207  1.680165
    w                        1.062360  1.082061  1.438466
    ============================================================
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.829, 0.916, 0.846, 0.861
    avg. step size: 0.004, 0.006, 0.006, 0.003




![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_10_4.png)



## Model predictions


```python
az.plot_ppc(trace, num_pp_samples=100, random_seed=123)
plt.tight_layout()
plt.show()
```



![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_12_0.png)




```python
psis_loo = az.loo(trace, pointwise=True)
psis_loo
```

    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/arviz/stats/stats.py:812: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.7 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(





    Computed from 4000 posterior samples and 355310 observations log-likelihood matrix.

             Estimate       SE
    elpd_loo -2190104.29   632.59
    p_loo    86951.24        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                              Count   Pct.
    (-Inf, 0.5]   (good)     308048   86.7%
     (0.5, 0.7]   (ok)        40221   11.3%
       (0.7, 1]   (bad)        6512    1.8%
       (1, Inf)   (very bad)    529    0.1%




```python
az.plot_khat(psis_loo)
plt.tight_layout()
plt.show()
```



![png](hnb-single-lineage-prostate_PYMC_NUMPYRO_files/hnb-single-lineage-prostate_PYMC_NUMPYRO_14_0.png)



---


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 191.16 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-12

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.66.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 20
    Architecture: 64bit

    Hostname: compute-f-17-13.o2.rc.hms.harvard.edu

    Git branch: simplify

    logging   : 0.5.1.2
    speclet   : 0.0.9000
    arviz     : 0.12.1
    matplotlib: 3.5.2
