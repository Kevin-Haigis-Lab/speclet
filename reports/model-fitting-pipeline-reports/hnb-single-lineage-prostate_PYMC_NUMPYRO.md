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
    alpha                         1.0  1.560324       NaN  1.560324  1.560324
    b                             5.0  3.475083  0.122325  3.343639  3.361751
    cell_line_effect         355310.0  3.425141  0.149258  2.726556  3.347240
    celllines                    10.0  2.451137  1.093536  1.010438  1.529344
    celllines_chol_cov            3.0  1.820139  0.533580  1.440497  1.515102
    celllines_chol_cov_corr       3.0  1.400105  0.341227  1.006090  1.301601
    celllines_chol_cov_stds       2.0  1.752220  0.958826  1.074228  1.413224
    d                         71062.0  2.091020  0.204475  1.530413  1.958871
    delta_celllines              10.0  2.066527  0.744262  1.117135  1.525375
    delta_d                   71062.0  1.320908  0.206545  1.059181  1.101916
    delta_genes              163071.0  1.322592  0.197383  1.057522  1.120137
    eta                      355310.0  1.466489  0.149080  1.036991  1.406112
    f                             5.0  1.427192  0.233389  1.010438  1.513412
    gene_effect              355310.0  2.073701  0.186800  1.481990  1.947807
    genes                    163071.0  1.636282  0.609602  1.051746  1.332739
    genes_chol_cov               45.0  1.691751  0.241988  1.170279  1.577284
    genes_chol_cov_corr          80.0  1.392052  0.312386  1.000026  1.151529
    genes_chol_cov_stds           9.0  1.667333  0.153567  1.538240  1.549928
    h                         18119.0  1.363102  0.188472  1.059125  1.152788
    k                         18119.0  1.497489  0.118286  1.065762  1.528973
    m                         18119.0  1.387773  0.167763  1.057758  1.230400
    mu                       355310.0  1.466496  0.149094  1.036766  1.406088
    mu_d                      18119.0  3.288503  0.217074  2.525423  3.073879
    mu_f                          1.0  1.621535       NaN  1.621535  1.621535
    mu_k                          1.0  1.553201       NaN  1.553201  1.553201
    mu_mu_d                       1.0  3.163717       NaN  3.163717  3.163717
    sigma_b                       1.0  2.430212       NaN  2.430212  2.430212
    sigma_d                       1.0  1.534588       NaN  1.534588  1.534588
    sigma_f                       1.0  1.074228       NaN  1.074228  1.074228
    sigma_h                       1.0  1.538240       NaN  1.538240  1.538240
    sigma_k                       1.0  1.549928       NaN  1.549928  1.549928
    sigma_m                       1.0  1.962637       NaN  1.962637  1.962637
    sigma_mu_d                    1.0  1.548896       NaN  1.548896  1.548896
    sigma_w                       5.0  1.681259  0.118334  1.561429  1.608707
    w                         90595.0  1.437934  0.165861  1.051746  1.319340

                                  50%       75%       max
    var_name
    alpha                    1.560324  1.560324  1.560324
    b                        3.479364  3.575396  3.615264
    cell_line_effect         3.407792  3.569152  3.616574
    celllines                2.448096  3.449960  3.615264
    celllines_chol_cov       1.589707  2.009960  2.430212
    celllines_chol_cov_corr  1.597112  1.597112  1.597112
    celllines_chol_cov_stds  1.752220  2.091216  2.430212
    d                        2.123374  2.229059  2.709673
    delta_celllines          1.930721  2.641827  3.108438
    delta_d                  1.299575  1.528723  1.595028
    delta_genes              1.283091  1.527922  2.056981
    eta                      1.534766  1.569347  1.642821
    f                        1.528911  1.530644  1.552553
    gene_effect              2.114800  2.201485  2.707175
    genes                    1.529342  1.577996  3.768629
    genes_chol_cov           1.632462  1.781627  2.394121
    genes_chol_cov_corr      1.320612  1.539833  2.210169
    genes_chol_cov_stds      1.608707  1.794521  1.962637
    h                        1.526785  1.529203  1.594223
    k                        1.538153  1.555029  1.632844
    m                        1.406130  1.546926  1.769049
    mu                       1.534769  1.569367  1.642821
    mu_d                     3.302294  3.470737  3.768629
    mu_f                     1.621535  1.621535  1.621535
    mu_k                     1.553201  1.553201  1.553201
    mu_mu_d                  3.163717  3.163717  3.163717
    sigma_b                  2.430212  2.430212  2.430212
    sigma_d                  1.534588  1.534588  1.534588
    sigma_f                  1.074228  1.074228  1.074228
    sigma_h                  1.538240  1.538240  1.538240
    sigma_k                  1.549928  1.549928  1.549928
    sigma_m                  1.962637  1.962637  1.962637
    sigma_mu_d               1.548896  1.548896  1.548896
    sigma_w                  1.619750  1.794521  1.821890
    w                        1.503485  1.568866  2.015926
    ============================================================
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 1.259, 0.908, 0.811, 0.912
    avg. step size: 0.0, 0.005, 0.003, 0.002




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

    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/arviz/stats/stats.py:1048: RuntimeWarning: overflow encountered in exp
      weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/numpy/core/_methods.py:48: RuntimeWarning: overflow encountered in reduce
      return umr_sum(a, axis, dtype, out, keepdims, initial, where)
    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/arviz/stats/stats.py:812: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.7 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(





    Computed from 4000 posterior samples and 355310 observations log-likelihood matrix.

             Estimate       SE
    elpd_loo -2327826.31   702.73
    p_loo    172310.89        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                              Count   Pct.
    (-Inf, 0.5]   (good)     175585   49.4%
     (0.5, 0.7]   (ok)        45360   12.8%
       (0.7, 1]   (bad)       39788   11.2%
       (1, Inf)   (very bad)  94577   26.6%




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

    execution time: 291.91 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-11

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.71.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-163.o2.rc.hms.harvard.edu

    Git branch: simplify

    logging   : 0.5.1.2
    speclet   : 0.0.9000
    arviz     : 0.12.1
    matplotlib: 3.5.2
