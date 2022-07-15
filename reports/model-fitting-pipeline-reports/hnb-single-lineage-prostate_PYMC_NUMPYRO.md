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
    alpha                         1.0  1.004593       NaN  1.004593  1.004593
    b                             5.0  1.008516  0.000597  1.008024  1.008089
    cell_line_effect         355310.0  1.007333  0.002055  1.000294  1.007357
    celllines                    10.0  1.004588  0.004182  0.999888  1.000710
    celllines_chol_cov            3.0  1.003652  0.004006  0.999587  1.001679
    celllines_chol_cov_corr       3.0  1.000483  0.000288  1.000150  1.000399
    celllines_chol_cov_stds       2.0  1.005241  0.003330  1.002887  1.004064
    delta_celllines              10.0  1.005002  0.002874  1.001147  1.002619
    delta_genes               72476.0  1.001321  0.001237  0.999165  1.000407
    eta                      355310.0  1.001309  0.001234  0.999157  1.000392
    f                             5.0  1.000661  0.000656  0.999888  1.000238
    gene_effect              355310.0  1.001309  0.001233  0.999177  1.000391
    genes                     72476.0  1.001317  0.001214  0.999128  1.000420
    genes_chol_cov               10.0  1.003589  0.003551  0.999881  1.000949
    genes_chol_cov_corr          15.0  1.002484  0.002349  0.999707  1.000628
    genes_chol_cov_stds           4.0  1.004479  0.004552  1.000058  1.001779
    h                         18119.0  1.001297  0.001215  0.999128  1.000395
    k                         18119.0  1.001322  0.001248  0.999157  1.000398
    m                         18119.0  1.001307  0.001149  0.999194  1.000455
    mu                       355310.0  1.001288  0.001222  0.999177  1.000379
    mu_d                      18119.0  1.001341  0.001241  0.999225  1.000425
    sigma_b                       1.0  1.007596       NaN  1.007596  1.007596
    sigma_f                       1.0  1.002887       NaN  1.002887  1.002887
    sigma_h                       1.0  1.000058       NaN  1.000058  1.000058
    sigma_k                       1.0  1.002353       NaN  1.002353  1.002353
    sigma_m                       1.0  1.004867       NaN  1.004867  1.004867
    sigma_mu_d                    1.0  1.010638       NaN  1.010638  1.010638

                                  50%       75%       max
    var_name
    alpha                    1.004593  1.004593  1.004593
    b                        1.008264  1.008747  1.009457
    cell_line_effect         1.008118  1.008300  1.009961
    celllines                1.004802  1.008220  1.009457
    celllines_chol_cov       1.003772  1.005684  1.007596
    celllines_chol_cov_corr  1.000649  1.000649  1.000649
    celllines_chol_cov_stds  1.005241  1.006419  1.007596
    delta_celllines          1.005136  1.007702  1.007829
    delta_genes              1.001089  1.001991  1.011739
    eta                      1.001081  1.001969  1.010131
    f                        1.000620  1.000980  1.001580
    gene_effect              1.001083  1.001972  1.010750
    genes                    1.001091  1.001976  1.011542
    genes_chol_cov           1.002654  1.004770  1.010638
    genes_chol_cov_corr      1.003089  1.003620  1.006868
    genes_chol_cov_stds      1.003610  1.006310  1.010638
    h                        1.001071  1.001970  1.009435
    k                        1.001084  1.001987  1.010079
    m                        1.001101  1.001932  1.009942
    mu                       1.001061  1.001944  1.009942
    mu_d                     1.001107  1.002018  1.011542
    sigma_b                  1.007596  1.007596  1.007596
    sigma_f                  1.002887  1.002887  1.002887
    sigma_h                  1.000058  1.000058  1.000058
    sigma_k                  1.002353  1.002353  1.002353
    sigma_m                  1.004867  1.004867  1.004867
    sigma_mu_d               1.010638  1.010638  1.010638
    ============================================================
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.683, 0.693, 0.692, 0.743
    avg. step size: 0.014, 0.009, 0.01, 0.01




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
    elpd_loo -2225773.67   671.70
    p_loo    20089.92        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                              Count   Pct.
    (-Inf, 0.5]   (good)     354708   99.8%
     (0.5, 0.7]   (ok)          514    0.1%
       (0.7, 1]   (bad)          81    0.0%
       (1, Inf)   (very bad)      7    0.0%




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

    execution time: 128.52 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-15

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

    Hostname: compute-f-17-15.o2.rc.hms.harvard.edu

    Git branch: simplify

    speclet   : 0.0.9000
    arviz     : 0.12.1
    logging   : 0.5.1.2
    matplotlib: 3.5.2
