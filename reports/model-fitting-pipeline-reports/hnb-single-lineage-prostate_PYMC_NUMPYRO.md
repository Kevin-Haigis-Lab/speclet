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

    /home/jc604/.conda/envs/speclet_smk/bin/../lib/gcc/x86_64-conda-linux-gnu/10.3.0/../../../../x86_64-conda-linux-gnu/bin/ld: /n/app/gcc/6.2.0/lib64/libquadmath.so.0: undefined reference to `memcpy@GLIBC_2.14'
    collect2: error: ld returned 1 exit status
    /home/jc604/.conda/envs/speclet_smk/bin/../lib/gcc/x86_64-conda-linux-gnu/10.3.0/../../../../x86_64-conda-linux-gnu/bin/ld: /n/app/gcc/6.2.0/lib64/libquadmath.so.0: undefined reference to `memcpy@GLIBC_2.14'
    collect2: error: ld returned 1 exit status
    /home/jc604/.conda/envs/speclet_smk/bin/../lib/gcc/x86_64-conda-linux-gnu/10.3.0/../../../../x86_64-conda-linux-gnu/bin/ld: /n/app/gcc/6.2.0/lib64/libquadmath.so.0: undefined reference to `memcpy@GLIBC_2.14'
    collect2: error: ld returned 1 exit status
    /home/jc604/.conda/envs/speclet_smk/bin/../lib/gcc/x86_64-conda-linux-gnu/10.3.0/../../../../x86_64-conda-linux-gnu/bin/ld: /n/app/gcc/6.2.0/lib64/libquadmath.so.0: undefined reference to `memcpy@GLIBC_2.14'
    collect2: error: ld returned 1 exit status
    /home/jc604/.conda/envs/speclet_smk/bin/../lib/gcc/x86_64-conda-linux-gnu/10.3.0/../../../../x86_64-conda-linux-gnu/bin/ld: /n/app/gcc/6.2.0/lib64/libquadmath.so.0: undefined reference to `memcpy@GLIBC_2.14'
    collect2: error: ld returned 1 exit status
    /home/jc604/.conda/envs/speclet_smk/bin/../lib/gcc/x86_64-conda-linux-gnu/10.3.0/../../../../x86_64-conda-linux-gnu/bin/ld: /n/app/gcc/6.2.0/lib64/libquadmath.so.0: undefined reference to `memcpy@GLIBC_2.14'
    collect2: error: ld returned 1 exit status



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
    alpha                         1.0  1.000852       NaN  1.000852  1.000852
    b                             5.0  3.431088  0.049707  3.350463  3.419943
    cell_line_effect         355310.0  3.374873  0.100861  2.895350  3.346091
    celllines                    10.0  2.216615  1.280596  1.001078  1.001334
    celllines_chol_cov            3.0  1.131455  0.204673  1.007043  1.013343
    celllines_chol_cov_corr       3.0  1.002767  0.001893  1.000582  1.002221
    celllines_chol_cov_stds       2.0  1.193533  0.246280  1.019387  1.106460
    d                         71062.0  1.979854  0.053026  1.760815  1.938581
    delta_celllines              10.0  1.384552  0.394458  1.003063  1.009359
    delta_genes              163071.0  1.037247  0.075813  0.999477  1.005219
    eta                      355310.0  1.001412  0.001243  0.999132  1.000567
    f                             5.0  1.002142  0.001959  1.001078  1.001267
    gene_effect              355310.0  1.957639  0.060971  1.691579  1.910102
    genes                    163071.0  1.311117  0.781904  0.999289  1.004278
    genes_chol_cov               45.0  1.489769  0.580544  1.022045  1.055727
    genes_chol_cov_corr          80.0  1.153677  0.362991  0.999793  1.007440
    genes_chol_cov_stds           9.0  1.120622  0.136113  1.002307  1.009114
    h                         18119.0  1.002777  0.002478  0.999363  1.001323
    k                         18119.0  1.004972  0.003425  0.999596  1.002550
    m                         18119.0  1.124286  0.013556  1.050020  1.117419
    mu                       355310.0  1.001409  0.001246  0.999134  1.000561
    mu_d                      18119.0  3.519737  0.005680  3.466106  3.516814
    mu_f                          1.0  1.013052       NaN  1.013052  1.013052
    mu_k                          1.0  1.001934       NaN  1.001934  1.001934
    mu_mu_d                       1.0  3.523119       NaN  3.523119  3.523119
    sigma_b                       1.0  1.367679       NaN  1.367679  1.367679
    sigma_d                       1.0  1.001193       NaN  1.001193  1.001193
    sigma_f                       1.0  1.019387       NaN  1.019387  1.019387
    sigma_h                       1.0  1.007197       NaN  1.007197  1.007197
    sigma_k                       1.0  1.009114       NaN  1.009114  1.009114
    sigma_m                       1.0  1.327091       NaN  1.327091  1.327091
    sigma_mu_d                    1.0  1.079316       NaN  1.079316  1.079316
    sigma_w                       5.0  1.132575  0.139522  1.002307  1.071629
    w                         90595.0  1.029656  0.029267  0.999289  1.016335

                                  50%       75%       max
    var_name
    alpha                    1.000852  1.000852  1.000852
    b                        3.450769  3.455451  3.478814
    cell_line_effect         3.396596  3.448805  3.482120
    celllines                2.178051  3.443062  3.478814
    celllines_chol_cov       1.019644  1.193662  1.367679
    celllines_chol_cov_corr  1.003860  1.003860  1.003860
    celllines_chol_cov_stds  1.193533  1.280606  1.367679
    d                        1.974917  2.022223  2.159146
    delta_celllines          1.380032  1.752202  1.791756
    delta_genes              1.013179  1.037539  2.111068
    eta                      1.001154  1.001951  1.026365
    f                        1.001305  1.001422  1.005639
    gene_effect              1.955333  2.008073  2.192119
    genes                    1.027000  1.067011  3.544316
    genes_chol_cov           1.143524  1.928936  2.791447
    genes_chol_cov_corr      1.024262  1.092290  2.604948
    genes_chol_cov_stds      1.079316  1.135933  1.367083
    h                        1.002258  1.003554  1.079868
    k                        1.004180  1.006534  1.044252
    m                        1.125774  1.133034  1.237649
    mu                       1.001149  1.001950  1.026365
    mu_d                     3.520234  3.523449  3.544316
    mu_f                     1.013052  1.013052  1.013052
    mu_k                     1.001934  1.001934  1.001934
    mu_mu_d                  3.523119  3.523119  3.523119
    sigma_b                  1.367679  1.367679  1.367679
    sigma_d                  1.001193  1.001193  1.001193
    sigma_f                  1.019387  1.019387  1.019387
    sigma_h                  1.007197  1.007197  1.007197
    sigma_k                  1.009114  1.009114  1.009114
    sigma_m                  1.327091  1.327091  1.327091
    sigma_mu_d               1.079316  1.079316  1.079316
    sigma_w                  1.085924  1.135933  1.367083
    w                        1.026987  1.037228  1.519884
    ============================================================
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.866, 0.802, 0.811, 0.777
    avg. step size: 0.002, 0.003, 0.002, 0.002




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
    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/arviz/stats/stats.py:812: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.7 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(





    Computed from 4000 posterior samples and 355310 observations log-likelihood matrix.

             Estimate       SE
    elpd_loo -2181409.00   644.53
    p_loo    70883.62        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                              Count   Pct.
    (-Inf, 0.5]   (good)     332762   93.7%
     (0.5, 0.7]   (ok)        19262    5.4%
       (0.7, 1]   (bad)        3059    0.9%
       (1, Inf)   (very bad)    227    0.1%




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

    execution time: 216.64 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-08

    Python implementation: CPython
    Python version       : 3.10.4
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.66.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-h-17-51.o2.rc.hms.harvard.edu

    Git branch: simplify

    logging   : 0.5.1.2
    speclet   : 0.0.9000
    matplotlib: 3.5.2
    arviz     : 0.12.1
