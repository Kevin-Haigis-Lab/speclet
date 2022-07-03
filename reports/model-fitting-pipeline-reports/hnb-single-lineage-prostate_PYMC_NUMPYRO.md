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
    a                         71062.0  1.658477  0.296690  1.101482  1.408767
    alpha                         1.0  3.734635       NaN  3.734635  3.734635
    b                             5.0  2.034869  0.293876  1.692438  1.908709
    cell_line_effect         355310.0  2.027597  0.265454  1.205065  1.803394
    celllines                    10.0  2.136156  0.286387  1.692438  1.941787
    celllines_chol_cov            3.0  2.679414  0.377736  2.304251  2.489285
    celllines_chol_cov_corr       3.0  1.944978  0.804123  1.016457  1.712848
    celllines_chol_cov_stds       2.0  2.530902  0.320533  2.304251  2.417576
    d                         18119.0  2.403557  0.181585  1.838556  2.276834
    delta_a                   71062.0  1.680771  0.320876  1.096626  1.401002
    delta_celllines              10.0  2.254722  0.493448  1.469257  1.846107
    delta_genes              163071.0  1.732417  0.348737  1.095033  1.454711
    eta                      355310.0  1.968717  0.382324  1.104309  1.675255
    f                             5.0  2.237442  0.269313  1.981654  2.029066
    gene_effect              355310.0  2.155949  0.343080  1.114514  1.937561
    genes                    163071.0  2.398710  0.411222  1.102900  2.129638
    genes_chol_cov               45.0  2.921592  0.545492  1.663809  2.607665
    genes_chol_cov_corr          80.0  1.671219  0.479851  1.005266  1.342686
    genes_chol_cov_stds           9.0  3.067284  0.489266  2.411860  2.700749
    h                         18119.0  2.409407  0.309362  1.557847  2.207861
    k                         18119.0  2.068060  0.241479  1.203565  1.916327
    m                         18119.0  2.623612  0.386225  1.417761  2.328928
    mu                       355310.0  1.972540  0.392236  1.105198  1.675225
    mu_f                          1.0  3.129007       NaN  3.129007  3.129007
    mu_k                          1.0  3.317361       NaN  3.317361  3.317361
    sigma_a                       1.0  1.617798       NaN  1.617798  1.617798
    sigma_b                       1.0  2.304251       NaN  2.304251  2.304251
    sigma_d                       1.0  3.822674       NaN  3.822674  3.822674
    sigma_f                       1.0  2.757553       NaN  2.757553  2.757553
    sigma_h                       1.0  2.807578       NaN  2.807578  2.807578
    sigma_k                       1.0  2.700749       NaN  2.700749  2.700749
    sigma_m                       1.0  2.632065       NaN  2.632065  2.632065
    sigma_w                       5.0  3.128499  0.483345  2.411860  2.965946
    w                         90595.0  2.416751  0.452574  1.102900  2.115942
    z                             1.0  1.913600       NaN  1.913600  1.913600

                                  50%       75%       max
    var_name
    a                        1.653733  1.852308  2.891274
    alpha                    3.734635  3.734635  3.734635
    b                        1.928498  2.188800  2.455902
    cell_line_effect         2.045344  2.193372  2.697361
    celllines                2.108933  2.274765  2.659641
    celllines_chol_cov       2.674320  2.866996  3.059672
    celllines_chol_cov_corr  2.409238  2.409238  2.409238
    celllines_chol_cov_stds  2.530902  2.644228  2.757553
    d                        2.374110  2.493592  3.698062
    delta_a                  1.669156  1.900935  3.329791
    delta_celllines          2.422965  2.575580  2.811898
    delta_genes              1.705234  1.950814  4.017006
    eta                      1.938104  2.239382  3.900014
    f                        2.225746  2.291104  2.659641
    gene_effect              2.205606  2.366543  4.005232
    genes                    2.402362  2.682117  4.083735
    genes_chol_cov           2.779783  3.184812  4.000566
    genes_chol_cov_corr      1.633635  1.816507  3.296919
    genes_chol_cov_stds      2.965946  3.541334  3.822674
    h                        2.495148  2.630477  3.190408
    k                        2.056811  2.198632  3.498346
    m                        2.576131  3.005002  3.687576
    mu                       1.935122  2.235681  3.900014
    mu_f                     3.129007  3.129007  3.129007
    mu_k                     3.317361  3.317361  3.317361
    sigma_a                  1.617798  1.617798  1.617798
    sigma_b                  2.304251  2.304251  2.304251
    sigma_d                  3.822674  3.822674  3.822674
    sigma_f                  2.757553  2.757553  2.757553
    sigma_h                  2.807578  2.807578  2.807578
    sigma_k                  2.700749  2.700749  2.700749
    sigma_m                  2.632065  2.632065  2.632065
    sigma_w                  3.121556  3.541334  3.601798
    w                        2.454557  2.749580  4.083735
    z                        1.913600  1.913600  1.913600
    ============================================================
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.344, 1.778, 0.735, 0.04
    avg. step size: 0.001, 0.0, 0.0, 0.0




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
    elpd_loo -2297042.42  2279.99
    p_loo    64208.39        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                              Count   Pct.
    (-Inf, 0.5]   (good)     280145   78.8%
     (0.5, 0.7]   (ok)        14893    4.2%
       (0.7, 1]   (bad)       15610    4.4%
       (1, Inf)   (very bad)  44662   12.6%




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

    execution time: 278.62 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-06-29

    Python implementation: CPython
    Python version       : 3.10.4
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.45.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-178.o2.rc.hms.harvard.edu

    Git branch: per-lineage

    logging   : 0.5.1.2
    speclet   : 0.0.9000
    arviz     : 0.12.1
    matplotlib: 3.5.2
