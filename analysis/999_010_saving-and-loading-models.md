# Saving and loading data




```python
import pickle
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from plotnine.data import diamonds

warnings.simplefilter(action="ignore", category=UserWarning)

RANDOM_SEED = 123
```


```python
diamonds.head()
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
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
with pm.Model() as m1:
    # Priors
    alpha = pm.Normal("alpha", 0, 5)
    beta = pm.Normal("beta", 0, 2)

    # Model
    mu = pm.Deterministic("mu", alpha + beta * diamonds.x)
    sigma = pm.Exponential("sigma", 1)

    # Likelihood
    carat = pm.Normal("carat", mu, sigma, observed=diamonds.carat)
```


```python
pm.model_to_graphviz(m1)
```





![svg](999_010_saving-and-loading-models_files/999_010_saving-and-loading-models_4_0.svg)




> **The following functions have been copied to "pymc3_helpers.py".**


```python
def write_pickle(x, fp):
    """Write `x` to disk as a pickle file."""
    with open(fp, "wb") as f:
        pickle.dump(x, f)
    return None


def get_pickle(fp):
    """Read a pickled file into Python."""
    with open(fp, "rb") as f:
        d = pickle.load(f)
    return d


def pymc3_sampling_procedure(
    model,
    num_mcmc=1000,
    tune=1000,
    chains=2,
    cores=None,
    prior_check_samples=1000,
    ppc_samples=1000,
    random_seed=1234,
    cache_dir=None,
    force=False,
):
    """
    Run the standard PyMC3 sampling procedure.

        Parameters:
            model(pymc3 model): A model from PyMC3
            num_mcmc(int): number of MCMC samples
            tune(int): number of MCMC tuning steps
            chains(int): number of of MCMC chains
            cores(int): number of cores for MCMC
            prior_check_samples(int): number of prior predictive samples to take
            ppc_samples(int): number of posterior predictive samples to take
            random_seed(int): random seed to use in all sampling processes
            cache_dir(Path): the directory to cache the output (leave as `None` to skip caching)
            force(bool): ignore cached results and compute trace and predictive checks
        Returns:
            dict: contains the "trace", "posterior_predictive", and "prior_predictive"
    """
    if cache_dir is not None:
        post_file_path = cache_dir / "posterior-predictive-check.pkl"
        prior_file_path = cache_dir / "prior-predictive-check.pkl"

    if not force and cache_dir is not None and cache_dir.exists():
        print("Loading cached trace and posterior sample...")
        trace = pm.load_trace(cache_dir.as_posix(), model=model)
        post_check = get_pickle(post_file_path)
        prior_check = get_pickle(prior_file_path)
    else:
        with model:
            prior_check = pm.sample_prior_predictive(
                prior_check_samples, random_seed=random_seed
            )
            trace = pm.sample(
                num_mcmc, tune=tune, random_seed=random_seed, chains=chains, cores=cores
            )
            post_check = pm.sample_posterior_predictive(
                trace, samples=ppc_samples, random_seed=random_seed
            )
        if cache_dir is not None:
            print("Caching trace and posterior sample...")
            pm.save_trace(m1_trace, directory=m1_save_dir, overwrite=True)
            write_pickle(post_check, post_file_path)
            write_pickle(prior_check, prior_file_path)
    return {
        "trace": trace,
        "posterior_predictive": post_check,
        "prior_predictive": prior_check,
    }
```


```python
m1_save_dir = Path("pymc3_model_cache/m1")

m1_results = pymc3_sampling_procedure(m1, cache_dir=m1_save_dir)
az_m1 = az.from_pymc3(
    model=m1,
    trace=m1_results["trace"],
    posterior_predictive=m1_results["posterior_predictive"],
    prior=m1_results["prior_predictive"],
)
```


```python
az.plot_trace(az_m1, var_names=["alpha", "beta", "sigma"])
plt.show()
```



![png](999_010_saving-and-loading-models_files/999_010_saving-and-loading-models_8_0.png)




```python

```
