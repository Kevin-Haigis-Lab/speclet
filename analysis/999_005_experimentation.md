# An ephemeral notebook for experimentation


```python
import pandas as pd
import numpy as np
import plotnine as gg
import pymc3 as pm
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import string
from itertools import product
from numpy.random import normal, exponential
```


```python
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
```


```python
gg.theme_set(gg.theme_minimal())
```


```python
RANDOM_SEED = 103
```

### Experimental design

I have 10 different cell lines and am testing 3 genes (the same 3 genes in each cell line).
For each combination of cell line and gene, I knock-out (i.e. remove) the gene in the cell line and then measure the growth rate of the cell lines.
I run the experiment 3 times for each combination.
The output is a log-fold-change (logFC) of the number of cells at the beginning of the experiment compared to the number at the end.

### The model

I want to model the logFC depending on the gene and cell line.
I think this means I need 2 varying intercepts, one for the gene and one for the cell line.

$
\log(FC) \sim \mathcal{N}(\mu_{g,c} \sigma) \\
\mu_{g,c} = \alpha_g + \beta_c \\
\quad \alpha_g \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha) \\
\qquad \mu_\alpha = \mathcal{N}(0,5) \quad \sigma_\alpha \sim \text{Exp}(1) \\
\quad \beta_c \sim \mathcal{N}(\mu_\beta, \sigma_\beta) \\
\qquad \mu_\beta = \mathcal{N}(0,5) \quad \sigma_\beta \sim \text{Exp}(1) \\
\sigma \sim \text{Exp}(1)
$


```python
np.random.seed(RANDOM_SEED)

# Data parameters.
num_cell_lines = 10
num_genes = 3
n_experiments = 3

# Model parameters.
real_params = {
    "mu_alpha": -1,
    "sigma_alpha": 0.5,
    "mu_beta": 0,
    "sigma_beta": 0.5,
    "sigma": 0.05,
}

real_params["alpha_g"] = normal(
    real_params["mu_alpha"], real_params["sigma_alpha"], num_genes
)
real_params["beta_c"] = normal(
    real_params["mu_beta"], real_params["sigma_beta"], num_cell_lines
)

data = pd.DataFrame(
    product(range(num_cell_lines), range(num_genes), range(n_experiments)),
    columns=["cell_line", "gene", "expt"],
)

data["logfc"] = np.nan
for i in range(len(data)):
    c = data.loc[i, "cell_line"]
    g = data.loc[i, "gene"]
    mu_gc = real_params["alpha_g"][g] + real_params["beta_c"][c]
    data.loc[i, "logfc"] = normal(mu_gc, real_params["sigma"], 1)
```


```python
print(data.head(10).to_markdown())
```

    |    |   cell_line |   gene |   expt |     logfc |
    |---:|------------:|-------:|-------:|----------:|
    |  0 |           0 |      0 |      0 | -1.73762  |
    |  1 |           0 |      0 |      1 | -1.87747  |
    |  2 |           0 |      0 |      2 | -1.88619  |
    |  3 |           0 |      1 |      0 | -1.27018  |
    |  4 |           0 |      1 |      1 | -1.32484  |
    |  5 |           0 |      1 |      2 | -1.28888  |
    |  6 |           0 |      2 |      0 | -0.934375 |
    |  7 |           0 |      2 |      1 | -0.936662 |
    |  8 |           0 |      2 |      2 | -1.08875  |
    |  9 |           1 |      0 |      0 | -2.13649  |



```python
(
    gg.ggplot(data, gg.aes(x="factor(gene)", y="logfc", color="factor(gene)"))
    + gg.geom_boxplot(outlier_color="")
    + gg.geom_jitter(width=0.2, height=0, size=1, alpha=0.7)
    + gg.scale_color_discrete(guide=False)
    + gg.labs(x="gene", y="logFC", title="Synthetic data")
)
```


    
![png](999_005_experimentation_files/999_005_experimentation_8_0.png)
    





    <ggplot: (8786276810051)>




```python
(
    gg.ggplot(data, gg.aes(x="factor(cell_line)", y="logfc"))
    + gg.geom_boxplot(outlier_color="")
    + gg.geom_jitter(
        gg.aes(color="factor(gene)"), width=0.2, height=0, size=1, alpha=0.7
    )
    + gg.scale_color_discrete()
    + gg.theme(figure_size=(8, 5))
    + gg.labs(x="cell line", y="logFC", title="Synthetic data", color="gene")
)
```


    
![png](999_005_experimentation_files/999_005_experimentation_9_0.png)
    





    <ggplot: (8786279778687)>




```python
cell_line_idx = data["cell_line"].values
gene_idx = data["gene"].values


with pm.Model() as model:
    # Hyper-priors
    mu_alpha = pm.Normal("mu_alpha", -1, 2)
    sigma_alpha = pm.Exponential("sigma_alpha", 0.5)
    mu_beta = pm.Normal("mu_beta", 0, 2)
    sigma_beta = pm.Exponential("sigma_beta", 0.5)

    # Priors
    alpha_g = pm.Normal("alpha_g", mu_alpha, sigma_alpha, shape=num_genes)
    beta_c = pm.Normal("beta_c", mu_beta, sigma_beta, shape=num_cell_lines)

    # Likelihood
    mu_gc = pm.Deterministic("mu_gc", alpha_g[gene_idx] + beta_c[cell_line_idx])
    sigma = pm.Exponential("sigma", 1)
    logfc = pm.Normal("logfc", mu_gc, sigma, observed=data["logfc"].values)

    # Sampling
    model_prior_check = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    model_trace = pm.sample(2000, tune=1000, random_seed=RANDOM_SEED)
    model_post_check = pm.sample_posterior_predictive(
        model_trace, random_seed=RANDOM_SEED
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, beta_c, alpha_g, sigma_beta, mu_beta, sigma_alpha, mu_alpha]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 03:54<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 235 seconds.
    The chain reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
    The chain reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
    The chain reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
    The chain reached the maximum tree depth. Increase max_treedepth, increase target_accept or reparameterize.
    The number of effective samples is smaller than 25% for some parameters.




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:08<00:00]
</div>




```python
az_model = az.from_pymc3(
    trace=model_trace,
    model=model,
    prior=model_prior_check,
    posterior_predictive=model_post_check,
)
```


```python
az.plot_trace(az_model, var_names=["alpha_g", "beta_c"])
plt.show()
```


    
![png](999_005_experimentation_files/999_005_experimentation_12_0.png)
    



```python
az.plot_forest(az_model, var_names=["alpha_g", "beta_c"], combined=True)
plt.show()
```


    
![png](999_005_experimentation_files/999_005_experimentation_13_0.png)
    



```python
var_names = ["mu_alpha", "sigma_alpha", "mu_beta", "sigma_beta"]
az.summary(az_model, var_names=var_names).assign(
    real_value=[np.mean(real_params[v]) for v in var_names]
)
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
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>real_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_alpha</th>
      <td>-1.062</td>
      <td>1.454</td>
      <td>-3.775</td>
      <td>1.739</td>
      <td>0.044</td>
      <td>0.031</td>
      <td>1094.0</td>
      <td>1094.0</td>
      <td>1095.0</td>
      <td>1291.0</td>
      <td>1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>sigma_alpha</th>
      <td>0.834</td>
      <td>0.680</td>
      <td>0.175</td>
      <td>1.980</td>
      <td>0.017</td>
      <td>0.012</td>
      <td>1651.0</td>
      <td>1651.0</td>
      <td>2433.0</td>
      <td>2209.0</td>
      <td>1.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>mu_beta</th>
      <td>-0.038</td>
      <td>1.471</td>
      <td>-2.742</td>
      <td>2.845</td>
      <td>0.047</td>
      <td>0.035</td>
      <td>973.0</td>
      <td>895.0</td>
      <td>977.0</td>
      <td>1050.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sigma_beta</th>
      <td>0.588</td>
      <td>0.170</td>
      <td>0.326</td>
      <td>0.899</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>2731.0</td>
      <td>2496.0</td>
      <td>3330.0</td>
      <td>2580.0</td>
      <td>1.0</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_posterior(az_model, var_names=var_names)
plt.show()
```


    
![png](999_005_experimentation_files/999_005_experimentation_15_0.png)
    



```python
mu_post = model_trace.get_values("mu_gc")
mu_post_mean = mu_post.mean(axis=0)
mu_post_hdi = [az.hpd(x) for x in mu_post.T]

post_data = data.copy()
post_data["mu_mean"] = mu_post_mean
post_data["mu_lower_ci"] = [x[0] for x in mu_post_hdi]
post_data["mu_upper_ci"] = [x[1] for x in mu_post_hdi]
post_data["row_i"] = list(range(len(post_data)))
(
    gg.ggplot(post_data, gg.aes(x="row_i"))
    + gg.geom_point(
        gg.aes(y="logfc", color="factor(cell_line)", shape="factor(gene)"), size=2
    )
    + gg.geom_point(gg.aes(y="mu_mean"), alpha=0.5, shape=".")
    + gg.geom_linerange(
        gg.aes(ymin="mu_lower_ci", ymax="mu_upper_ci"), alpha=0.5, size=1
    )
    + gg.theme(figure_size=(10, 5))
    + gg.labs(
        x="experiment number",
        y="logFC",
        color="cell line",
        shape="gene",
        title="Posterior predictions",
    )
)
```


    
![png](999_005_experimentation_files/999_005_experimentation_16_0.png)
    





    <ggplot: (8786277693555)>




```python
col_names = ["gene_" + str(i) for i in range(num_genes)]
col_names += ["cell_" + str(i) for i in range(num_cell_lines)]

alpha_g_post = model_trace.get_values("alpha_g").mean(axis=1)
beta_a_post = model_trace.get_values("beta_c").mean(axis=1)

d = pd.DataFrame({"alpha_g": alpha_g_post, "beta_a": beta_a_post})
(
    gg.ggplot(d, gg.aes("alpha_g", "beta_a"))
    + gg.geom_hline(yintercept=0, linetype="--")
    + gg.geom_vline(xintercept=0, linetype="--")
    + gg.geom_point(size=0.2, alpha=0.1, color="darkred")
    + gg.labs(
        x="average alpha_g",
        y="average beta_a",
        title="Posterior of model varying intercepts",
    )
)
```


    
![png](999_005_experimentation_files/999_005_experimentation_17_0.png)
    





    <ggplot: (8786277693648)>




```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    plotnine 0.7.1
    pymc3    3.9.3
    pandas   1.1.3
    seaborn  0.11.0
    numpy    1.19.2
    arviz    0.10.0
    last updated: 2020-10-26 
    
    CPython 3.8.5
    IPython 7.18.1
    
    compiler   : GCC 7.3.0
    system     : Linux
    release    : 3.10.0-1062.el7.x86_64
    machine    : x86_64
    processor  : x86_64
    CPU cores  : 28
    interpreter: 64bit
    host name  : compute-e-16-237.o2.rc.hms.harvard.edu
    Git branch : models



```python

```
