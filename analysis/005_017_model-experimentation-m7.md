```python
import pandas as pd
import numpy as np
import plotnine as gg
import pymc3 as pm
from theano import tensor as tt
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import string
from itertools import product
from numpy.random import normal, exponential, seed
import warnings

# Remove annoying filters from some dated ArViz functions.
warnings.simplefilter(action="ignore", category=UserWarning)

# Default theme for Plotnine.
gg.theme_set(gg.theme_minimal())

# A value to use in all random seed setting instances.
RANDOM_SEED = 103
```

---

## Model 7. Generation using varying sgRNA|gene and cell line effects

Model the effects of knocking out gene $g$ with sgRNA $s$ in cell line $c$.
The data will be generated with the following model, but then different models structures will be tested.

$
logFC_{s,c} \sim \mathcal{N}(\mu_{s,c}, \sigma) \\
\mu_{s,c} = \alpha_s + \beta_c \\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha_s}, \sigma_\alpha) \\
\qquad \mu_{\alpha_s} = \gamma_g \\
\qquad\quad \gamma_g \sim \mathcal{N}(\mu_\gamma, \sigma_\gamma) \\
\qquad\qquad \mu_\gamma \sim \mathcal{N}(0, 5) \quad \sigma_\gamma \sim \text{Exp}(1) \\
\qquad \sigma_\alpha \sim \text{Exp}(1) \\
\quad \beta_c \sim \mathcal{N}(\mu_\beta, \sigma_\beta) \\
\qquad \mu_\beta \sim \mathcal{N}(0, 5) \quad \sigma_\beta \sim \text{Exp}(1) \\
\sigma \sim \text{Exp}(1)
$

Simulated values:

- number of cell lines: 20
- number of genes: 10
- number of repeated measures: $[1 , 2 , \dots , 10]$
- $\mu_\gamma = -1.0$, $\sigma_\gamma = 0.5$
- $\sigma_\alpha = 0.2$
- $\mu_\beta = 0$, $\sigma_\beta = 1$
- $\sigma = 0.3$


```python
def prefixed_count(prefix, n, plus=0):
    """Make an array of 1-n with the number and some prefix."""
    return [prefix + str(i + plus) for i in range(n)]
```


```python
def make_cat(df, col, ordered=True):
    """Make a column of a data frame into a categorical data type."""
    vals = df[col].drop_duplicates().to_list()
    df[col] = pd.Categorical(df[col], categories=vals, ordered=ordered)
    return df
```


```python
seed(RANDOM_SEED)

num_cell_lines = 20
num_genes = 10
num_sgrna_per_gene = list(range(1, num_genes + 1))
num_sgrnas = sum(num_sgrna_per_gene)

cell_lines = prefixed_count("cell_", num_cell_lines)
genes = prefixed_count("gene_", num_genes)
sgrnas = prefixed_count("sgRNA_", num_sgrnas)

# RP ("real parameters")
RP = {
    "mu_gamma": -1.0,
    "sigma_gamma": 0.5,
    "sigma_alpha": 0.2,
    "mu_beta": 0.0,
    "sigma_beta": 1.0,
    "sigma": 0.3,
}

RP["gamma_g"] = normal(loc=RP["mu_gamma"], scale=RP["sigma_gamma"], size=num_genes)
RP["beta_c"] = normal(loc=RP["mu_beta"], scale=RP["sigma_beta"], size=num_cell_lines)

sgrna_df = pd.DataFrame({"gene": np.repeat(genes, num_sgrna_per_gene), "sgRNA": sgrnas})
for col in sgrna_df.columns:
    sgrna_df = make_cat(sgrna_df, col)

alpha_s = []
for gene_i in sgrna_df["gene"].cat.codes:
    alpha_s.append(normal(loc=RP["gamma_g"][gene_i], scale=RP["sigma_alpha"]))


RP["alpha_s"] = alpha_s

data = pd.DataFrame(product(cell_lines, sgrnas), columns=["cell_line", "sgRNA"])
data = data.merge(sgrna_df, on="sgRNA")


for col in data.columns:
    data = make_cat(data, col)


for i in range(len(data)):
    cell_i = data["cell_line"].cat.codes[i]
    sgrna_i = data["sgRNA"].cat.codes[i]
    mu_sc = RP["alpha_s"][sgrna_i] + RP["beta_c"][cell_i]
    data.loc[i, "mu_sc"] = mu_sc
    data.loc[i, "log_fc"] = normal(loc=mu_sc, scale=RP["sigma"])

data
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
      <th>cell_line</th>
      <th>sgRNA</th>
      <th>gene</th>
      <th>mu_sc</th>
      <th>log_fc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cell_0</td>
      <td>sgRNA_0</td>
      <td>gene_0</td>
      <td>-1.284686</td>
      <td>-0.974021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cell_1</td>
      <td>sgRNA_0</td>
      <td>gene_0</td>
      <td>-1.986886</td>
      <td>-2.358464</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cell_2</td>
      <td>sgRNA_0</td>
      <td>gene_0</td>
      <td>-0.883180</td>
      <td>-0.911884</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cell_3</td>
      <td>sgRNA_0</td>
      <td>gene_0</td>
      <td>-0.267206</td>
      <td>-0.774813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cell_4</td>
      <td>sgRNA_0</td>
      <td>gene_0</td>
      <td>-3.064072</td>
      <td>-2.866197</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>cell_15</td>
      <td>sgRNA_54</td>
      <td>gene_9</td>
      <td>-2.116856</td>
      <td>-2.188318</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>cell_16</td>
      <td>sgRNA_54</td>
      <td>gene_9</td>
      <td>-2.245777</td>
      <td>-1.630069</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>cell_17</td>
      <td>sgRNA_54</td>
      <td>gene_9</td>
      <td>-1.079010</td>
      <td>-0.752161</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>cell_18</td>
      <td>sgRNA_54</td>
      <td>gene_9</td>
      <td>-1.900383</td>
      <td>-1.697479</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>cell_19</td>
      <td>sgRNA_54</td>
      <td>gene_9</td>
      <td>-1.452215</td>
      <td>-1.609552</td>
    </tr>
  </tbody>
</table>
<p>1100 rows × 5 columns</p>
</div>




```python
real_gene_vals = pd.DataFrame({"gene": genes, "log_fc": RP["gamma_g"]})

(
    gg.ggplot(data, gg.aes(x="gene", y="log_fc"))
    + gg.geom_jitter(gg.aes(color="cell_line"), height=0, width=0.3, alpha=0.7)
    + gg.geom_crossbar(gg.aes(ymin="log_fc", ymax="log_fc"), data=real_gene_vals)
    + gg.scale_color_discrete(guide=gg.guide_legend(title="cell line", ncol=2))
    + gg.theme(axis_text_x=gg.element_text(angle=30, hjust=1, vjust=0.2))
    + gg.labs(
        x="gene",
        y="logFC",
        title="Synthetic data by gene annotated with real gene effect",
    )
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_6_0.png)
    





    <ggplot: (8760584917471)>




```python
real_cellline_vals = pd.DataFrame({"cell_line": cell_lines, "log_fc": RP["beta_c"]})

(
    gg.ggplot(data, gg.aes(x="cell_line", y="log_fc"))
    + gg.geom_jitter(gg.aes(color="gene"), height=0, width=0.3, alpha=0.7, size=1)
    + gg.geom_crossbar(gg.aes(ymin="log_fc", ymax="log_fc"), data=real_cellline_vals)
    + gg.scale_color_discrete(guide=gg.guide_legend(title="gene", ncol=1))
    + gg.theme(axis_text_x=gg.element_text(angle=90, hjust=1, vjust=0.5))
    + gg.labs(
        x="cell line",
        y="logFC",
        title="Synthetic data by cell line annotated with real cell line effect",
    )
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_7_0.png)
    





    <ggplot: (8760573461218)>




```python
real_cellline_vals = pd.DataFrame({"cell_line": cell_lines, "log_fc": RP["beta_c"]})

(
    gg.ggplot(data, gg.aes(x="sgRNA", y="log_fc"))
    + gg.geom_jitter(gg.aes(color="cell_line"), height=0, width=0, alpha=0.75)
    + gg.scale_color_discrete(guide=gg.guide_legend(title="cell line", ncol=2))
    + gg.theme(
        axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1), figure_size=(12, 5)
    )
    + gg.labs(x="sgRNA", y="logFC", title="Synthetic data by sgRNA",)
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_8_0.png)
    





    <ggplot: (8760582603644)>



## Model 7a. A 2-Dimensional varying intercept.

$
logFC_{i,g,c} \sim \mathcal{N}(\mu_{g,c}, \sigma) \\
\mu_{g,c} = \alpha_{g,c} \\
\quad \alpha_{g,c} \sim \mathcal{N}(\mu_{\alpha_{g,c}}, \sigma_\alpha) \\
\qquad \mu_{\alpha_{g,c}} \sim \mathcal{N}(0,5) \quad \sigma_\alpha \sim \text{Exp}(1) \\
\sigma \sim \text{Exp}(1)
$


```python
gene_idx = data.gene.cat.codes.to_list()
cell_line_idx = data.cell_line.cat.codes.to_list()

with pm.Model() as m7a:
    # Hyper-priors
    mu_alpha_gc = pm.Normal("mu_alpha_gc", 0, 5)
    sigma_alpha = pm.Exponential("sigma_alpha", 1)

    # Linear model parameters
    alpha_gc = pm.Normal(
        "alpha_sc", mu_alpha_gc, sigma_alpha, shape=(num_genes, num_cell_lines)
    )

    # Linear model
    mu_gc = pm.Deterministic("mu_gc", alpha_gc[gene_idx, cell_line_idx])
    sigma = pm.Exponential("sigma", 1)

    # Likelihood
    log_fc = pm.Normal("log_fc", mu_gc, sigma, observed=data.log_fc.to_list())

    # Sampling
    m7a_prior_check = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    m7a_trace = pm.sample(2000, tune=2000, random_seed=RANDOM_SEED, target_accept=0.95)
    m7a_post_check = pm.sample_posterior_predictive(m7a_trace, random_seed=RANDOM_SEED)
```

    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, alpha_sc, sigma_alpha, mu_alpha_gc]




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
  <progress value='16000' class='' max='16000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [16000/16000 00:24<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 24 seconds.
    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.




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



    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.



```python
pm.model_to_graphviz(m7a)
```




    
![svg](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_11_0.svg)
    




```python
az_m7a = az.from_pymc3(
    trace=m7a_trace,
    model=m7a,
    posterior_predictive=m7a_post_check,
    prior=m7a_prior_check,
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.



```python
m7a_post = (
    az.summary(az_m7a, var_names=["alpha_sc"])
    .reset_index()
    .rename(columns={"index": "intercept"})
    .assign(
        gene_idx=lambda d: d["intercept"].str.extract(r"\[(\d+),"),
        cell_line_idx=lambda d: d["intercept"].str.extract(r",(\d+)\]"),
    )
    .assign(
        gene_idx=lambda d: [int(x) for x in d.gene_idx],
        gene=lambda d: [genes[i] for i in d.gene_idx],
        cell_line_idx=lambda d: [int(x) for x in d.cell_line_idx],
        cell_line=lambda d: [cell_lines[i] for i in d.cell_line_idx],
    )
)
m7a_post.head(n=10)
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
      <th>intercept</th>
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
      <th>gene_idx</th>
      <th>cell_line_idx</th>
      <th>gene</th>
      <th>cell_line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alpha_sc[0,0]</td>
      <td>-0.972</td>
      <td>0.327</td>
      <td>-1.561</td>
      <td>-0.362</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>14043.0</td>
      <td>11904.0</td>
      <td>14082.0</td>
      <td>6079.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>gene_0</td>
      <td>cell_0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alpha_sc[0,1]</td>
      <td>-2.230</td>
      <td>0.329</td>
      <td>-2.864</td>
      <td>-1.631</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>14008.0</td>
      <td>13747.0</td>
      <td>14010.0</td>
      <td>5482.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>gene_0</td>
      <td>cell_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alpha_sc[0,2]</td>
      <td>-0.912</td>
      <td>0.331</td>
      <td>-1.516</td>
      <td>-0.274</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12070.0</td>
      <td>10600.0</td>
      <td>12058.0</td>
      <td>5677.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
      <td>gene_0</td>
      <td>cell_2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alpha_sc[0,3]</td>
      <td>-0.789</td>
      <td>0.335</td>
      <td>-1.406</td>
      <td>-0.152</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>13450.0</td>
      <td>10194.0</td>
      <td>13453.0</td>
      <td>5940.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>gene_0</td>
      <td>cell_3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alpha_sc[0,4]</td>
      <td>-2.692</td>
      <td>0.335</td>
      <td>-3.307</td>
      <td>-2.054</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>14792.0</td>
      <td>14280.0</td>
      <td>14820.0</td>
      <td>5968.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>4</td>
      <td>gene_0</td>
      <td>cell_4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>alpha_sc[0,5]</td>
      <td>-2.934</td>
      <td>0.335</td>
      <td>-3.578</td>
      <td>-2.310</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12694.0</td>
      <td>12551.0</td>
      <td>12683.0</td>
      <td>6005.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>5</td>
      <td>gene_0</td>
      <td>cell_5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>alpha_sc[0,6]</td>
      <td>-0.800</td>
      <td>0.335</td>
      <td>-1.405</td>
      <td>-0.148</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>16092.0</td>
      <td>11237.0</td>
      <td>16109.0</td>
      <td>6175.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>6</td>
      <td>gene_0</td>
      <td>cell_6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alpha_sc[0,7]</td>
      <td>-1.661</td>
      <td>0.334</td>
      <td>-2.290</td>
      <td>-1.041</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>13435.0</td>
      <td>12682.0</td>
      <td>13434.0</td>
      <td>5346.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>7</td>
      <td>gene_0</td>
      <td>cell_7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>alpha_sc[0,8]</td>
      <td>-1.630</td>
      <td>0.337</td>
      <td>-2.245</td>
      <td>-0.978</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12626.0</td>
      <td>11765.0</td>
      <td>12662.0</td>
      <td>5187.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>8</td>
      <td>gene_0</td>
      <td>cell_8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>alpha_sc[0,9]</td>
      <td>-0.460</td>
      <td>0.332</td>
      <td>-1.089</td>
      <td>0.164</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>15090.0</td>
      <td>7381.0</td>
      <td>15109.0</td>
      <td>5020.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>9</td>
      <td>gene_0</td>
      <td>cell_9</td>
    </tr>
  </tbody>
</table>
</div>




```python
(
    gg.ggplot(m7a_post, gg.aes(x="gene", y="cell_line"))
    + gg.geom_tile(gg.aes(fill="mean"))
    + gg.labs(
        x="gene",
        y="cell line",
        fill="posterior mean",
        title="Posterior means of the varying intercepts",
    )
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_14_0.png)
    





    <ggplot: (8760557300017)>




```python
real_gene_vals = pd.DataFrame({"gene": genes, "log_fc": RP["gamma_g"]})
gene_posteriors = m7a_post[["gene", "mean"]].groupby("gene").mean().reset_index()

(
    gg.ggplot(data, gg.aes(x="gene", y="log_fc"))
    + gg.geom_jitter(gg.aes(color="cell_line"), height=0, width=0.3, alpha=0.7)
    + gg.geom_crossbar(gg.aes(ymin="log_fc", ymax="log_fc"), data=real_gene_vals)
    + gg.geom_crossbar(
        gg.aes(ymin="mean", ymax="mean", y="mean"), data=gene_posteriors, color="blue"
    )
    + gg.scale_color_discrete(guide=gg.guide_legend(title="cell line", ncol=2))
    + gg.theme(axis_text_x=gg.element_text(angle=30, hjust=1, vjust=0.2))
    + gg.labs(
        x="gene",
        y="logFC",
        title="Synthetic data by gene\nAnnotated with real (black) and estimated (blue) gene effect",
    )
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_15_0.png)
    





    <ggplot: (8760572722387)>




```python
real_cellline_vals = pd.DataFrame({"cell_line": cell_lines, "log_fc": RP["beta_c"]})
cell_line_posteriors = (
    m7a_post[["cell_line", "mean"]].groupby("cell_line").mean().reset_index()
)

(
    gg.ggplot(data, gg.aes(x="cell_line", y="log_fc"))
    + gg.geom_jitter(gg.aes(color="gene"), height=0, width=0.3, alpha=0.7, size=1)
    + gg.geom_crossbar(gg.aes(ymin="log_fc", ymax="log_fc"), data=real_cellline_vals)
    + gg.geom_crossbar(
        gg.aes(ymin="mean", ymax="mean", y="mean"),
        data=cell_line_posteriors,
        color="blue",
    )
    + gg.scale_color_discrete(guide=gg.guide_legend(title="gene", ncol=1))
    + gg.theme(axis_text_x=gg.element_text(angle=90, hjust=1, vjust=0.5))
    + gg.labs(
        x="cell line",
        y="logFC",
        title="Synthetic data by cell line\nAnnotated with real (black) and estimated (blue) cell line effect",
    )
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_16_0.png)
    





    <ggplot: (8760557317581)>




```python
gene_idx = data.gene.cat.codes.to_list()
cell_line_idx = data.cell_line.cat.codes.to_list()

with pm.Model() as m7a_pool:

    # Linear model parameters
    alpha_gc = pm.Normal("alpha_sc", 0, 5, shape=(num_genes, num_cell_lines))

    # Linear model
    mu_gc = pm.Deterministic("mu_gc", alpha_gc[gene_idx, cell_line_idx])
    sigma = pm.Exponential("sigma", 1)

    # Likelihood
    log_fc = pm.Normal("log_fc", mu_gc, sigma, observed=data.log_fc.to_list())

    # Sampling
    m7a_pool_prior_check = pm.sample_prior_predictive(random_seed=RANDOM_SEED)
    m7a_pool_trace = pm.sample(
        2000, tune=2000, random_seed=RANDOM_SEED, target_accept=0.95
    )
    m7a_pool_post_check = pm.sample_posterior_predictive(
        m7a_pool_trace, random_seed=RANDOM_SEED
    )
```

    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [sigma, alpha_sc]




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
  <progress value='16000' class='' max='16000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [16000/16000 00:23<00:00 Sampling 4 chains, 0 divergences]
</div>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 24 seconds.




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



    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.



```python
pm.model_to_graphviz(m7a_pool)
```




    
![svg](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_18_0.svg)
    




```python
az_m7a_pool = az.from_pymc3(
    trace=m7a_pool_trace,
    model=m7a_pool,
    posterior_predictive=m7a_pool_post_check,
    prior=m7a_pool_prior_check,
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.8/site-packages/theano/tensor/subtensor.py:2197: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.



```python
m7a_pool_post = (
    az.summary(az_m7a_pool, var_names=["alpha_sc"])
    .reset_index()
    .rename(columns={"index": "intercept"})
    .assign(
        gene_idx=lambda d: d["intercept"].str.extract(r"\[(\d+),"),
        cell_line_idx=lambda d: d["intercept"].str.extract(r",(\d+)\]"),
    )
    .assign(
        gene_idx=lambda d: [int(x) for x in d.gene_idx],
        gene=lambda d: [genes[i] for i in d.gene_idx],
        cell_line_idx=lambda d: [int(x) for x in d.cell_line_idx],
        cell_line=lambda d: [cell_lines[i] for i in d.cell_line_idx],
    )
)
m7a_pool_post.head(n=10)
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
      <th>intercept</th>
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
      <th>gene_idx</th>
      <th>cell_line_idx</th>
      <th>gene</th>
      <th>cell_line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alpha_sc[0,0]</td>
      <td>-0.966</td>
      <td>0.340</td>
      <td>-1.615</td>
      <td>-0.333</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12413.0</td>
      <td>10757.0</td>
      <td>12382.0</td>
      <td>5739.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>gene_0</td>
      <td>cell_0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alpha_sc[0,1]</td>
      <td>-2.349</td>
      <td>0.348</td>
      <td>-3.020</td>
      <td>-1.710</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>11758.0</td>
      <td>11251.0</td>
      <td>11762.0</td>
      <td>5584.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>gene_0</td>
      <td>cell_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alpha_sc[0,2]</td>
      <td>-0.907</td>
      <td>0.340</td>
      <td>-1.550</td>
      <td>-0.274</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>11552.0</td>
      <td>9620.0</td>
      <td>11581.0</td>
      <td>5955.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>2</td>
      <td>gene_0</td>
      <td>cell_2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alpha_sc[0,3]</td>
      <td>-0.770</td>
      <td>0.351</td>
      <td>-1.399</td>
      <td>-0.085</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>13773.0</td>
      <td>10072.0</td>
      <td>13783.0</td>
      <td>5782.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>gene_0</td>
      <td>cell_3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alpha_sc[0,4]</td>
      <td>-2.849</td>
      <td>0.347</td>
      <td>-3.503</td>
      <td>-2.186</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>15364.0</td>
      <td>14853.0</td>
      <td>15340.0</td>
      <td>6178.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>4</td>
      <td>gene_0</td>
      <td>cell_4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>alpha_sc[0,5]</td>
      <td>-3.114</td>
      <td>0.356</td>
      <td>-3.748</td>
      <td>-2.426</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>15264.0</td>
      <td>15264.0</td>
      <td>15290.0</td>
      <td>5429.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>5</td>
      <td>gene_0</td>
      <td>cell_5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>alpha_sc[0,6]</td>
      <td>-0.793</td>
      <td>0.345</td>
      <td>-1.445</td>
      <td>-0.134</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12937.0</td>
      <td>10412.0</td>
      <td>12951.0</td>
      <td>6447.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>6</td>
      <td>gene_0</td>
      <td>cell_6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alpha_sc[0,7]</td>
      <td>-1.723</td>
      <td>0.352</td>
      <td>-2.413</td>
      <td>-1.105</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>12401.0</td>
      <td>11951.0</td>
      <td>12422.0</td>
      <td>5660.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>7</td>
      <td>gene_0</td>
      <td>cell_7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>alpha_sc[0,8]</td>
      <td>-1.690</td>
      <td>0.347</td>
      <td>-2.317</td>
      <td>-1.028</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>13534.0</td>
      <td>12979.0</td>
      <td>13550.0</td>
      <td>5134.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>8</td>
      <td>gene_0</td>
      <td>cell_8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>alpha_sc[0,9]</td>
      <td>-0.413</td>
      <td>0.346</td>
      <td>-1.064</td>
      <td>0.233</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>14611.0</td>
      <td>7202.0</td>
      <td>14599.0</td>
      <td>5734.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>9</td>
      <td>gene_0</td>
      <td>cell_9</td>
    </tr>
  </tbody>
</table>
</div>



Note that there is more pooling in the first genes than the later genes.
The first gene has 1 sgRNA and the last gene has 10 sgRNA.


```python
var_names = ["gene", "cell_line", "mean"]
m7a_compare_post = pd.merge(
    m7a_post[var_names].rename(columns={"mean": "part_pool"}),
    m7a_pool_post[var_names].rename(columns={"mean": "full_pool"}),
    on=["gene", "cell_line"],
).melt(id_vars=['gene', 'cell_line'], var_name="pool", value_name="mean")

for col in ["gene", "cell_line"]:
    m7a_compare_post = make_cat(m7a_compare_post, col)

(
    gg.ggplot(m7a_compare_post, gg.aes(x="gene", y="mean"))
    + gg.geom_point(gg.aes(color="pool"), position = gg.position_dodge(width = 0.5))
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.labs(x="gene", y="posterior mean", title="Comparing posteriors of fully and partially pooled models")
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_22_0.png)
    





    <ggplot: (8760557019031)>




```python
(
    gg.ggplot(m7a_compare_post, gg.aes(x="cell_line", y="mean"))
    + gg.geom_point(gg.aes(color="pool"), position = gg.position_dodge(width = 0.5))
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.theme(axis_text_x = gg.element_text(angle=90))
    + gg.labs(x="cell_line", y="posterior mean", title="Comparing posteriors of fully and partially pooled models")
)
```


    
![png](005_017_model-experimentation-m7_files/005_017_model-experimentation-m7_23_0.png)
    





    <ggplot: (8760557103854)>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

---


```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    pandas   1.1.3
    arviz    0.10.0
    pymc3    3.9.3
    seaborn  0.11.0
    plotnine 0.7.1
    numpy    1.19.2
    last updated: 2020-11-09 
    
    CPython 3.8.5
    IPython 7.18.1
    
    compiler   : GCC 7.3.0
    system     : Linux
    release    : 3.10.0-1062.el7.x86_64
    machine    : x86_64
    processor  : x86_64
    CPU cores  : 28
    interpreter: 64bit
    host name  : compute-e-16-235.o2.rc.hms.harvard.edu
    Git branch : models

