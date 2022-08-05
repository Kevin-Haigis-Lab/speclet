# Preparation and testing of model specification


```python
%load_ext autoreload
%autoreload 2
```


```python
from itertools import product
from pathlib import Path
from time import time

import aesara
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc as pm
import pymc.sampling_jax
import seaborn as sns
from matplotlib.lines import Line2D
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.10/site-packages/pymc/sampling_jax.py:36: UserWarning: This module is experimental.



```python
from speclet.analysis.arviz_analysis import extract_coords_param_names
from speclet.bayesian_models.lineage_hierarchical_nb import LineageHierNegBinomModel
from speclet.data_processing.common import head_tail
from speclet.io import DataFile, data_path
from speclet.loggers import set_console_handler_level
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.plot import set_speclet_theme
from speclet.project_configuration import arviz_config
```


```python
# Logging.
set_console_handler_level("DEBUG")
aesara.config.exception_verbosity = "high"

# Notebook execution timer.
notebook_tic = time()

# Plotting setup.
set_speclet_theme()
%config InlineBackend.figure_format = "retina"

# Constants
SEED = 847
np.random.seed(SEED)
arviz_config()
```

## Data


```python
def _broad_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.query("screen == 'broad'").reset_index(drop=True)
```


```python
crispr_data_manager = CrisprScreenDataManager(
    DataFile.DEPMAP_CRC_SUBSAMPLE, transformations=[_broad_only]
)
crc_data = crispr_data_manager.get_data()
```


```python
for col in ["sgrna", "hugo_symbol", "depmap_id"]:
    print(f"'{col}': {crc_data[col].nunique()}")
```

    'sgrna': 162
    'hugo_symbol': 103
    'depmap_id': 8


## Distributions


```python
N = 2000
df = pd.DataFrame()
df = pd.concat(
    [
        df,
        # pd.DataFrame({"value": pm.draw(pm.HalfNormal.dist(0.5), N), "dist": "HN(0.5)"}),
        pd.DataFrame({"value": pm.draw(pm.Exponential.dist(5), N), "dist": "Exp(5)"}),
        pd.DataFrame({"value": pm.draw(pm.Exponential.dist(2), N), "dist": "Exp(2)"}),
        pd.DataFrame({"value": pm.draw(pm.Exponential.dist(1), N), "dist": "Exp(1)"}),
        # pd.DataFrame(
        #     {"value": pm.draw(pm.Exponential.dist(0.2), N), "dist": "Exp(0.5)"}
        # ),
        # pd.DataFrame({"value": pm.draw(pm.Gamma.dist(2, 1), N), "dist": "Gamma(2,1)"}),
        # pd.DataFrame({"value": pm.draw(pm.Gamma.dist(3, 1), N), "dist": "Gamma(3,1)"}),
        # pd.DataFrame({"value": pm.draw(pm.Gamma.dist(3, 2), N), "dist": "Gamma(3,2)"}),
        # pd.DataFrame({"value": pm.draw(pm.Gamma.dist(3, 4), N), "dist": "Gamma(3,4)"}),
        # pd.DataFrame({"value": pm.draw(pm.Gamma.dist(5, 1), N), "dist": "Gamma(5,1)"}),
        # pd.DataFrame(
        #     {"value": pm.draw(pm.Gamma.dist(10, 1), N), "dist": "Gamma(10,1)"}
        # ),
    ]
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 7))
sns.histplot(data=df, x="value", hue="dist", ax=ax, binwidth=0.25)
ax.get_legend().set_title("ditribution")
ax.set_xlim(0, None)
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_10_0.png)




```python
df.groupby("dist").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">value</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>dist</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Exp(1)</th>
      <td>2000.0</td>
      <td>0.987511</td>
      <td>0.941098</td>
      <td>0.001320</td>
      <td>0.282469</td>
      <td>0.702319</td>
      <td>1.413113</td>
      <td>7.314379</td>
    </tr>
    <tr>
      <th>Exp(2)</th>
      <td>2000.0</td>
      <td>0.494138</td>
      <td>0.494454</td>
      <td>0.000304</td>
      <td>0.148168</td>
      <td>0.343012</td>
      <td>0.690708</td>
      <td>4.301374</td>
    </tr>
    <tr>
      <th>Exp(5)</th>
      <td>2000.0</td>
      <td>0.199270</td>
      <td>0.203034</td>
      <td>0.000039</td>
      <td>0.059844</td>
      <td>0.135940</td>
      <td>0.270201</td>
      <td>1.889470</td>
    </tr>
  </tbody>
</table>
</div>




```python
N = 2000
df = pd.concat(
    [
        pd.DataFrame(
            {"value": pm.draw(pm.Normal.dist(0, 0.2), N), "dist": "N(0, 0.2)"}
        ),
        pd.DataFrame(
            {"value": pm.draw(pm.Normal.dist(0, 0.5), N), "dist": "N(0, 0.5)"}
        ),
        pd.DataFrame({"value": pm.draw(pm.Normal.dist(0, 1), N), "dist": "N(0, 1.0)"}),
    ]
).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5, 4))
sns.histplot(data=df, x="value", hue="dist", ax=ax, kde=True)
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_12_0.png)




```python
N = 2000


def _hn(sigma: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": pm.draw(pm.HalfNormal.dist(sigma=sigma), N),
            "dist": f"HN(0, {sigma})",
        }
    )


df = pd.concat([_hn(s) for s in [0.1, 0.2, 0.5, 1.0]]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(data=df, x="value", hue="dist", ax=ax, kde=False)
ax.set_xlim(0, None)
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_13_0.png)




```python
df.groupby("dist").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">value</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>dist</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HN(0, 0.1)</th>
      <td>2000.0</td>
      <td>0.078773</td>
      <td>0.058920</td>
      <td>0.000238</td>
      <td>0.032667</td>
      <td>0.067093</td>
      <td>0.113422</td>
      <td>0.321725</td>
    </tr>
    <tr>
      <th>HN(0, 0.2)</th>
      <td>2000.0</td>
      <td>0.161141</td>
      <td>0.121962</td>
      <td>0.000011</td>
      <td>0.064464</td>
      <td>0.136247</td>
      <td>0.232477</td>
      <td>0.766230</td>
    </tr>
    <tr>
      <th>HN(0, 0.5)</th>
      <td>2000.0</td>
      <td>0.389072</td>
      <td>0.294415</td>
      <td>0.000019</td>
      <td>0.159532</td>
      <td>0.327502</td>
      <td>0.547573</td>
      <td>1.875374</td>
    </tr>
    <tr>
      <th>HN(0, 1.0)</th>
      <td>2000.0</td>
      <td>0.783566</td>
      <td>0.582628</td>
      <td>0.000333</td>
      <td>0.326765</td>
      <td>0.670526</td>
      <td>1.130155</td>
      <td>3.364777</td>
    </tr>
  </tbody>
</table>
</div>




```python
N = 2000
df = pd.DataFrame()
df = pd.concat(
    [
        df,
        pd.DataFrame(
            {"value": pm.draw(pm.NegativeBinomial.dist(1, 1), N), "dist": "NB(1,1)"}
        ),
        pd.DataFrame(
            {"value": pm.draw(pm.NegativeBinomial.dist(10, 1), N), "dist": "NB(10,1)"}
        ),
        pd.DataFrame(
            {"value": pm.draw(pm.NegativeBinomial.dist(10, 5), N), "dist": "NB(10,5)"}
        ),
        pd.DataFrame(
            {
                "value": pm.draw(pm.NegativeBinomial.dist(10, 10), N),
                "dist": "NB(10,10)",
            }
        ),
    ]
)
df = df.reset_index(drop=True)
ax = sns.histplot(data=df, x="value", hue="dist", kde=True, binwidth=1)
ax.set_xlim(0, None)
```




    (0.0, 101.85)





![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_15_1.png)




```python
N = 2000


def _lkj(eta: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": pm.draw(pm.LKJCorr.dist(n=2, eta=eta), N).flatten(),
            "dist": f"eta={eta}",
        }
    )


df = pd.concat([_lkj(i) for i in range(1, 6)]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5, 4))
sns.histplot(data=df, x="value", hue="dist", kde=True, binwidth=0.1, ax=ax)
ax.set_xlim(-1, 1)
ax.set_title("Example LKJCorr distributions")
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_16_0.png)




```python
df.groupby("dist").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">value</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>dist</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>eta=1</th>
      <td>2000.0</td>
      <td>-0.020639</td>
      <td>0.577520</td>
      <td>-0.997240</td>
      <td>-0.521119</td>
      <td>-0.036630</td>
      <td>0.488077</td>
      <td>0.998603</td>
    </tr>
    <tr>
      <th>eta=2</th>
      <td>2000.0</td>
      <td>0.007028</td>
      <td>0.449241</td>
      <td>-0.985251</td>
      <td>-0.351010</td>
      <td>0.019441</td>
      <td>0.353885</td>
      <td>0.989104</td>
    </tr>
    <tr>
      <th>eta=3</th>
      <td>2000.0</td>
      <td>0.004810</td>
      <td>0.379125</td>
      <td>-0.929154</td>
      <td>-0.277673</td>
      <td>0.016175</td>
      <td>0.285917</td>
      <td>0.930181</td>
    </tr>
    <tr>
      <th>eta=4</th>
      <td>2000.0</td>
      <td>0.006035</td>
      <td>0.323950</td>
      <td>-0.903282</td>
      <td>-0.231161</td>
      <td>0.008042</td>
      <td>0.241769</td>
      <td>0.866332</td>
    </tr>
    <tr>
      <th>eta=5</th>
      <td>2000.0</td>
      <td>-0.003324</td>
      <td>0.301282</td>
      <td>-0.858675</td>
      <td>-0.223390</td>
      <td>0.006455</td>
      <td>0.211751</td>
      <td>0.837248</td>
    </tr>
  </tbody>
</table>
</div>



## Modeling


```python
from speclet.bayesian_models.lineage_hierarchical_nb import LineageHierNegBinomModel
```


```python
crc_model = LineageHierNegBinomModel(
    lineage="colorectal", reduce_deterministic_vars=False
)
valid_crc_data = crc_model.data_processing_pipeline(crc_data.copy())
model_crc_data = crc_model.make_data_structure(valid_crc_data.copy())
```

    [INFO] 2022-08-05 14:12:07 [(lineage_hierarchical_nb.py:data_processing_pipeline:285] Processing data for modeling.
    [INFO] 2022-08-05 14:12:07 [(lineage_hierarchical_nb.py:data_processing_pipeline:286] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-08-05 14:12:07 [(lineage_hierarchical_nb.py:data_processing_pipeline:337] number of data points dropped: 0
    [INFO] 2022-08-05 14:12:08 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:534] number of genes mutated in all cells lines: 1
    [DEBUG] 2022-08-05 14:12:08 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:537] Genes always mutated: APC
    [DEBUG] 2022-08-05 14:12:08 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:68] all_mut: {}
    [INFO] 2022-08-05 14:12:08 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:77] Dropping 2 cancer genes.
    [DEBUG] 2022-08-05 14:12:08 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:79] Dropped cancer genes: ['APC', 'MDM2']



```python
model_crc_data.coords["cancer_gene"]
```




    ['FBXW7', 'KRAS', 'PIK3CA']




```python
crc_pymc_model = crc_model.pymc_model(crispr_data_manager.data.copy())
pm.model_to_graphviz(crc_pymc_model)
```

    [INFO] 2022-08-05 14:13:09 [(lineage_hierarchical_nb.py:data_processing_pipeline:285] Processing data for modeling.
    [INFO] 2022-08-05 14:13:09 [(lineage_hierarchical_nb.py:data_processing_pipeline:286] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-08-05 14:13:10 [(lineage_hierarchical_nb.py:data_processing_pipeline:337] number of data points dropped: 0
    [INFO] 2022-08-05 14:13:10 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:534] number of genes mutated in all cells lines: 1
    [DEBUG] 2022-08-05 14:13:10 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:537] Genes always mutated: APC
    [DEBUG] 2022-08-05 14:13:10 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:68] all_mut: {}
    [INFO] 2022-08-05 14:13:10 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:77] Dropping 2 cancer genes.
    [DEBUG] 2022-08-05 14:13:10 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:79] Dropped cancer genes: ['APC', 'MDM2']
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:341] Lineage: colorectal
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:342] Number of genes: 103
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:343] Number of sgRNA: 162
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:344] Number of cell lines: 8
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:345] Number of cancer genes: 3
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:346] Number of screens: 1
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:347] Number of data points: 1296
    [INFO] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:_pre_model_messages:352] Including all non-essential deterministic variables.
    [DEBUG] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:pymc_model:388] shape of cancer gene matrix: (1296, 3)
    [DEBUG] 2022-08-05 14:13:11 [(lineage_hierarchical_nb.py:pymc_model:403] location for `mu_mu_a`: 0.1523






![svg](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_22_1.svg)





```python
with crc_pymc_model:
    pm_prior_pred = pm.sample_prior_predictive(
        var_names=["eta", "ct_final", "mu_a", "gene_effect"], random_seed=SEED
    )

print("prior predictive distribution")
pm_pred_draws = pm_prior_pred.prior_predictive["ct_final"].values.squeeze()
for q in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]:
    res = np.quantile(pm_pred_draws, q=q)
    print(f"  {int(q*100)}%: {int(res):0,d}")

print("")
obs = valid_crc_data["counts_final"].astype(int)
print(f"final counts\n  min: {np.min(obs):,d},  max: {np.max(obs):,d}")

obs = valid_crc_data["counts_initial_adj"].astype(int)
print(f"initial counts\n  min: {np.min(obs):,d},  max: {np.max(obs):,d}")
```

    prior predictive distribution
      0%: 0
      1%: 0
      10%: 8
      20%: 48
      30%: 107
      40%: 190
      50%: 308
      60%: 485
      70%: 772
      80%: 1,311
      90%: 2,737
      99%: 18,874
      100%: 46,891,488

    final counts
      min: 0,  max: 9,819
    initial counts
      min: 57,  max: 4,741



```python
eta_prior = np.random.choice(pm_prior_pred.prior["eta"].values.flatten(), 2000)
ge_prior = np.random.choice(pm_prior_pred.prior["gene_effect"].values.flatten(), 4000)
mu_prior = np.random.choice(pm_prior_pred.prior["mu_a"].values.flatten(), 4000)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3))
sns.histplot(mu_prior, kde=True, ax=axes[0], binwidth=0.5, stat="proportion")
sns.histplot(ge_prior, kde=True, ax=axes[1], binwidth=1, stat="proportion")
sns.histplot(eta_prior, kde=True, ax=axes[2], binwidth=1, stat="proportion")
axes[0].set_xlabel(r"$\mu_a$")
axes[1].set_xlabel(r"gene effect")
axes[2].set_xlabel(r"$\eta$")

for ax in axes.flatten():
    ax.set_ylabel(None)
    ax.set_title(None)

fig.supylabel("proportion")
fig.suptitle("Prior predictive distribution")
fig.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_24_0.png)




```python
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 7))

stat = "proportion"

obs_max = crc_data["counts_final"].max()
truncated_prior_preds = [x for x in pm_pred_draws.flatten() if x <= obs_max]

prior_pred_pal = {"prior pred.": "tab:orange", "observed": "gray"}

# Untransformed
bw: float = 100
sns.histplot(
    x=truncated_prior_preds,
    ax=axes[0],
    binwidth=bw,
    stat=stat,
    color=prior_pred_pal["prior pred."],
)
sns.histplot(
    x=valid_crc_data["counts_final"],
    ax=axes[0],
    binwidth=bw,
    stat=stat,
    color=prior_pred_pal["observed"],
)
# Log10 transformed
bw = 0.25
sns.histplot(
    x=np.log10(pm_pred_draws.flatten() + 1),
    ax=axes[1],
    binwidth=bw,
    stat=stat,
    color=prior_pred_pal["prior pred."],
)
sns.histplot(
    x=np.log10(valid_crc_data["counts_final"] + 1),
    ax=axes[1],
    binwidth=bw,
    stat=stat,
    color=prior_pred_pal["observed"],
)
# Log-fold change
pp_lfc = np.log(
    (pm_pred_draws + 1) / (valid_crc_data["counts_initial_adj"].values[None, :])
)
obs_lfc = np.log(
    (valid_crc_data["counts_final"] + 1) / valid_crc_data["counts_initial_adj"]
)
bw = 0.5
sns.histplot(
    x=pp_lfc.flatten(),
    ax=axes[2],
    binwidth=bw,
    stat=stat,
    color=prior_pred_pal["prior pred."],
)
sns.histplot(
    x=obs_lfc,
    ax=axes[2],
    binwidth=bw,
    stat=stat,
    color=prior_pred_pal["observed"],
)


axes[0].set_xlabel("final counts")
axes[1].set_xlabel("log10(final counts + 1)")
axes[2].set_xlabel("log((final + 1) / initial)")

prior_pred_leg_handles = [
    Line2D([0], [0], linewidth=10, color=v, label=k) for k, v in prior_pred_pal.items()
]
axes[0].legend(handles=prior_pred_leg_handles, loc="upper right", frameon=False)

for ax in axes.flatten():
    ax.set_ylabel(stat)

plt.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_25_0.png)



## Sampling


```python
with crc_pymc_model:
    trace = pymc.sampling_jax.sample_numpyro_nuts(
        draws=500,
        tune=1000,
        target_accept=0.90,
        random_seed=SEED,
        idata_kwargs={"log_likelihood": False},
        nuts_kwargs={"step_size": 0.1, "max_tree_depth": 12}
        # discard_tuned_samples=False,
    )
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=SEED)
```

    Compiling...
    Compilation time =  0:00:16.416372
    Sampling...



      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/1500 [00:00<?, ?it/s]


    Sampling time =  0:02:01.515370
    Transforming variables...
    Transformation time =  0:00:07.937862




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:03&lt;00:00]
</div>




```python
saved_warmup = hasattr(trace, "warmup_sample_stats")
```

## Posterior analysis


```python
az.plot_energy(trace);
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_30_0.png)




```python
divs = trace.sample_stats["diverging"].values.sum(axis=1)
print(f"divergences: {divs}")
```

    divergences: [0 0 1 0]



```python
if saved_warmup:
    ar = trace.warmup_sample_stats.acceptance_rate.values
    ss = trace.warmup_sample_stats.step_size.values
    n_chains = ss.shape[0]
    x = np.arange(ss.shape[1])
    fig, axes = plt.subplots(n_chains // 2, 2, figsize=(8, 4), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flatten()):
        sns.scatterplot(x=x, y=ar[c, :], color="tab:blue", edgecolor=None, s=2, ax=ax)
        sns.scatterplot(x=x, y=ss[c, :], color="tab:green", edgecolor=None, s=2, ax=ax)

    fig.tight_layout()
    plt.show()
```


```python
ar = trace.sample_stats.acceptance_rate.values
ss = trace.sample_stats.step_size.values
n_chains = ss.shape[0]
x = np.arange(ss.shape[1])
fig, axes = plt.subplots(n_chains // 2, 2, figsize=(8, 4), sharex=True, sharey=True)
for c, ax in enumerate(axes.flatten()):
    sns.scatterplot(x=x, y=ar[c, :], color="tab:blue", edgecolor=None, s=2, ax=ax)
    sns.scatterplot(x=x, y=ss[c, :], color="tab:green", edgecolor=None, s=2, ax=ax)
    ax.set_title(f"chain {c}")

fig.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_33_0.png)




```python
stat_cats = ["energy", "step_size", "n_steps", "acceptance_rate"]
trace.sample_stats.get(stat_cats).to_dataframe().groupby("chain").mean().round(3)
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
      <th>energy</th>
      <th>step_size</th>
      <th>n_steps</th>
      <th>acceptance_rate</th>
    </tr>
    <tr>
      <th>chain</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10212.710</td>
      <td>0.069</td>
      <td>63.000</td>
      <td>0.919</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10203.462</td>
      <td>0.042</td>
      <td>127.000</td>
      <td>0.953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10209.039</td>
      <td>0.104</td>
      <td>55.282</td>
      <td>0.867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10208.635</td>
      <td>0.044</td>
      <td>127.000</td>
      <td>0.914</td>
    </tr>
  </tbody>
</table>
</div>




```python
if saved_warmup:
    print(trace.warmup_posterior["sigma_mu_a"].values[:, 0:5])
```


```python
if saved_warmup:
    az.plot_trace(
        trace.warmup_posterior,
        var_names=crc_model.vars_regex() + ["~^h$"],
        filter_vars="regex",
        coords={"draw": list(range(100))},
    )
    plt.tight_layout()
    plt.show()
```


```python
if saved_warmup:
    az.plot_trace(
        trace.warmup_posterior,
        var_names=crc_model.vars_regex() + ["~^h$"],
        filter_vars="regex",
    )
    plt.tight_layout()
    plt.show()
```


```python
az.plot_trace(trace, var_names=crc_model.vars_regex() + ["~^h$"], filter_vars="regex")
plt.tight_layout();
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_38_0.png)




```python
n_chains = len(trace.posterior.coords["chain"])
fig, axes = plt.subplots(2, 2, figsize=(6, 5))
for c, ax in enumerate(axes.flatten()):
    cor_mat = trace.posterior["genes_chol_cov_corr"][c, :, :, :].mean(axis=(0))
    sns.heatmap(cor_mat, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)

fig.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_39_0.png)




```python
n_chains = len(trace.posterior.coords["chain"])
fig, axes = plt.subplots(2, 2, figsize=(5, 4))
for c, ax in enumerate(axes.flatten()):
    cor_mat = trace.posterior["cells_chol_cov_corr"][c, :, :, :].mean(axis=(0))
    sns.heatmap(cor_mat, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)

fig.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_40_0.png)




```python
axes = az.plot_forest(
    trace,
    var_names=["mu_b", "mu_d", "mu_f", "mu_h", "^sigma.*$"],
    filter_vars="regex",
    combined=True,
    figsize=(5, 5),
    r_hat=True,
)
line = axes[0].axvline(color="k", alpha=0.5, linewidth=1)
line.set_zorder(0)
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_41_0.png)




```python
axes = az.plot_forest(
    trace, var_names=["k", "m"], combined=True, r_hat=True, figsize=(5, 6)
)
line = axes[0].axvline(color="gray")
line.set_zorder(0)
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_42_0.png)




```python
gene_order = (
    az.summary(trace, var_names=["mu_a"], kind="stats")
    .pipe(extract_coords_param_names, names=["hugo_symbol"])
    .sort_values("mean")["hugo_symbol"]
    .tolist()
)

sgrna_to_gene_map = (
    crc_data[["hugo_symbol", "sgrna"]].drop_duplicates().reset_index(drop=True)
)
a_post = (
    az.summary(trace, var_names=["a"], kind="stats")
    .pipe(extract_coords_param_names, names=["sgrna"])
    .reset_index(drop=True)
    .merge(sgrna_to_gene_map, on="sgrna", validate="one_to_one")
)
mu_a_post = (
    az.summary(trace, var_names=["mu_a"], kind="stats")
    .pipe(extract_coords_param_names, names=["hugo_symbol"])
    .reset_index(drop=True)
)

a_post["hugo_symbol"] = pd.Categorical(
    a_post["hugo_symbol"], categories=gene_order, ordered=True
)
a_post = a_post.sort_values("hugo_symbol").reset_index(drop=True)
mu_a_post["hugo_symbol"] = pd.Categorical(
    mu_a_post["hugo_symbol"], categories=gene_order, ordered=True
)
mu_a_post = mu_a_post.sort_values("hugo_symbol").reset_index(drop=True)

mu_mu_a_post = az.summary(trace, var_names=["mu_mu_a"], kind="stats")
assert len(mu_mu_a_post) == 1
mu_mu_a_avg = mu_mu_a_post["mean"][0]
mu_mu_a_hdi = (mu_mu_a_post["hdi_5.5%"][0], mu_mu_a_post["hdi_94.5%"][0])

fig, ax = plt.subplots(figsize=(4, 20))
# plt.axvline(0, color="grey")

# Population average and HDI
plt.fill_between(x=mu_mu_a_hdi, y1=-1, y2=len(mu_a_post), alpha=0.1)
plt.axvline(mu_mu_a_avg, color="k", linestyle="--")

# Gene estimates.
plt.hlines(
    y=mu_a_post["hugo_symbol"],
    xmin=mu_a_post["hdi_5.5%"],
    xmax=mu_a_post["hdi_94.5%"],
    color="tab:blue",
    linewidth=2,
)
plt.scatter(x=mu_a_post["mean"], y=mu_a_post["hugo_symbol"], s=20, c="tab:blue")

# sgRNA estimates.
plt.hlines(
    y=a_post["hugo_symbol"],
    xmin=a_post["hdi_5.5%"],
    xmax=a_post["hdi_94.5%"],
    color="tab:red",
    linewidth=1,
    alpha=0.5,
)
plt.scatter(x=a_post["mean"], y=a_post["hugo_symbol"], s=10, c="tab:red")

plt.ylim(-1, len(mu_a_post))
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_43_0.png)




```python
top_mut_effect_genes = (
    az.summary(trace, var_names=["f"])
    .sort_values("mean")
    .pipe(head_tail)
    .pipe(extract_coords_param_names, "hugo_symbol")["hugo_symbol"]
    .tolist()
)
ax = az.plot_forest(
    trace,
    var_names=["f"],
    coords={"gene": top_mut_effect_genes},
    combined=True,
    figsize=(5, 6),
)
ax[0].axvline(0, color="k")
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_44_0.png)




```python
az.plot_pair(
    trace,
    var_names=["mu_a", "b", "d", "f"],
    coords={"gene": ["BRAF"]},
    scatter_kwargs={"alpha": 0.25, "markersize": 1},
    figsize=(6, 6),
)
plt.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_45_0.png)




```python
az.plot_pair(
    trace,
    var_names=["mu_mu_a", "mu_b", "mu_m"],
    scatter_kwargs={"alpha": 0.5, "markersize": 1},
    figsize=(5, 5),
)
plt.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_46_0.png)




```python
az.plot_pair(
    trace,
    var_names=["d", "m"],
    coords={
        "gene": crc_data["hugo_symbol"].cat.categories[:3],
        "cell_line": crc_data["depmap_id"].cat.categories[:2],
    },
    scatter_kwargs={"alpha": 0.5, "markersize": 1},
    figsize=(9, 9),
)
plt.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_47_0.png)




```python
for var_name in ["mu_a", "b", "d", "h"]:
    az.plot_pair(
        trace,
        var_names=[var_name],
        coords={
            "gene": crc_data["hugo_symbol"].cat.categories[:5],
            "cancer_gene": ["KRAS"],
        },
        scatter_kwargs={"alpha": 0.5, "markersize": 1},
        figsize=(9, 9),
    )
    plt.tight_layout()
    plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_48_0.png)





![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_48_1.png)





![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_48_2.png)





![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_48_3.png)




```python
for var_name in ["k", "m"]:
    az.plot_pair(
        trace,
        var_names=[var_name],
        coords={
            "cell_line": crc_data["depmap_id"].cat.categories[:5],
        },
        scatter_kwargs={"alpha": 0.5, "markersize": 1},
        figsize=(9, 9),
    )
    plt.tight_layout()
    plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_49_0.png)





![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_49_1.png)




```python
cn_data = valid_crc_data[
    ["hugo_symbol", "depmap_id", "cn_gene", "cn_cell_line"]
].drop_duplicates()
ax = sns.scatterplot(
    data=cn_data,
    x="cn_gene",
    y="cn_cell_line",
    hue="depmap_id",
    s=10,
    alpha=0.7,
    edgecolor=None,
)
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_50_0.png)




```python
gene_effects = (
    trace.posterior.get(["mu_a", "b", "d", "f", "h"])
    .to_dataframe()
    .groupby(["gene"])
    .mean()
)
g = sns.pairplot(
    gene_effects,
    corner=True,
    diag_kind="kde",
    height=2,
    aspect=1,
    plot_kws={"edgecolor": None, "alpha": 0.7, "s": 10, "color": "tab:blue"},
)
g.map_lower(sns.kdeplot, levels=4, color="gray", alpha=0.5)
# g.map_lower(lambda *args, **kwargs: plt.gca().axhline(color="k", linewidth=0.7))
# g.map_lower(lambda *args, **kwargs: plt.gca().axvline(color="k", linewidth=0.7))
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_51_0.png)




```python
b_f_post = (
    az.summary(trace, var_names=["f", "b"])
    .pipe(extract_coords_param_names, names=["hugo_symbol"])
    .assign(var_name=lambda d: [x[0] for x in d.index.values])
    .pivot_wider("hugo_symbol", names_from="var_name", values_from="mean")
    .set_index("hugo_symbol")
)

jp = sns.jointplot(
    data=b_f_post,
    x="b",
    y="f",
    marginal_kws={"kde": True},
    joint_kws={"edgecolor": None, "alpha": 0.7},
)
ax = jp.ax_joint
ax.axhline(0, color="k", alpha=0.5)
ax.axvline(0, color="k", alpha=0.5)

genes_to_label = list(trace.posterior.coords["cancer_gene"].values)
genes_to_label += b_f_post[b_f_post["f"] < -0.1].index.tolist()
genes_to_label += b_f_post[b_f_post["f"] > 0.1].index.tolist()
genes_to_label += b_f_post[b_f_post["b"] < -0.03].index.tolist()
genes_to_label += b_f_post[b_f_post["b"] > 0.03].index.tolist()
genes_to_label = list(set(genes_to_label))
for gene in genes_to_label:
    data = b_f_post.query(f"hugo_symbol == '{gene}'")
    assert len(data) == 1
    ax.text(data["b"], data["f"], s=gene, alpha=0.9)


plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_52_0.png)




```python
(
    valid_crc_data.filter_column_isin(
        "hugo_symbol", trace.posterior.coords["cancer_gene"].values
    )[["hugo_symbol", "depmap_id", "is_mutated"]]
    .drop_duplicates()
    .assign(is_mutated=lambda d: d["is_mutated"].map({True: "X", False: ""}))
    .pivot_wider("depmap_id", "hugo_symbol", "is_mutated")
    .set_index("depmap_id")
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
      <th>KRAS</th>
      <th>FBXW7</th>
      <th>PIK3CA</th>
    </tr>
    <tr>
      <th>depmap_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACH-000253</th>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>ACH-000286</th>
      <td>X</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>ACH-000296</th>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>ACH-000350</th>
      <td>X</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>ACH-000470</th>
      <td>X</td>
      <td>X</td>
      <td></td>
    </tr>
    <tr>
      <th>ACH-000958</th>
      <td></td>
      <td>X</td>
      <td>X</td>
    </tr>
    <tr>
      <th>ACH-001786</th>
      <td></td>
      <td>X</td>
      <td>X</td>
    </tr>
    <tr>
      <th>ACH-002024</th>
      <td>X</td>
      <td></td>
      <td>X</td>
    </tr>
  </tbody>
</table>
</div>




```python
h_post_summary = (
    az.summary(trace, var_names="h", kind="stats")
    .pipe(extract_coords_param_names, names=["hugo_symbol", "cancer_gene"])
    .pivot_wider("hugo_symbol", names_from="cancer_gene", values_from="mean")
    .set_index("hugo_symbol")
)

vmax = np.abs(h_post_summary.values).max()

figsize = (3.3, 12)
dendro_ratio = (0.1, figsize[0] * 0.1 / figsize[1])
cm = sns.clustermap(
    h_post_summary,
    z_score=None,
    cmap="coolwarm",
    vmin=-vmax,
    vmax=vmax,
    figsize=figsize,
    dendrogram_ratio=dendro_ratio,
    cbar_pos=(1, 0.4, 0.1, 0.2),
    yticklabels=1,
)
cm.ax_heatmap.tick_params("both", labelsize=7, size=0)
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_54_0.png)




```python
cg_mut_labels = (
    valid_crc_data.filter_column_isin(
        "hugo_symbol", trace.posterior.coords["cancer_gene"].values
    )[["hugo_symbol", "depmap_id", "is_mutated"]]
    .drop_duplicates()
    .pivot_wider("depmap_id", "hugo_symbol", "is_mutated")
    .set_index("depmap_id")
)
_labels = []
for i in range(len(cg_mut_labels)):
    muts = cg_mut_labels.columns[cg_mut_labels.values[i, :]].tolist()
    if len(muts) == 0:
        _labels.append("WT")
    else:
        _labels.append(",".join(muts))

cg_mut_labels["mut_label"] = _labels
cg_mut_labels = cg_mut_labels[["mut_label"]].reset_index(drop=False)
cg_mut_labels
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
      <th>depmap_id</th>
      <th>mut_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACH-000253</td>
      <td>WT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACH-000286</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACH-000296</td>
      <td>WT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACH-000350</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACH-000470</td>
      <td>KRAS,FBXW7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ACH-000958</td>
      <td>FBXW7,PIK3CA</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ACH-001786</td>
      <td>FBXW7,PIK3CA</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ACH-002024</td>
      <td>KRAS,PIK3CA</td>
    </tr>
  </tbody>
</table>
</div>




```python
pik3ca_hits = h_post_summary.sort_values("PIK3CA").query("PIK3CA > 0.5").index.tolist()

pik3ca_hits_data = (
    crc_data.copy()
    .filter_column_isin("hugo_symbol", pik3ca_hits)
    .merge(cg_mut_labels, on="depmap_id")
    .reset_index(drop=True)
    .assign(
        hugo_symbol=lambda d: pd.Categorical(
            d["hugo_symbol"], categories=pik3ca_hits, ordered=True
        )
    )
)


fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(
    data=pik3ca_hits_data,
    x="hugo_symbol",
    y="lfc",
    hue="mut_label",
    dodge=True,
    ax=ax,
    flierprops={"markersize": 0},
    boxprops={"alpha": 0.5},
)
sns.move_legend(ax, title="mutations", loc="upper left", bbox_to_anchor=(1, 1))
ax.add_artist(ax.get_legend())
sns.swarmplot(
    data=pik3ca_hits_data,
    x="hugo_symbol",
    y="lfc",
    hue="mut_label",
    dodge=True,
    ax=ax,
    s=4,
)

ax.set_title("$\mathit{PIK3CA}$ hits")
ax.set_xlabel(None)
ax.set_ylabel("log-fold change")
ax.get_legend().remove()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_56_0.png)




```python
gene_vars = ["$\mu_a$", "$b$", "$d$", "$f$"]
gene_vars += ["$h_{" + g + "}$" for g in trace.posterior.coords["cancer_gene"].values]
gene_corr_post = (
    az.summary(trace, var_names=["genes_chol_cov_corr"], kind="stats")
    .pipe(extract_coords_param_names, names=["d0", "d1"])
    .astype({"d0": int, "d1": int})
    .assign(
        p0=lambda d: [gene_vars[i] for i in d["d0"]],
        p1=lambda d: [gene_vars[i] for i in d["d1"]],
    )
)

plot_df = gene_corr_post.pivot_wider(
    "p0", names_from="p1", values_from="mean"
).set_index("p0")
ax = sns.heatmap(plot_df, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xlabel(None)
ax.set_ylabel(None)
plt.tight_layout()
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_57_0.png)




```python
N = 100

pp_dist = trace.posterior_predictive["ct_final"]
n_chains, n_draws, n_samples = pp_dist.shape
draws_idx = np.random.choice(np.arange(n_draws), N // n_chains, replace=False)

fig, ax = plt.subplots(figsize=(8, 5))

ppc_pal = {
    "draws": "tab:blue",
    "median": "tab:orange",
    "mean": "tab:red",
    "observed": "black",
}

# Example draws.
for c, d in product(range(n_chains), draws_idx):
    values = np.log(pp_dist[c, d, :].values + 1)
    sns.kdeplot(values, color=ppc_pal["draws"], alpha=0.1, ax=ax)

# Average distributions.
pp_dist_mean = np.log(pp_dist.median(axis=(0, 1)) + 1)
pp_dist_mid = np.log(pp_dist.mean(axis=(0, 1)) + 1)
sns.kdeplot(pp_dist_mean, color=ppc_pal["mean"], ax=ax, alpha=0.5)
sns.kdeplot(pp_dist_mid, color=ppc_pal["median"], ax=ax, alpha=0.5)

# Observed distribution.
sns.kdeplot(
    np.log(trace.observed_data["ct_final"] + 1), ax=ax, color=ppc_pal["observed"]
)

ppc_leg_handles: list[Line2D] = []
for lbl, color in ppc_pal.items():
    ppc_leg_handles.append(Line2D([0], [0], color=color, label=lbl))

plt.legend(handles=ppc_leg_handles, loc="best")


ax.set_xlabel("log10( final counts + 1 )")
ax.set_ylabel("density")
ax.set_title("Posterior predictive distribution")
plt.show()
```



![png](050_simplify-single-lineage-model_files/050_simplify-single-lineage-model_58_0.png)



---


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 7.59 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-08-05

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : Clang 13.0.1
    OS          : Darwin
    Release     : 21.5.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: jhcookmac.harvardsecure.wireless.med.harvard.edu

    Git branch: simplify

    numpy     : 1.23.1
    arviz     : 0.12.1
    pymc      : 4.1.3
    seaborn   : 0.11.2
    plotnine  : 0.0.0
    aesara    : 2.7.7
    matplotlib: 3.5.2
    pandas    : 1.4.3




```python

```
