# Inspect the single-lineage model run on the prostate data (002)

Model attributes:

- sgRNA | gene varying intercept
- RNA and CN varying effects per gene
- correlation between gene varying effects modeled using the multivariate normal and Cholesky decomposition (non-centered parameterization)
- target gene mutation variable and cancer gene comutation variable.


```python
%load_ext autoreload
%autoreload 2
```


```python
from time import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qnorm
import seaborn as sns
```


```python
from speclet.analysis.arviz_analysis import extract_coords_param_names
from speclet.bayesian_models.lineage_hierarchical_nb import LineageHierNegBinomModel
from speclet.data_processing.common import head_tail
from speclet.io import modeling_data_dir, models_dir
from speclet.managers.data_managers import CrisprScreenDataManager
from speclet.plot import set_speclet_theme
from speclet.project_configuration import arviz_config
```


```python
# Notebook execution timer.
notebook_tic = time()

# Plotting setup.
set_speclet_theme()
%config InlineBackend.figure_format = "retina"
arviz_config()
```

## Data


```python
saved_model_dir = models_dir() / "hnb-single-lineage-prostate-002_PYMC_NUMPYRO"
```


```python
with open(saved_model_dir / "description.txt") as f:
    model_description = "".join(list(f))

print(model_description)
```

    name: 'hnb-single-lineage-prostate-002'
    fit method: 'PYMC_NUMPYRO'

    --------------------------------------------------------------------------------

    CONFIGURATION

    {
        "name": "hnb-single-lineage-prostate-002",
        "description": " Single lineage hierarchical negative binomial model for prostate data from the Broad. This model builds on 001 with a variable for the mutation of the target gene and a variable for the mutation of cancer genes. ",
        "active": true,
        "model": "LINEAGE_HIERARCHICAL_NB",
        "data_file": "modeling_data/lineage-modeling-data/depmap-modeling-data_prostate.csv",
        "model_kwargs": {
            "lineage": "prostate"
        },
        "sampling_kwargs": {
            "pymc_mcmc": null,
            "pymc_advi": null,
            "pymc_numpyro": {
                "draws": 1000,
                "tune": 1000,
                "chains": 4,
                "target_accept": 0.97,
                "progress_bar": true,
                "chain_method": "parallel",
                "postprocessing_backend": "cpu",
                "idata_kwargs": {
                    "log_likelihood": false
                },
                "nuts_kwargs": null
            }
        }
    }

    --------------------------------------------------------------------------------

    POSTERIOR

    <xarray.Dataset>
    Dimensions:                    (chain: 4, draw: 1000, cancer_gene: 1,
                                    delta_genes_dim_0: 5, delta_genes_dim_1: 18119,
                                    sgrna: 71062, genes_chol_cov_dim_0: 15,
                                    genes_chol_cov_corr_dim_0: 5,
                                    genes_chol_cov_corr_dim_1: 5,
                                    genes_chol_cov_stds_dim_0: 5, sigma_h_dim_0: 1,
                                    gene: 18119)
    Coordinates:
      * chain                      (chain) int64 0 1 2 3
      * draw                       (draw) int64 0 1 2 3 4 5 ... 995 996 997 998 999
      * cancer_gene                (cancer_gene) object 'ZFHX3'
      * delta_genes_dim_0          (delta_genes_dim_0) int64 0 1 2 3 4
      * delta_genes_dim_1          (delta_genes_dim_1) int64 0 1 2 ... 18117 18118
      * sgrna                      (sgrna) object 'AAAAAAATCCAGCAATGCAG' ... 'TTT...
      * genes_chol_cov_dim_0       (genes_chol_cov_dim_0) int64 0 1 2 3 ... 12 13 14
      * genes_chol_cov_corr_dim_0  (genes_chol_cov_corr_dim_0) int64 0 1 2 3 4
      * genes_chol_cov_corr_dim_1  (genes_chol_cov_corr_dim_1) int64 0 1 2 3 4
      * genes_chol_cov_stds_dim_0  (genes_chol_cov_stds_dim_0) int64 0 1 2 3 4
      * sigma_h_dim_0              (sigma_h_dim_0) int64 0
      * gene                       (gene) object 'A1BG' 'A1CF' ... 'ZZEF1' 'ZZZ3'
    Data variables: (12/23)
        mu_mu_a                    (chain, draw) float64 ...
        mu_b                       (chain, draw) float64 ...
        mu_d                       (chain, draw) float64 ...
        mu_f                       (chain, draw) float64 ...
        mu_h                       (chain, draw, cancer_gene) float64 ...
        delta_genes                (chain, draw, delta_genes_dim_0, delta_genes_dim_1) float64 ...
        ...                         ...
        mu_a                       (chain, draw, gene) float64 ...
        b                          (chain, draw, gene) float64 ...
        d                          (chain, draw, gene) float64 ...
        f                          (chain, draw, gene) float64 ...
        h                          (chain, draw, gene, cancer_gene) float64 ...
        a                          (chain, draw, sgrna) float64 ...
    Attributes:
        created_at:           2022-07-25 17:29:16.063991
        arviz_version:        0.12.1
        previous_created_at:  ['2022-07-25 17:29:16.063991', '2022-07-25T21:25:43...

    --------------------------------------------------------------------------------

    SAMPLE STATS

    <xarray.Dataset>
    Dimensions:          (chain: 4, draw: 1000)
    Coordinates:
      * chain            (chain) int64 0 1 2 3
      * draw             (draw) int64 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
    Data variables:
        acceptance_rate  (chain, draw) float64 ...
        step_size        (chain, draw) float64 ...
        diverging        (chain, draw) bool ...
        energy           (chain, draw) float64 ...
        n_steps          (chain, draw) int64 ...
        tree_depth       (chain, draw) int64 ...
        lp               (chain, draw) float64 ...
    Attributes:
        created_at:           2022-07-25 17:29:16.063991
        arviz_version:        0.12.1
        previous_created_at:  ['2022-07-25 17:29:16.063991', '2022-07-25T21:25:43...

    --------------------------------------------------------------------------------

    MCMC DESCRIPTION

    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.649, 0.624, 1.298, 1.889
    avg. step size: 0.021, 0.021, 0.0, 0.0


### Load posterior summary


```python
prostate_post_summary = pd.read_csv(saved_model_dir / "posterior-summary.csv").assign(
    var_name=lambda d: [x.split("[")[0] for x in d["parameter"]]
)
prostate_post_summary.head()
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
      <th>parameter</th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>var_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mu_mu_a</td>
      <td>0.099</td>
      <td>0.020</td>
      <td>0.077</td>
      <td>0.121</td>
      <td>0.010</td>
      <td>0.008</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>2.14</td>
      <td>mu_mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_b</td>
      <td>0.001</td>
      <td>0.003</td>
      <td>-0.003</td>
      <td>0.004</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5.0</td>
      <td>28.0</td>
      <td>2.10</td>
      <td>mu_b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_d</td>
      <td>-0.020</td>
      <td>0.001</td>
      <td>-0.022</td>
      <td>-0.018</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>6.0</td>
      <td>48.0</td>
      <td>2.01</td>
      <td>mu_d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mu_f</td>
      <td>0.067</td>
      <td>0.077</td>
      <td>0.011</td>
      <td>0.201</td>
      <td>0.038</td>
      <td>0.029</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>2.87</td>
      <td>mu_f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mu_h[ZFHX3]</td>
      <td>-0.024</td>
      <td>0.017</td>
      <td>-0.043</td>
      <td>-0.006</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>2.24</td>
      <td>mu_h</td>
    </tr>
  </tbody>
</table>
</div>



### Load trace object


```python
trace_file = saved_model_dir / "posterior.netcdf"
assert trace_file.exists()
trace = az.from_netcdf(trace_file)
```

### Prostate data


```python
def _broad_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["screen"] == "broad"].reset_index(drop=True)


prostate_dm = CrisprScreenDataManager(
    modeling_data_dir() / "lineage-modeling-data" / "depmap-modeling-data_prostate.csv",
    transformations=[_broad_only],
)
```


```python
prostate_data = prostate_dm.get_data(read_kwargs={"low_memory": False})
prostate_data.head()
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
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>p_dna_batch</th>
      <th>genome_alignment</th>
      <th>hugo_symbol</th>
      <th>screen</th>
      <th>multiple_hits_on_gene</th>
      <th>sgrna_target_chr</th>
      <th>sgrna_target_pos</th>
      <th>...</th>
      <th>any_deleterious</th>
      <th>any_tcga_hotspot</th>
      <th>any_cosmic_hotspot</th>
      <th>is_mutated</th>
      <th>copy_number</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>primary_or_metastasis</th>
      <th>is_male</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAGCCCAGGAGTATGGGAG</td>
      <td>Vcap-304Cas9_RepA_p4_batch3</td>
      <td>0.246450</td>
      <td>3</td>
      <td>chr2_130522105_-</td>
      <td>CFC1B</td>
      <td>broad</td>
      <td>True</td>
      <td>2</td>
      <td>130522105</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.999455</td>
      <td>prostate</td>
      <td>prostate_adenocarcinoma</td>
      <td>metastasis</td>
      <td>True</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAATCAGAGAAACCTGAACG</td>
      <td>Vcap-304Cas9_RepA_p4_batch3</td>
      <td>0.626518</td>
      <td>3</td>
      <td>chr11_89916950_-</td>
      <td>TRIM49D1</td>
      <td>broad</td>
      <td>True</td>
      <td>11</td>
      <td>89916950</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.281907</td>
      <td>prostate</td>
      <td>prostate_adenocarcinoma</td>
      <td>metastasis</td>
      <td>True</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACGTCTTTGAAGAAAGCTG</td>
      <td>Vcap-304Cas9_RepA_p4_batch3</td>
      <td>0.165114</td>
      <td>3</td>
      <td>chr5_71055421_-</td>
      <td>GTF2H2</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>71055421</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.616885</td>
      <td>prostate</td>
      <td>prostate_adenocarcinoma</td>
      <td>metastasis</td>
      <td>True</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACGTCTTTGAAGGAAGCTG</td>
      <td>Vcap-304Cas9_RepA_p4_batch3</td>
      <td>-0.094688</td>
      <td>3</td>
      <td>chr5_69572480_+</td>
      <td>GTF2H2C</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>69572480</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.616885</td>
      <td>prostate</td>
      <td>prostate_adenocarcinoma</td>
      <td>metastasis</td>
      <td>True</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAGAGGTTCCAGACTACTTA</td>
      <td>Vcap-304Cas9_RepA_p4_batch3</td>
      <td>0.294496</td>
      <td>3</td>
      <td>chrX_155898173_+</td>
      <td>VAMP7</td>
      <td>broad</td>
      <td>True</td>
      <td>X</td>
      <td>155898173</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.615935</td>
      <td>prostate</td>
      <td>prostate_adenocarcinoma</td>
      <td>metastasis</td>
      <td>True</td>
      <td>59.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



### Single lineage model


```python
prostate_model = LineageHierNegBinomModel(lineage="prostate")
```


```python
valid_prostate_data = prostate_model.data_processing_pipeline(prostate_data.copy())
prostate_mdl_data = prostate_model.make_data_structure(valid_prostate_data)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/27/22 09:24:47] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Processing data for modeling.     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#269" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">269</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> LFC limits: <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-5.0</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.0</span><span style="font-weight: bold">)</span>           <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#270" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">270</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/27/22 09:26:10] </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> number of data points dropped: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>  <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#321" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">321</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/27/22 09:26:11] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> number of genes mutated in all    <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#470" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">470</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span>         cells lines: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>                    <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">                              </span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Dropping <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span> cancer genes.          <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#526" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">526</span></a>
</pre>



## Analysis


```python
sns.histplot(x=prostate_post_summary["r_hat"], binwidth=0.01, stat="proportion");
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_20_0.png)




```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=prostate_post_summary, x="var_name", y="r_hat", ax=ax)
ax.tick_params(rotation=90)
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_21_0.png)




```python
az.plot_energy(trace);
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_22_0.png)




```python
energy = trace.sample_stats.energy.values
marginal_e = pd.DataFrame((energy - energy.mean(axis=1)[:, None]).T).assign(
    energy="marginal"
)
transition_e = pd.DataFrame((energy[:, :-1] - energy[:, 1:]).T).assign(
    energy="transition"
)
energy_df = pd.concat([marginal_e, transition_e]).reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for i, ax in enumerate(axes.flatten()):
    sns.kdeplot(data=energy_df, x=i, hue="energy", ax=ax)
    ax.set_title(f"chain {i}")
    ax.set_xlabel(None)

fig.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_23_0.png)




```python
HAPPY_CHAINS = [0, 1]
UNHAPPY_CHAINS = [2, 3]

trace.posterior = trace.posterior.drop_sel(chain=UNHAPPY_CHAINS)
trace.sample_stats = trace.sample_stats.drop_sel(chain=UNHAPPY_CHAINS)
trace.posterior_predictive = trace.posterior_predictive.drop_sel(chain=UNHAPPY_CHAINS)
```


```python
az.plot_energy(trace);
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_25_0.png)




```python
az.plot_trace(
    trace, var_names=["mu_mu_a", "mu_b", "mu_d", "mu_f", "mu_h"], compact=False
)
plt.tight_layout()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_26_0.png)




```python
az.plot_trace(trace, var_names=["^sigma_*"], filter_vars="regex", compact=False)
plt.tight_layout()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_27_0.png)




```python
az.plot_forest(
    trace, var_names=["^sigma_*"], filter_vars="regex", combined=True, figsize=(5, 3)
)
plt.tight_layout()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_28_0.png)




```python
var_names = ["a", "mu_a", "b", "d", "f", "h"]
_, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True)
for ax, var_name in zip(axes.flatten(), var_names):
    x = prostate_post_summary.query(f"var_name == '{var_name}'")["mean"]
    sns.kdeplot(x=x, ax=ax)
    ax.set_title(var_name)
    ax.set_xlim(-2, 1)

plt.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_29_0.png)




```python
sgrna_to_gene_map = (
    prostate_data.copy()[["hugo_symbol", "sgrna"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
```


```python
az.summary(trace, var_names="mu_a").sort_values("mean").pipe(head_tail, n=5)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_a[KIF11]</th>
      <td>-1.222</td>
      <td>0.101</td>
      <td>-1.359</td>
      <td>-1.037</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2054.0</td>
      <td>1398.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[HSPE1]</th>
      <td>-1.045</td>
      <td>0.097</td>
      <td>-1.206</td>
      <td>-0.888</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2391.0</td>
      <td>1443.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[SPC24]</th>
      <td>-1.040</td>
      <td>0.102</td>
      <td>-1.205</td>
      <td>-0.883</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2390.0</td>
      <td>1609.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[RAN]</th>
      <td>-1.021</td>
      <td>0.092</td>
      <td>-1.176</td>
      <td>-0.881</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2320.0</td>
      <td>1607.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[EEF2]</th>
      <td>-1.017</td>
      <td>0.097</td>
      <td>-1.181</td>
      <td>-0.876</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2073.0</td>
      <td>1538.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[PRAMEF4]</th>
      <td>0.394</td>
      <td>0.095</td>
      <td>0.245</td>
      <td>0.548</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>1904.0</td>
      <td>1637.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[ZNF334]</th>
      <td>0.394</td>
      <td>0.103</td>
      <td>0.237</td>
      <td>0.561</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>2323.0</td>
      <td>1810.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[ARHGAP44]</th>
      <td>0.399</td>
      <td>0.091</td>
      <td>0.260</td>
      <td>0.549</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2032.0</td>
      <td>1785.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[HLA-DQB1]</th>
      <td>0.412</td>
      <td>0.102</td>
      <td>0.251</td>
      <td>0.574</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2487.0</td>
      <td>1477.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_a[TP53]</th>
      <td>0.447</td>
      <td>0.089</td>
      <td>0.314</td>
      <td>0.595</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2038.0</td>
      <td>1420.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(trace, var_names="b").sort_values("mean").pipe(head_tail, n=5)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b[EP300]</th>
      <td>-0.320</td>
      <td>0.042</td>
      <td>-0.384</td>
      <td>-0.255</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3058.0</td>
      <td>1299.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>b[TP63]</th>
      <td>-0.180</td>
      <td>0.041</td>
      <td>-0.251</td>
      <td>-0.119</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5515.0</td>
      <td>1715.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[STAG2]</th>
      <td>-0.175</td>
      <td>0.039</td>
      <td>-0.233</td>
      <td>-0.110</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4220.0</td>
      <td>1604.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[EBP]</th>
      <td>-0.166</td>
      <td>0.047</td>
      <td>-0.243</td>
      <td>-0.094</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4558.0</td>
      <td>1411.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[TADA1]</th>
      <td>-0.163</td>
      <td>0.040</td>
      <td>-0.229</td>
      <td>-0.098</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5474.0</td>
      <td>1449.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[ATP6V1F]</th>
      <td>0.219</td>
      <td>0.043</td>
      <td>0.155</td>
      <td>0.291</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5096.0</td>
      <td>1408.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[MRPL57]</th>
      <td>0.223</td>
      <td>0.043</td>
      <td>0.154</td>
      <td>0.292</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4616.0</td>
      <td>1486.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[GPI]</th>
      <td>0.239</td>
      <td>0.044</td>
      <td>0.171</td>
      <td>0.312</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>4311.0</td>
      <td>1504.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[AIFM1]</th>
      <td>0.250</td>
      <td>0.045</td>
      <td>0.173</td>
      <td>0.319</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3708.0</td>
      <td>1408.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>b[NARS2]</th>
      <td>0.252</td>
      <td>0.039</td>
      <td>0.193</td>
      <td>0.316</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5685.0</td>
      <td>1556.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(trace, var_names="d").sort_values("mean").pipe(head_tail, n=5)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d[CHMP3]</th>
      <td>-0.257</td>
      <td>0.035</td>
      <td>-0.310</td>
      <td>-0.198</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>4968.0</td>
      <td>1452.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[ATP1A1]</th>
      <td>-0.245</td>
      <td>0.041</td>
      <td>-0.310</td>
      <td>-0.181</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4885.0</td>
      <td>1378.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[LONP1]</th>
      <td>-0.209</td>
      <td>0.040</td>
      <td>-0.275</td>
      <td>-0.145</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4753.0</td>
      <td>1133.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[TRIT1]</th>
      <td>-0.209</td>
      <td>0.040</td>
      <td>-0.270</td>
      <td>-0.142</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4994.0</td>
      <td>1335.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[UBE2N]</th>
      <td>-0.206</td>
      <td>0.038</td>
      <td>-0.265</td>
      <td>-0.143</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4603.0</td>
      <td>1518.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[HCCS]</th>
      <td>0.121</td>
      <td>0.043</td>
      <td>0.051</td>
      <td>0.187</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>5114.0</td>
      <td>1574.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[ENO1]</th>
      <td>0.122</td>
      <td>0.038</td>
      <td>0.060</td>
      <td>0.183</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4798.0</td>
      <td>1430.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[TARS2]</th>
      <td>0.124</td>
      <td>0.039</td>
      <td>0.055</td>
      <td>0.180</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>5994.0</td>
      <td>1017.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[MRPL39]</th>
      <td>0.129</td>
      <td>0.041</td>
      <td>0.066</td>
      <td>0.196</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>3744.0</td>
      <td>1439.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d[DMAC1]</th>
      <td>0.133</td>
      <td>0.041</td>
      <td>0.063</td>
      <td>0.195</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>5069.0</td>
      <td>1532.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(trace, var_names="h").sort_values("mean").pipe(head_tail, n=5)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h[KIF11, ZFHX3]</th>
      <td>-0.796</td>
      <td>0.090</td>
      <td>-0.944</td>
      <td>-0.657</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3629.0</td>
      <td>1434.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>h[ELL, ZFHX3]</th>
      <td>-0.791</td>
      <td>0.080</td>
      <td>-0.912</td>
      <td>-0.657</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4479.0</td>
      <td>1457.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[TRNT1, ZFHX3]</th>
      <td>-0.766</td>
      <td>0.082</td>
      <td>-0.888</td>
      <td>-0.633</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3645.0</td>
      <td>1476.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[LONP1, ZFHX3]</th>
      <td>-0.753</td>
      <td>0.086</td>
      <td>-0.888</td>
      <td>-0.616</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3525.0</td>
      <td>1384.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[ATP6V1B2, ZFHX3]</th>
      <td>-0.742</td>
      <td>0.078</td>
      <td>-0.867</td>
      <td>-0.623</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3487.0</td>
      <td>1679.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[CHMP1B, ZFHX3]</th>
      <td>0.194</td>
      <td>0.067</td>
      <td>0.088</td>
      <td>0.303</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3437.0</td>
      <td>1805.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[AFF4, ZFHX3]</th>
      <td>0.218</td>
      <td>0.073</td>
      <td>0.104</td>
      <td>0.332</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3646.0</td>
      <td>1630.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[ELOA, ZFHX3]</th>
      <td>0.238</td>
      <td>0.071</td>
      <td>0.122</td>
      <td>0.348</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3578.0</td>
      <td>1692.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[TP53, ZFHX3]</th>
      <td>0.268</td>
      <td>0.072</td>
      <td>0.155</td>
      <td>0.381</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3262.0</td>
      <td>1644.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>h[EP300, ZFHX3]</th>
      <td>0.468</td>
      <td>0.073</td>
      <td>0.349</td>
      <td>0.584</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2222.0</td>
      <td>1590.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
example_genes = ["KIF11", "AR", "NF2"]
az.plot_trace(
    trace, var_names=["mu_a", "b", "d"], coords={"gene": example_genes}, compact=False
)
plt.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_35_0.png)




```python
sgrnas_sample = trace.posterior.coords["sgrna"].values[:5]

az.plot_trace(trace, var_names="a", coords={"sgrna": sgrnas_sample}, compact=False)
plt.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_36_0.png)




```python
example_gene = "KIF11"
example_gene_sgrna = sgrna_to_gene_map.query(f"hugo_symbol == '{example_gene}'")[
    "sgrna"
].tolist()
az.plot_forest(
    trace,
    var_names=[
        "mu_mu_a",
        "mu_a",
        "a",
        "mu_b",
        "b",
        "mu_d",
        "d",
        "mu_f",
        "f",
        "mu_h",
        "h",
    ],
    coords={"gene": [example_gene], "sgrna": example_gene_sgrna},
    combined=False,
    figsize=(6, 7),
)
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_37_0.png)




```python
prostate_post_summary.filter_string("var_name", "^sigma_*")
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
      <th>parameter</th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>var_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>sigma_a</td>
      <td>0.106</td>
      <td>0.106</td>
      <td>0.000</td>
      <td>0.212</td>
      <td>0.053</td>
      <td>0.040</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>2.23</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sigma_mu_a</td>
      <td>0.118</td>
      <td>0.118</td>
      <td>0.000</td>
      <td>0.237</td>
      <td>0.059</td>
      <td>0.045</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>2.23</td>
      <td>sigma_mu_a</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sigma_b</td>
      <td>0.029</td>
      <td>0.029</td>
      <td>0.000</td>
      <td>0.058</td>
      <td>0.014</td>
      <td>0.011</td>
      <td>5.0</td>
      <td>32.0</td>
      <td>2.10</td>
      <td>sigma_b</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sigma_d</td>
      <td>0.025</td>
      <td>0.025</td>
      <td>0.000</td>
      <td>0.051</td>
      <td>0.013</td>
      <td>0.010</td>
      <td>6.0</td>
      <td>51.0</td>
      <td>1.75</td>
      <td>sigma_d</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sigma_f</td>
      <td>0.303</td>
      <td>0.325</td>
      <td>0.086</td>
      <td>0.865</td>
      <td>0.162</td>
      <td>0.124</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>2.74</td>
      <td>sigma_f</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sigma_h[0]</td>
      <td>0.099</td>
      <td>0.066</td>
      <td>0.007</td>
      <td>0.163</td>
      <td>0.033</td>
      <td>0.025</td>
      <td>5.0</td>
      <td>11.0</td>
      <td>2.74</td>
      <td>sigma_h</td>
    </tr>
  </tbody>
</table>
</div>




```python
eg_gene = trace.posterior.coords["gene"].values[0]

for gene in [eg_gene, "KIF11"]:
    axes = az.plot_pair(
        trace,
        var_names=["mu_a", "b", "d"],
        coords={"gene": [gene]},
        figsize=(7, 7),
        scatter_kwargs={"alpha": 0.2},
    )
    for ax in axes.flatten():
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
    plt.tight_layout()
    plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_39_0.png)





![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_39_1.png)




```python
def _get_average_per_chain(trace: az.InferenceData, var_name: str) -> pd.DataFrame:
    return (
        trace.posterior[var_name]
        .mean(axis=(1))
        .to_dataframe()
        .reset_index()
        .astype({"chain": str})
    )


mu_a_post_avg = _get_average_per_chain(trace, "mu_a")
b_post_avg = _get_average_per_chain(trace, "b")
d_post_avg = _get_average_per_chain(trace, "d")

gene_post_avg = mu_a_post_avg.merge(b_post_avg, on=["chain", "gene"]).merge(
    d_post_avg, on=["chain", "gene"]
)


fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(7, 3.5))

ax = axes[0]
sns.scatterplot(
    data=gene_post_avg,
    x="mu_a",
    y="b",
    hue="chain",
    alpha=0.1,
    edgecolor=None,
    s=5,
    ax=ax,
)
ax.set_xlabel(r"$\mu_a$")
ax.set_ylabel(r"$b$")


ax = axes[1]
sns.scatterplot(
    data=gene_post_avg,
    x="b",
    y="d",
    hue="chain",
    alpha=0.1,
    edgecolor=None,
    s=5,
    ax=ax,
)
ax.set_xlabel(r"$b$")
ax.set_ylabel(r"$d$")

for ax in axes.flatten():
    ax.axhline(color="k")
    ax.axvline(color="k")
    ax.get_legend().remove()


fig.tight_layout()
fig.suptitle("Joint posterior distribution", va="bottom")

plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_40_0.png)




```python
genes_var_names = ["mu_a", "b", "d", "f"]
genes_var_names += [f"h[{g}]" for g in trace.posterior.coords["cancer_gene"].values]
gene_corr_post = (
    az.summary(trace, "genes_chol_cov_corr", kind="stats")
    .pipe(extract_coords_param_names, names=["d1", "d2"])
    .astype({"d1": int, "d2": int})
    .assign(
        p1=lambda d: [genes_var_names[i] for i in d["d1"]],
        p2=lambda d: [genes_var_names[i] for i in d["d2"]],
    )
    .assign(
        p1=lambda d: pd.Categorical(d["p1"], categories=d["p1"].unique(), ordered=True)
    )
    .assign(
        p2=lambda d: pd.Categorical(
            d["p2"], categories=d["p1"].cat.categories, ordered=True
        )
    )
)
gene_corr_post
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>d1</th>
      <th>d2</th>
      <th>p1</th>
      <th>p2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>genes_chol_cov_corr[0, 0]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>0</td>
      <td>0</td>
      <td>mu_a</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 1]</th>
      <td>-0.380</td>
      <td>0.014</td>
      <td>-0.403</td>
      <td>-0.358</td>
      <td>0</td>
      <td>1</td>
      <td>mu_a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 2]</th>
      <td>0.106</td>
      <td>0.016</td>
      <td>0.083</td>
      <td>0.134</td>
      <td>0</td>
      <td>2</td>
      <td>mu_a</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 3]</th>
      <td>-0.476</td>
      <td>0.034</td>
      <td>-0.528</td>
      <td>-0.419</td>
      <td>0</td>
      <td>3</td>
      <td>mu_a</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 4]</th>
      <td>0.843</td>
      <td>0.010</td>
      <td>0.828</td>
      <td>0.859</td>
      <td>0</td>
      <td>4</td>
      <td>mu_a</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 0]</th>
      <td>-0.380</td>
      <td>0.014</td>
      <td>-0.403</td>
      <td>-0.358</td>
      <td>1</td>
      <td>0</td>
      <td>b</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 1]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1</td>
      <td>1</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 2]</th>
      <td>-0.014</td>
      <td>0.021</td>
      <td>-0.050</td>
      <td>0.018</td>
      <td>1</td>
      <td>2</td>
      <td>b</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 3]</th>
      <td>0.366</td>
      <td>0.054</td>
      <td>0.284</td>
      <td>0.454</td>
      <td>1</td>
      <td>3</td>
      <td>b</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 4]</th>
      <td>-0.538</td>
      <td>0.017</td>
      <td>-0.565</td>
      <td>-0.512</td>
      <td>1</td>
      <td>4</td>
      <td>b</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 0]</th>
      <td>0.106</td>
      <td>0.016</td>
      <td>0.083</td>
      <td>0.134</td>
      <td>2</td>
      <td>0</td>
      <td>d</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 1]</th>
      <td>-0.014</td>
      <td>0.021</td>
      <td>-0.050</td>
      <td>0.018</td>
      <td>2</td>
      <td>1</td>
      <td>d</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 2]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>2</td>
      <td>2</td>
      <td>d</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 3]</th>
      <td>-0.102</td>
      <td>0.049</td>
      <td>-0.181</td>
      <td>-0.024</td>
      <td>2</td>
      <td>3</td>
      <td>d</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 4]</th>
      <td>0.123</td>
      <td>0.017</td>
      <td>0.093</td>
      <td>0.148</td>
      <td>2</td>
      <td>4</td>
      <td>d</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 0]</th>
      <td>-0.476</td>
      <td>0.034</td>
      <td>-0.528</td>
      <td>-0.419</td>
      <td>3</td>
      <td>0</td>
      <td>f</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 1]</th>
      <td>0.366</td>
      <td>0.054</td>
      <td>0.284</td>
      <td>0.454</td>
      <td>3</td>
      <td>1</td>
      <td>f</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 2]</th>
      <td>-0.102</td>
      <td>0.049</td>
      <td>-0.181</td>
      <td>-0.024</td>
      <td>3</td>
      <td>2</td>
      <td>f</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 3]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>3</td>
      <td>3</td>
      <td>f</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 4]</th>
      <td>-0.664</td>
      <td>0.028</td>
      <td>-0.705</td>
      <td>-0.617</td>
      <td>3</td>
      <td>4</td>
      <td>f</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 0]</th>
      <td>0.843</td>
      <td>0.010</td>
      <td>0.828</td>
      <td>0.859</td>
      <td>4</td>
      <td>0</td>
      <td>h[ZFHX3]</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 1]</th>
      <td>-0.538</td>
      <td>0.017</td>
      <td>-0.565</td>
      <td>-0.512</td>
      <td>4</td>
      <td>1</td>
      <td>h[ZFHX3]</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 2]</th>
      <td>0.123</td>
      <td>0.017</td>
      <td>0.093</td>
      <td>0.148</td>
      <td>4</td>
      <td>2</td>
      <td>h[ZFHX3]</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 3]</th>
      <td>-0.664</td>
      <td>0.028</td>
      <td>-0.705</td>
      <td>-0.617</td>
      <td>4</td>
      <td>3</td>
      <td>h[ZFHX3]</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 4]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>4</td>
      <td>4</td>
      <td>h[ZFHX3]</td>
      <td>h[ZFHX3]</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_df = gene_corr_post.pivot_wider("p1", "p2", "mean").set_index("p1")
sns.heatmap(plot_df, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_42_0.png)




```python
cancer_genes = trace.posterior.coords["cancer_gene"].values.tolist()
cancer_gene_mutants = (
    valid_prostate_data.filter_column_isin("hugo_symbol", cancer_genes)[
        ["hugo_symbol", "depmap_id", "is_mutated"]
    ]
    .drop_duplicates()
    .assign(is_mutated=lambda d: d["is_mutated"].map({True: "X", False: ""}))
    .pivot_wider("depmap_id", names_from="hugo_symbol", values_from="is_mutated")
    .set_index("depmap_id")
)
cancer_gene_mutants
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
      <th>ZFHX3</th>
    </tr>
    <tr>
      <th>depmap_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACH-000115</th>
      <td></td>
    </tr>
    <tr>
      <th>ACH-000977</th>
      <td>X</td>
    </tr>
    <tr>
      <th>ACH-001453</th>
      <td></td>
    </tr>
    <tr>
      <th>ACH-001627</th>
      <td>X</td>
    </tr>
    <tr>
      <th>ACH-001648</th>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(trace, var_names=["mu_h", "sigma_h"], compact=False)
plt.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_44_0.png)




```python
h_post_summary = az.summary(trace, var_names=["h"], kind="stats").pipe(
    extract_coords_param_names, names=["hugo_symbol", "cancer_gene"]
)
h_post_summary.head()
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>hugo_symbol</th>
      <th>cancer_gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h[A1BG, ZFHX3]</th>
      <td>0.034</td>
      <td>0.083</td>
      <td>-0.099</td>
      <td>0.164</td>
      <td>A1BG</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[A1CF, ZFHX3]</th>
      <td>0.083</td>
      <td>0.074</td>
      <td>-0.036</td>
      <td>0.198</td>
      <td>A1CF</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[A2M, ZFHX3]</th>
      <td>0.070</td>
      <td>0.084</td>
      <td>-0.062</td>
      <td>0.207</td>
      <td>A2M</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[A2ML1, ZFHX3]</th>
      <td>-0.043</td>
      <td>0.074</td>
      <td>-0.160</td>
      <td>0.077</td>
      <td>A2ML1</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[A3GALT2, ZFHX3]</th>
      <td>-0.012</td>
      <td>0.071</td>
      <td>-0.121</td>
      <td>0.108</td>
      <td>A3GALT2</td>
      <td>ZFHX3</td>
    </tr>
  </tbody>
</table>
</div>




```python
h_post_summary.sort_values("mean").pipe(head_tail, n=5)
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>hugo_symbol</th>
      <th>cancer_gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h[KIF11, ZFHX3]</th>
      <td>-0.796</td>
      <td>0.090</td>
      <td>-0.944</td>
      <td>-0.657</td>
      <td>KIF11</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[ELL, ZFHX3]</th>
      <td>-0.791</td>
      <td>0.080</td>
      <td>-0.912</td>
      <td>-0.657</td>
      <td>ELL</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[TRNT1, ZFHX3]</th>
      <td>-0.766</td>
      <td>0.082</td>
      <td>-0.888</td>
      <td>-0.633</td>
      <td>TRNT1</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[LONP1, ZFHX3]</th>
      <td>-0.753</td>
      <td>0.086</td>
      <td>-0.888</td>
      <td>-0.616</td>
      <td>LONP1</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[ATP6V1B2, ZFHX3]</th>
      <td>-0.742</td>
      <td>0.078</td>
      <td>-0.867</td>
      <td>-0.623</td>
      <td>ATP6V1B2</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[CHMP1B, ZFHX3]</th>
      <td>0.194</td>
      <td>0.067</td>
      <td>0.088</td>
      <td>0.303</td>
      <td>CHMP1B</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[AFF4, ZFHX3]</th>
      <td>0.218</td>
      <td>0.073</td>
      <td>0.104</td>
      <td>0.332</td>
      <td>AFF4</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[ELOA, ZFHX3]</th>
      <td>0.238</td>
      <td>0.071</td>
      <td>0.122</td>
      <td>0.348</td>
      <td>ELOA</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[TP53, ZFHX3]</th>
      <td>0.268</td>
      <td>0.072</td>
      <td>0.155</td>
      <td>0.381</td>
      <td>TP53</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>h[EP300, ZFHX3]</th>
      <td>0.468</td>
      <td>0.073</td>
      <td>0.349</td>
      <td>0.584</td>
      <td>EP300</td>
      <td>ZFHX3</td>
    </tr>
  </tbody>
</table>
</div>




```python
mu_h_post_summary = az.summary(trace, var_names="mu_h", kind="stats").pipe(
    extract_coords_param_names, names=["cancer_gene"]
)
mu_h_post_summary
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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>cancer_gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_h[ZFHX3]</th>
      <td>-0.041</td>
      <td>0.002</td>
      <td>-0.044</td>
      <td>-0.039</td>
      <td>ZFHX3</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.kdeplot(data=h_post_summary, x="mean", hue="cancer_gene")
ax.set_xlabel(r"$\bar{h}_g$ posterior")
ax.set_ylabel("density")
ax.get_legend().set_title("cancer gene\ncomut.")
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_48_0.png)




```python
fig, axes = plt.subplots(
    len(cancer_genes), 1, squeeze=False, figsize=(8, len(cancer_genes) * 5)
)
for ax, cg in zip(axes.flatten(), cancer_genes):
    h_hits = (
        h_post_summary.filter_column_isin("cancer_gene", [cg])
        .sort_values("mean")
        .pipe(head_tail, n=5)["hugo_symbol"]
        .tolist()
    )

    h_hits_data = (
        valid_prostate_data.filter_column_isin("hugo_symbol", h_hits)
        .merge(cancer_gene_mutants.reset_index(), on="depmap_id")
        .reset_index()
        .astype({"hugo_symbol": str})
        .assign(
            hugo_symbol=lambda d: pd.Categorical(d["hugo_symbol"], categories=h_hits),
            _cg_mut=lambda d: d[cg].map({"X": "mut.", "": "WT"}),
        )
    )
    mut_pal = {"mut.": "tab:red", "WT": "gray"}
    sns.boxplot(
        data=h_hits_data,
        x="hugo_symbol",
        y="lfc",
        hue="_cg_mut",
        palette=mut_pal,
        ax=ax,
        showfliers=False,
        boxprops={"alpha": 0.5},
    )
    sns.stripplot(
        data=h_hits_data,
        x="hugo_symbol",
        y="lfc",
        hue="_cg_mut",
        dodge=True,
        palette=mut_pal,
        ax=ax,
    )
    ax.get_legend().set_title(cg)

plt.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_49_0.png)




```python
n_examples = 40
n_chains, n_draws, n_data = trace.posterior_predictive["ct_final"].shape
ex_draws_idx = np.random.choice(
    np.arange(n_draws), size=n_examples // n_chains, replace=False
)
example_ppc_draws = trace.posterior_predictive["ct_final"][
    :, ex_draws_idx, :
].values.reshape(-1, n_data)
example_ppc_draws.shape
```




    (40, 355308)




```python
fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharex=False, sharey=False)
ax1 = axes[0]
ax2 = axes[1]

pp_avg = trace.posterior_predictive["ct_final"].mean(axis=(0, 1))

for i in range(example_ppc_draws.shape[0]):
    sns.kdeplot(
        x=np.log10(example_ppc_draws[i, :] + 1), alpha=0.2, color="tab:blue", ax=ax1
    )

sns.kdeplot(x=np.log10(pp_avg + 1), color="tab:orange", ax=ax1)
sns.kdeplot(x=np.log10(valid_prostate_data["counts_final"] + 1), color="k", ax=ax1)
ax1.set_xlabel("log10(counts final + 1)")
ax1.set_ylabel("density")


for i in range(example_ppc_draws.shape[0]):
    sns.kdeplot(x=example_ppc_draws[i, :], alpha=0.2, color="tab:blue", ax=ax2)

sns.kdeplot(x=pp_avg, color="tab:orange", ax=ax2)
sns.kdeplot(x=valid_prostate_data["counts_final"], color="k", ax=ax2)
ax2.set_xlabel("counts final")
ax2.set_ylabel("density")
ax2.set_xlim(0, 1000)

fig.suptitle("PPC")
fig.tight_layout()
plt.show()
```



![png](023_single-lineage-prostate-inspection_002_files/023_single-lineage-prostate-inspection_002_51_0.png)



---

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 21.09 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-27

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.45.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-162.o2.rc.hms.harvard.edu

    Git branch: simplify

    seaborn   : 0.11.2
    qnorm     : 0.8.1
    numpy     : 1.22.4
    arviz     : 0.12.1
    matplotlib: 3.5.2
    pandas    : 1.4.3
