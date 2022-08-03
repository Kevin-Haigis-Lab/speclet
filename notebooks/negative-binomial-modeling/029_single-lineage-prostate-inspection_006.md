# Inspect the single-lineage model run on the prostate data (006)

 Model attributes:

- sgRNA | gene varying intercept
- RNA and CN varying effects per gene
- correlation between gene varying effects modeled using the multivariate normal and Cholesky decomposition (non-centered parameterization)
- target gene mutation variable and cancer gene comutation variable.
- varying effect for cell line


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
import seaborn as sns
```


```python
from speclet.analysis.arviz_analysis import extract_coords_param_names
from speclet.bayesian_models.lineage_hierarchical_nb import LineageHierNegBinomModel
from speclet.data_processing.common import head_tail
from speclet.io import modeling_data_dir, models_dir
from speclet.managers.data_managers import CrisprScreenDataManager, broad_only
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
saved_model_dir = models_dir() / "hnb-single-lineage-prostate-006_PYMC_NUMPYRO"
```


```python
with open(saved_model_dir / "description.txt") as f:
    model_description = "".join(list(f))

print(model_description)
```

    name: 'hnb-single-lineage-prostate-006'
    fit method: 'PYMC_NUMPYRO'

    --------------------------------------------------------------------------------

    CONFIGURATION

    {
        "name": "hnb-single-lineage-prostate-006",
        "description": " Single lineage hierarchical negative binomial model for prostate data from the Broad. Added a variable for cell line varying effect. (Skipped 005.) ",
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
                "target_accept": 0.98,
                "progress_bar": true,
                "chain_method": "parallel",
                "postprocessing_backend": "cpu",
                "idata_kwargs": {
                    "log_likelihood": false
                },
                "nuts_kwargs": {
                    "step_size": 0.01
                }
            }
        }
    }

    --------------------------------------------------------------------------------

    POSTERIOR

    <xarray.Dataset>
    Dimensions:                    (chain: 4, draw: 1000, delta_genes_dim_0: 5,
                                    delta_genes_dim_1: 18119, sgrna: 71062,
                                    cell_line: 5, genes_chol_cov_dim_0: 15,
                                    genes_chol_cov_corr_dim_0: 5,
                                    genes_chol_cov_corr_dim_1: 5,
                                    genes_chol_cov_stds_dim_0: 5, cancer_gene: 1,
                                    gene: 18119)
    Coordinates:
      * chain                      (chain) int64 0 1 2 3
      * draw                       (draw) int64 0 1 2 3 4 5 ... 995 996 997 998 999
      * delta_genes_dim_0          (delta_genes_dim_0) int64 0 1 2 3 4
      * delta_genes_dim_1          (delta_genes_dim_1) int64 0 1 2 ... 18117 18118
      * sgrna                      (sgrna) object 'AAAAAAATCCAGCAATGCAG' ... 'TTT...
      * cell_line                  (cell_line) object 'ACH-000115' ... 'ACH-001648'
      * genes_chol_cov_dim_0       (genes_chol_cov_dim_0) int64 0 1 2 3 ... 12 13 14
      * genes_chol_cov_corr_dim_0  (genes_chol_cov_corr_dim_0) int64 0 1 2 3 4
      * genes_chol_cov_corr_dim_1  (genes_chol_cov_corr_dim_1) int64 0 1 2 3 4
      * genes_chol_cov_stds_dim_0  (genes_chol_cov_stds_dim_0) int64 0 1 2 3 4
      * cancer_gene                (cancer_gene) object 'ZFHX3'
      * gene                       (gene) object 'A1BG' 'A1CF' ... 'ZZEF1' 'ZZZ3'
    Data variables: (12/24)
        mu_mu_a                    (chain, draw) float64 ...
        mu_b                       (chain, draw) float64 ...
        mu_d                       (chain, draw) float64 ...
        delta_genes                (chain, draw, delta_genes_dim_0, delta_genes_dim_1) float64 ...
        delta_a                    (chain, draw, sgrna) float64 ...
        delta_k                    (chain, draw, cell_line) float64 ...
        ...                         ...
        b                          (chain, draw, gene) float64 ...
        d                          (chain, draw, gene) float64 ...
        f                          (chain, draw, gene) float64 ...
        h                          (chain, draw, gene, cancer_gene) float64 ...
        a                          (chain, draw, sgrna) float64 ...
        k                          (chain, draw, cell_line) float64 ...
    Attributes:
        created_at:           2022-08-03 10:04:17.925894
        arviz_version:        0.12.1
        previous_created_at:  ['2022-08-03 10:04:17.925894', '2022-08-03T12:45:45...

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
        created_at:           2022-08-03 10:04:17.925894
        arviz_version:        0.12.1
        previous_created_at:  ['2022-08-03 10:04:17.925894', '2022-08-03T12:45:45...

    --------------------------------------------------------------------------------

    MCMC DESCRIPTION

    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.659, 2.062, 0.705, 0.639
    avg. step size: 0.009, 0.0, 0.012, 0.01


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
      <td>0.079</td>
      <td>0.029</td>
      <td>0.048</td>
      <td>0.119</td>
      <td>0.012</td>
      <td>0.009</td>
      <td>8.0</td>
      <td>1549.0</td>
      <td>1.44</td>
      <td>mu_mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_b</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>mu_b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_d</td>
      <td>-0.020</td>
      <td>0.002</td>
      <td>-0.021</td>
      <td>-0.018</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7.0</td>
      <td>2911.0</td>
      <td>1.52</td>
      <td>mu_d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sigma_a</td>
      <td>0.159</td>
      <td>0.092</td>
      <td>0.000</td>
      <td>0.213</td>
      <td>0.046</td>
      <td>0.035</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sigma_k</td>
      <td>0.029</td>
      <td>0.025</td>
      <td>0.000</td>
      <td>0.054</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>sigma_k</td>
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
prostate_dm = CrisprScreenDataManager(
    modeling_data_dir() / "lineage-modeling-data" / "depmap-modeling-data_prostate.csv",
    transformations=[broad_only],
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

    [INFO] 2022-08-03 13:24:20 [(lineage_hierarchical_nb.py:data_processing_pipeline:274] Processing data for modeling.
    [INFO] 2022-08-03 13:24:20 [(lineage_hierarchical_nb.py:data_processing_pipeline:275] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-08-03 13:25:23 [(lineage_hierarchical_nb.py:data_processing_pipeline:326] number of data points dropped: 2
    [INFO] 2022-08-03 13:25:23 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:503] number of genes mutated in all cells lines: 0
    [DEBUG] 2022-08-03 13:25:23 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:506] Genes always mutated:
    [INFO] 2022-08-03 13:25:23 [(lineage_hierarchical_nb.py:_trim_cancer_genes:559] Dropping 8 cancer genes.
    [DEBUG] 2022-08-03 13:25:23 [(lineage_hierarchical_nb.py:_trim_cancer_genes:560] Dropped cancer genes: ['AR', 'AXIN1', 'FOXA1', 'KLF6', 'NCOR2', 'PTEN', 'SALL4', 'SPOP']


## Analysis


```python
sns.histplot(x=prostate_post_summary["r_hat"], binwidth=0.01, stat="proportion");
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_20_0.png)




```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=prostate_post_summary, x="var_name", y="r_hat", ax=ax)
ax.tick_params(rotation=90)
plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_21_0.png)




```python
az.plot_energy(trace);
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_22_0.png)




```python
energy = trace.sample_stats.energy.values
marginal_e = pd.DataFrame((energy - energy.mean(axis=1)[:, None]).T).assign(
    energy="marginal"
)
transition_e = pd.DataFrame((energy[:, :-1] - energy[:, 1:]).T).assign(
    energy="transition"
)
energy_df = pd.concat([marginal_e, transition_e]).reset_index(drop=True)
bfmi = az.bfmi(trace)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for i, ax in enumerate(axes.flatten()):
    sns.kdeplot(data=energy_df, x=i, hue="energy", ax=ax)
    ax.set_title(f"chain {i} – BFMI: {bfmi[i]:0.2f}")
    ax.set_xlabel(None)
    xmin, _ = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.get_legend().set_frame_on(False)

fig.tight_layout()
plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_23_0.png)




```python
stats = ["step_size", "n_steps", "tree_depth", "acceptance_rate", "energy"]
trace.sample_stats.get(stats).to_dataframe().groupby("chain").mean()
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
      <th>step_size</th>
      <th>n_steps</th>
      <th>tree_depth</th>
      <th>acceptance_rate</th>
      <th>energy</th>
    </tr>
    <tr>
      <th>chain</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.936947e-03</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.978535</td>
      <td>2.452054e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.829003e-21</td>
      <td>1023.0</td>
      <td>10.0</td>
      <td>0.986086</td>
      <td>2.551388e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.215136e-02</td>
      <td>1023.0</td>
      <td>10.0</td>
      <td>0.969588</td>
      <td>2.452130e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.044637e-02</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.977086</td>
      <td>2.452060e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(trace, var_names=["mu_mu_a", "mu_b", "mu_d"], compact=False)
plt.tight_layout()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")




![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_25_1.png)




```python
az.plot_trace(trace, var_names=["^sigma_*"], filter_vars="regex", compact=False)
plt.tight_layout()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")




![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_26_1.png)




```python
sigmas = ["sigma_mu_a", "sigma_b", "sigma_d", "sigma_f", "sigma_k"]
trace.posterior.get(sigmas).mean(dim="draw").to_dataframe()
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
      <th>sigma_mu_a</th>
      <th>sigma_b</th>
      <th>sigma_d</th>
      <th>sigma_f</th>
      <th>sigma_k</th>
    </tr>
    <tr>
      <th>chain</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.364676e-01</td>
      <td>5.774309e-02</td>
      <td>0.050418</td>
      <td>0.131896</td>
      <td>3.737015e-02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.154574e-19</td>
      <td>1.255465e-22</td>
      <td>0.000353</td>
      <td>0.217349</td>
      <td>1.854613e-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.364556e-01</td>
      <td>5.770427e-02</td>
      <td>0.050426</td>
      <td>0.130979</td>
      <td>3.692001e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.364197e-01</td>
      <td>5.772364e-02</td>
      <td>0.050369</td>
      <td>0.131410</td>
      <td>3.997159e-02</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(trace, var_names=["alpha"], compact=False)
plt.tight_layout()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")




![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_28_1.png)




```python
az.plot_forest(
    trace, var_names=["^sigma_*"], filter_vars="regex", combined=False, figsize=(5, 5)
)
plt.tight_layout()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_29_0.png)




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



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_30_0.png)




```python
sgrna_to_gene_map = (
    prostate_data.copy()[["hugo_symbol", "sgrna"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
```


```python
prostate_post_summary.query("var_name == 'mu_a'").sort_values("mean").pipe(head_tail, 5)
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
      <th>7785</th>
      <td>mu_a[KIF11]</td>
      <td>-0.894</td>
      <td>0.592</td>
      <td>-1.343</td>
      <td>0.119</td>
      <td>0.292</td>
      <td>0.223</td>
      <td>7.0</td>
      <td>1785.0</td>
      <td>1.53</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>6987</th>
      <td>mu_a[HSPE1]</td>
      <td>-0.764</td>
      <td>0.517</td>
      <td>-1.159</td>
      <td>0.119</td>
      <td>0.254</td>
      <td>0.194</td>
      <td>7.0</td>
      <td>1770.0</td>
      <td>1.53</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>14744</th>
      <td>mu_a[SPC24]</td>
      <td>-0.762</td>
      <td>0.516</td>
      <td>-1.161</td>
      <td>0.119</td>
      <td>0.254</td>
      <td>0.194</td>
      <td>7.0</td>
      <td>3007.0</td>
      <td>1.52</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>12653</th>
      <td>mu_a[RAN]</td>
      <td>-0.749</td>
      <td>0.509</td>
      <td>-1.143</td>
      <td>0.119</td>
      <td>0.250</td>
      <td>0.191</td>
      <td>7.0</td>
      <td>1699.0</td>
      <td>1.53</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>13210</th>
      <td>mu_a[RPL12]</td>
      <td>-0.742</td>
      <td>0.505</td>
      <td>-1.133</td>
      <td>0.119</td>
      <td>0.248</td>
      <td>0.190</td>
      <td>7.0</td>
      <td>2359.0</td>
      <td>1.53</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>mu_a[ARMCX4]</td>
      <td>0.315</td>
      <td>0.139</td>
      <td>0.119</td>
      <td>0.480</td>
      <td>0.057</td>
      <td>0.042</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>1.49</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>940</th>
      <td>mu_a[ARHGAP44]</td>
      <td>0.318</td>
      <td>0.141</td>
      <td>0.119</td>
      <td>0.484</td>
      <td>0.058</td>
      <td>0.043</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.49</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>2455</th>
      <td>mu_a[CCNF]</td>
      <td>0.322</td>
      <td>0.148</td>
      <td>0.119</td>
      <td>0.504</td>
      <td>0.059</td>
      <td>0.044</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>1.48</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>6762</th>
      <td>mu_a[HLA-DQB1]</td>
      <td>0.328</td>
      <td>0.151</td>
      <td>0.119</td>
      <td>0.511</td>
      <td>0.061</td>
      <td>0.045</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>1.49</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>16214</th>
      <td>mu_a[TP53]</td>
      <td>0.352</td>
      <td>0.157</td>
      <td>0.119</td>
      <td>0.528</td>
      <td>0.068</td>
      <td>0.051</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.51</td>
      <td>mu_a</td>
    </tr>
  </tbody>
</table>
</div>




```python
prostate_post_summary.query("var_name == 'b'").sort_values("mean").pipe(head_tail, 5)
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
      <th>22902</th>
      <td>b[EP300]</td>
      <td>-0.243</td>
      <td>0.145</td>
      <td>-0.371</td>
      <td>0.000</td>
      <td>0.070</td>
      <td>0.053</td>
      <td>7.0</td>
      <td>2134.0</td>
      <td>1.53</td>
      <td>b</td>
    </tr>
    <tr>
      <th>34344</th>
      <td>b[TP63]</td>
      <td>-0.134</td>
      <td>0.085</td>
      <td>-0.221</td>
      <td>0.000</td>
      <td>0.039</td>
      <td>0.029</td>
      <td>7.0</td>
      <td>2593.0</td>
      <td>1.52</td>
      <td>b</td>
    </tr>
    <tr>
      <th>33098</th>
      <td>b[STAG2]</td>
      <td>-0.131</td>
      <td>0.083</td>
      <td>-0.216</td>
      <td>0.000</td>
      <td>0.038</td>
      <td>0.029</td>
      <td>7.0</td>
      <td>2384.0</td>
      <td>1.52</td>
      <td>b</td>
    </tr>
    <tr>
      <th>23683</th>
      <td>b[FOXA1]</td>
      <td>-0.125</td>
      <td>0.082</td>
      <td>-0.212</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>0.027</td>
      <td>7.0</td>
      <td>2007.0</td>
      <td>1.52</td>
      <td>b</td>
    </tr>
    <tr>
      <th>22614</th>
      <td>b[EBP]</td>
      <td>-0.125</td>
      <td>0.083</td>
      <td>-0.216</td>
      <td>0.000</td>
      <td>0.036</td>
      <td>0.027</td>
      <td>7.0</td>
      <td>2059.0</td>
      <td>1.52</td>
      <td>b</td>
    </tr>
    <tr>
      <th>19386</th>
      <td>b[ATP6V1F]</td>
      <td>0.166</td>
      <td>0.103</td>
      <td>0.000</td>
      <td>0.268</td>
      <td>0.048</td>
      <td>0.036</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>b</td>
    </tr>
    <tr>
      <th>27507</th>
      <td>b[MRPL57]</td>
      <td>0.168</td>
      <td>0.104</td>
      <td>0.000</td>
      <td>0.268</td>
      <td>0.049</td>
      <td>0.037</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>b</td>
    </tr>
    <tr>
      <th>24316</th>
      <td>b[GPI]</td>
      <td>0.181</td>
      <td>0.111</td>
      <td>0.000</td>
      <td>0.288</td>
      <td>0.052</td>
      <td>0.040</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18591</th>
      <td>b[AIFM1]</td>
      <td>0.189</td>
      <td>0.116</td>
      <td>0.000</td>
      <td>0.301</td>
      <td>0.055</td>
      <td>0.042</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>b</td>
    </tr>
    <tr>
      <th>27884</th>
      <td>b[NARS2]</td>
      <td>0.190</td>
      <td>0.115</td>
      <td>0.000</td>
      <td>0.296</td>
      <td>0.055</td>
      <td>0.042</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
prostate_post_summary.query("var_name == 'd'").sort_values("mean").pipe(head_tail, 5)
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
      <th>39174</th>
      <td>d[CHMP3]</td>
      <td>-0.195</td>
      <td>0.107</td>
      <td>-0.292</td>
      <td>-0.018</td>
      <td>0.051</td>
      <td>0.039</td>
      <td>7.0</td>
      <td>2660.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>37445</th>
      <td>d[ATP1A1]</td>
      <td>-0.184</td>
      <td>0.103</td>
      <td>-0.282</td>
      <td>-0.017</td>
      <td>0.048</td>
      <td>0.037</td>
      <td>7.0</td>
      <td>2114.0</td>
      <td>1.53</td>
      <td>d</td>
    </tr>
    <tr>
      <th>52640</th>
      <td>d[TRIT1]</td>
      <td>-0.159</td>
      <td>0.089</td>
      <td>-0.249</td>
      <td>-0.017</td>
      <td>0.041</td>
      <td>0.031</td>
      <td>7.0</td>
      <td>2684.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>44642</th>
      <td>d[LONP1]</td>
      <td>-0.159</td>
      <td>0.089</td>
      <td>-0.248</td>
      <td>-0.017</td>
      <td>0.041</td>
      <td>0.031</td>
      <td>7.0</td>
      <td>1996.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>52955</th>
      <td>d[UBE2N]</td>
      <td>-0.156</td>
      <td>0.087</td>
      <td>-0.243</td>
      <td>-0.018</td>
      <td>0.040</td>
      <td>0.030</td>
      <td>7.0</td>
      <td>1992.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>40992</th>
      <td>d[ENO1]</td>
      <td>0.084</td>
      <td>0.068</td>
      <td>-0.018</td>
      <td>0.159</td>
      <td>0.030</td>
      <td>0.022</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>51556</th>
      <td>d[TARS2]</td>
      <td>0.085</td>
      <td>0.068</td>
      <td>-0.018</td>
      <td>0.161</td>
      <td>0.030</td>
      <td>0.023</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>42786</th>
      <td>d[HCCS]</td>
      <td>0.085</td>
      <td>0.069</td>
      <td>-0.017</td>
      <td>0.164</td>
      <td>0.030</td>
      <td>0.022</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>45608</th>
      <td>d[MRPL39]</td>
      <td>0.091</td>
      <td>0.071</td>
      <td>-0.017</td>
      <td>0.170</td>
      <td>0.031</td>
      <td>0.024</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
    <tr>
      <th>40413</th>
      <td>d[DMAC1]</td>
      <td>0.094</td>
      <td>0.074</td>
      <td>-0.017</td>
      <td>0.175</td>
      <td>0.033</td>
      <td>0.024</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>




```python
prostate_post_summary.query("var_name == 'h'").sort_values("mean").pipe(head_tail, 5)
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
      <th>77155</th>
      <td>h[ELL, ZFHX3]</td>
      <td>-0.569</td>
      <td>0.335</td>
      <td>-0.845</td>
      <td>-0.003</td>
      <td>0.163</td>
      <td>0.125</td>
      <td>7.0</td>
      <td>1636.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
    <tr>
      <th>80261</th>
      <td>h[KIF11, ZFHX3]</td>
      <td>-0.568</td>
      <td>0.331</td>
      <td>-0.847</td>
      <td>-0.011</td>
      <td>0.161</td>
      <td>0.122</td>
      <td>7.0</td>
      <td>2908.0</td>
      <td>1.53</td>
      <td>h</td>
    </tr>
    <tr>
      <th>88900</th>
      <td>h[TRNT1, ZFHX3]</td>
      <td>-0.544</td>
      <td>0.328</td>
      <td>-0.816</td>
      <td>0.011</td>
      <td>0.160</td>
      <td>0.122</td>
      <td>7.0</td>
      <td>2877.0</td>
      <td>1.53</td>
      <td>h</td>
    </tr>
    <tr>
      <th>73737</th>
      <td>h[ATP6V1B2, ZFHX3]</td>
      <td>-0.537</td>
      <td>0.293</td>
      <td>-0.782</td>
      <td>-0.043</td>
      <td>0.143</td>
      <td>0.109</td>
      <td>7.0</td>
      <td>2103.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
    <tr>
      <th>83363</th>
      <td>h[OSGEP, ZFHX3]</td>
      <td>-0.524</td>
      <td>0.288</td>
      <td>-0.773</td>
      <td>-0.042</td>
      <td>0.139</td>
      <td>0.106</td>
      <td>7.0</td>
      <td>2196.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
    <tr>
      <th>86241</th>
      <td>h[SERPINA7, ZFHX3]</td>
      <td>0.184</td>
      <td>0.110</td>
      <td>0.027</td>
      <td>0.312</td>
      <td>0.046</td>
      <td>0.034</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.49</td>
      <td>h</td>
    </tr>
    <tr>
      <th>72883</th>
      <td>h[AFF4, ZFHX3]</td>
      <td>0.196</td>
      <td>0.128</td>
      <td>0.001</td>
      <td>0.335</td>
      <td>0.057</td>
      <td>0.043</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
    <tr>
      <th>77166</th>
      <td>h[ELOA, ZFHX3]</td>
      <td>0.212</td>
      <td>0.135</td>
      <td>0.005</td>
      <td>0.357</td>
      <td>0.060</td>
      <td>0.045</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
    <tr>
      <th>88690</th>
      <td>h[TP53, ZFHX3]</td>
      <td>0.227</td>
      <td>0.157</td>
      <td>-0.022</td>
      <td>0.387</td>
      <td>0.072</td>
      <td>0.054</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
    <tr>
      <th>77259</th>
      <td>h[EP300, ZFHX3]</td>
      <td>0.384</td>
      <td>0.236</td>
      <td>-0.009</td>
      <td>0.595</td>
      <td>0.114</td>
      <td>0.086</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
    </tr>
  </tbody>
</table>
</div>




```python
example_genes = ["KIF11", "AR", "NF2"]
az.plot_trace(
    trace,
    var_names=["mu_a", "b", "d", "f", "h"],
    coords={"gene": example_genes},
    compact=False,
)
plt.tight_layout()
plt.show()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")




![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_36_1.png)




```python
sgrnas_sample = trace.posterior.coords["sgrna"].values[:5]

az.plot_trace(trace, var_names="a", coords={"sgrna": sgrnas_sample}, compact=False)
plt.tight_layout()
plt.show()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")
    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")




![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_37_1.png)




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
        # "mu_f",
        "f",
        "h",
    ],
    coords={"gene": [example_gene], "sgrna": example_gene_sgrna},
    combined=False,
    figsize=(6, 7),
)
plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_38_0.png)




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
      <th>3</th>
      <td>sigma_a</td>
      <td>0.159</td>
      <td>0.092</td>
      <td>0.000</td>
      <td>0.213</td>
      <td>0.046</td>
      <td>0.035</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sigma_k</td>
      <td>0.029</td>
      <td>0.025</td>
      <td>0.000</td>
      <td>0.054</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>sigma_k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sigma_mu_a</td>
      <td>0.177</td>
      <td>0.102</td>
      <td>0.000</td>
      <td>0.238</td>
      <td>0.051</td>
      <td>0.039</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>sigma_mu_a</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sigma_b</td>
      <td>0.043</td>
      <td>0.025</td>
      <td>0.000</td>
      <td>0.059</td>
      <td>0.012</td>
      <td>0.010</td>
      <td>7.0</td>
      <td>33.0</td>
      <td>1.53</td>
      <td>sigma_b</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sigma_d</td>
      <td>0.038</td>
      <td>0.022</td>
      <td>0.000</td>
      <td>0.051</td>
      <td>0.011</td>
      <td>0.008</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.53</td>
      <td>sigma_d</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sigma_f</td>
      <td>0.153</td>
      <td>0.038</td>
      <td>0.126</td>
      <td>0.217</td>
      <td>0.019</td>
      <td>0.014</td>
      <td>7.0</td>
      <td>1174.0</td>
      <td>1.53</td>
      <td>sigma_f</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sigma_h[ZFHX3]</td>
      <td>0.135</td>
      <td>0.047</td>
      <td>0.054</td>
      <td>0.164</td>
      <td>0.023</td>
      <td>0.018</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
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
        var_names=["mu_a", "b", "d", "f", "h"],
        coords={"gene": [gene]},
        figsize=(7, 7),
        scatter_kwargs={"alpha": 0.1, "markersize": 2},
    )
    for ax in axes.flatten():
        ax.axhline(0, color="k")
        ax.axvline(0, color="k")
    plt.tight_layout()
    plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_40_0.png)





![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_40_1.png)




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
    palette="Set1",
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
    palette="Set1",
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



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_41_0.png)




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
      <td>-0.279</td>
      <td>0.261</td>
      <td>-0.429</td>
      <td>0.256</td>
      <td>0</td>
      <td>1</td>
      <td>mu_a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 2]</th>
      <td>0.331</td>
      <td>0.383</td>
      <td>0.094</td>
      <td>0.993</td>
      <td>0</td>
      <td>2</td>
      <td>mu_a</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 3]</th>
      <td>-0.356</td>
      <td>0.199</td>
      <td>-0.504</td>
      <td>-0.015</td>
      <td>0</td>
      <td>3</td>
      <td>mu_a</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 4]</th>
      <td>0.825</td>
      <td>0.022</td>
      <td>0.790</td>
      <td>0.846</td>
      <td>0</td>
      <td>4</td>
      <td>mu_a</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 0]</th>
      <td>-0.279</td>
      <td>0.261</td>
      <td>-0.429</td>
      <td>0.256</td>
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
      <td>-0.036</td>
      <td>0.190</td>
      <td>-0.366</td>
      <td>0.284</td>
      <td>1</td>
      <td>2</td>
      <td>b</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 3]</th>
      <td>0.490</td>
      <td>0.254</td>
      <td>0.291</td>
      <td>0.994</td>
      <td>1</td>
      <td>3</td>
      <td>b</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[1, 4]</th>
      <td>-0.450</td>
      <td>0.215</td>
      <td>-0.589</td>
      <td>-0.056</td>
      <td>1</td>
      <td>4</td>
      <td>b</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 0]</th>
      <td>0.331</td>
      <td>0.383</td>
      <td>0.094</td>
      <td>0.993</td>
      <td>2</td>
      <td>0</td>
      <td>d</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 1]</th>
      <td>-0.036</td>
      <td>0.190</td>
      <td>-0.366</td>
      <td>0.284</td>
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
      <td>0.046</td>
      <td>-0.160</td>
      <td>-0.020</td>
      <td>2</td>
      <td>3</td>
      <td>d</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[2, 4]</th>
      <td>0.296</td>
      <td>0.298</td>
      <td>0.106</td>
      <td>0.812</td>
      <td>2</td>
      <td>4</td>
      <td>d</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 0]</th>
      <td>-0.356</td>
      <td>0.199</td>
      <td>-0.504</td>
      <td>-0.015</td>
      <td>3</td>
      <td>0</td>
      <td>f</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 1]</th>
      <td>0.490</td>
      <td>0.254</td>
      <td>0.291</td>
      <td>0.994</td>
      <td>3</td>
      <td>1</td>
      <td>f</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[3, 2]</th>
      <td>-0.102</td>
      <td>0.046</td>
      <td>-0.160</td>
      <td>-0.020</td>
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
      <td>-0.538</td>
      <td>0.207</td>
      <td>-0.685</td>
      <td>-0.180</td>
      <td>3</td>
      <td>4</td>
      <td>f</td>
      <td>h[ZFHX3]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 0]</th>
      <td>0.825</td>
      <td>0.022</td>
      <td>0.790</td>
      <td>0.846</td>
      <td>4</td>
      <td>0</td>
      <td>h[ZFHX3]</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 1]</th>
      <td>-0.450</td>
      <td>0.215</td>
      <td>-0.589</td>
      <td>-0.056</td>
      <td>4</td>
      <td>1</td>
      <td>h[ZFHX3]</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 2]</th>
      <td>0.296</td>
      <td>0.298</td>
      <td>0.106</td>
      <td>0.812</td>
      <td>4</td>
      <td>2</td>
      <td>h[ZFHX3]</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[4, 3]</th>
      <td>-0.538</td>
      <td>0.207</td>
      <td>-0.685</td>
      <td>-0.180</td>
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



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_43_0.png)




```python
genes_var_corr = trace.posterior["genes_chol_cov_corr"].mean(axis=(1))
genes_var_names = ["mu_a", "b", "d", "f", "h"]
fig, axes = plt.subplots(2, 2, figsize=(7, 6))

for c, ax in enumerate(axes.flatten()):
    data = genes_var_corr[c, :, :].values.copy()
    np.fill_diagonal(data, np.nan)
    data[~np.tril(data).astype(bool)] = np.nan
    sns.heatmap(data, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)
    ax.set_xticklabels(genes_var_names)
    ax.set_yticklabels(genes_var_names)
    ax.set_title(f"chain {c}")

fig.tight_layout()
plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_44_0.png)




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
az.plot_trace(trace, var_names=["sigma_h"], compact=False)
plt.tight_layout()
plt.show()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/stats/density_utils.py:491: UserWarning: Your data appears to have a single value or no finite values
      warnings.warn("Your data appears to have a single value or no finite values")




![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_46_1.png)




```python
h_post_summary = (
    prostate_post_summary.query("var_name == 'h'")
    .reset_index(drop=True)
    .pipe(
        extract_coords_param_names,
        names=["hugo_symbol", "cancer_gene"],
        col="parameter",
    )
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
      <th>hugo_symbol</th>
      <th>cancer_gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>h[A1BG, ZFHX3]</td>
      <td>0.060</td>
      <td>0.078</td>
      <td>-0.059</td>
      <td>0.187</td>
      <td>0.012</td>
      <td>0.009</td>
      <td>60.0</td>
      <td>2183.0</td>
      <td>1.10</td>
      <td>h</td>
      <td>A1BG</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>h[A1CF, ZFHX3]</td>
      <td>0.093</td>
      <td>0.081</td>
      <td>0.003</td>
      <td>0.213</td>
      <td>0.027</td>
      <td>0.019</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>1.29</td>
      <td>h</td>
      <td>A1CF</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>h[A2M, ZFHX3]</td>
      <td>0.091</td>
      <td>0.081</td>
      <td>-0.001</td>
      <td>0.229</td>
      <td>0.021</td>
      <td>0.015</td>
      <td>17.0</td>
      <td>1912.0</td>
      <td>1.16</td>
      <td>h</td>
      <td>A2M</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>h[A2ML1, ZFHX3]</td>
      <td>-0.000</td>
      <td>0.062</td>
      <td>-0.110</td>
      <td>0.097</td>
      <td>0.001</td>
      <td>0.007</td>
      <td>4505.0</td>
      <td>2084.0</td>
      <td>1.53</td>
      <td>h</td>
      <td>A2ML1</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>h[A3GALT2, ZFHX3]</td>
      <td>0.025</td>
      <td>0.064</td>
      <td>-0.077</td>
      <td>0.133</td>
      <td>0.001</td>
      <td>0.007</td>
      <td>4673.0</td>
      <td>2305.0</td>
      <td>1.53</td>
      <td>h</td>
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
      <th>hugo_symbol</th>
      <th>cancer_gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4668</th>
      <td>h[ELL, ZFHX3]</td>
      <td>-0.569</td>
      <td>0.335</td>
      <td>-0.845</td>
      <td>-0.003</td>
      <td>0.163</td>
      <td>0.125</td>
      <td>7.0</td>
      <td>1636.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>ELL</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>7774</th>
      <td>h[KIF11, ZFHX3]</td>
      <td>-0.568</td>
      <td>0.331</td>
      <td>-0.847</td>
      <td>-0.011</td>
      <td>0.161</td>
      <td>0.122</td>
      <td>7.0</td>
      <td>2908.0</td>
      <td>1.53</td>
      <td>h</td>
      <td>KIF11</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>16413</th>
      <td>h[TRNT1, ZFHX3]</td>
      <td>-0.544</td>
      <td>0.328</td>
      <td>-0.816</td>
      <td>0.011</td>
      <td>0.160</td>
      <td>0.122</td>
      <td>7.0</td>
      <td>2877.0</td>
      <td>1.53</td>
      <td>h</td>
      <td>TRNT1</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>h[ATP6V1B2, ZFHX3]</td>
      <td>-0.537</td>
      <td>0.293</td>
      <td>-0.782</td>
      <td>-0.043</td>
      <td>0.143</td>
      <td>0.109</td>
      <td>7.0</td>
      <td>2103.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>ATP6V1B2</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>10876</th>
      <td>h[OSGEP, ZFHX3]</td>
      <td>-0.524</td>
      <td>0.288</td>
      <td>-0.773</td>
      <td>-0.042</td>
      <td>0.139</td>
      <td>0.106</td>
      <td>7.0</td>
      <td>2196.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>OSGEP</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>13754</th>
      <td>h[SERPINA7, ZFHX3]</td>
      <td>0.184</td>
      <td>0.110</td>
      <td>0.027</td>
      <td>0.312</td>
      <td>0.046</td>
      <td>0.034</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.49</td>
      <td>h</td>
      <td>SERPINA7</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>396</th>
      <td>h[AFF4, ZFHX3]</td>
      <td>0.196</td>
      <td>0.128</td>
      <td>0.001</td>
      <td>0.335</td>
      <td>0.057</td>
      <td>0.043</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>AFF4</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>4679</th>
      <td>h[ELOA, ZFHX3]</td>
      <td>0.212</td>
      <td>0.135</td>
      <td>0.005</td>
      <td>0.357</td>
      <td>0.060</td>
      <td>0.045</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>ELOA</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>16203</th>
      <td>h[TP53, ZFHX3]</td>
      <td>0.227</td>
      <td>0.157</td>
      <td>-0.022</td>
      <td>0.387</td>
      <td>0.072</td>
      <td>0.054</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>TP53</td>
      <td>ZFHX3</td>
    </tr>
    <tr>
      <th>4772</th>
      <td>h[EP300, ZFHX3]</td>
      <td>0.384</td>
      <td>0.236</td>
      <td>-0.009</td>
      <td>0.595</td>
      <td>0.114</td>
      <td>0.086</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.52</td>
      <td>h</td>
      <td>EP300</td>
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



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_49_0.png)




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



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_50_0.png)




```python
top_n = 5
top_b_hits = (
    prostate_post_summary.query("var_name == 'b'")
    .sort_values("mean")
    .reset_index(drop=True)
    .pipe(extract_coords_param_names, names=["hugo_symbol"], col="parameter")
    .pipe(head_tail, n=top_n)
)

negative_b = top_b_hits["hugo_symbol"][:top_n].values
positive_b = top_b_hits["hugo_symbol"][top_n:].values


fig, axes = plt.subplots(2, top_n, figsize=(12, 6))

for i, genes in enumerate([positive_b, negative_b]):
    for j, gene in enumerate(genes):
        ax = axes[i, j]
        ax.set_title(gene)
        obs_data = prostate_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(data=obs_data, x="rna_expr", y="lfc", ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("log RNA expression")
fig.supylabel("log-fold change")

fig.tight_layout()
plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_51_0.png)




```python
top_n = 5
top_d_hits = (
    prostate_post_summary.query("var_name == 'd'")
    .sort_values("mean")
    .reset_index(drop=True)
    .pipe(extract_coords_param_names, names=["hugo_symbol"], col="parameter")
    .pipe(head_tail, n=top_n)
)

negative_d = top_d_hits["hugo_symbol"][:top_n].values
positive_d = top_d_hits["hugo_symbol"][top_n:].values


fig, axes = plt.subplots(2, top_n, figsize=(12, 6))

for i, genes in enumerate([positive_d, negative_d]):
    for j, gene in enumerate(genes):
        ax = axes[i, j]
        ax.set_title(gene)
        obs_data = prostate_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(data=obs_data, x="copy_number", y="lfc", ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("copy number")
fig.supylabel("log-fold change")
fig.tight_layout()
plt.show()
```



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_52_0.png)



## PPC


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



![png](029_single-lineage-prostate-inspection_006_files/029_single-lineage-prostate-inspection_006_55_0.png)



---

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 6.04 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-08-03

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.45.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-233.o2.rc.hms.harvard.edu

    Git branch: simplify

    arviz     : 0.12.1
    pandas    : 1.4.3
    matplotlib: 3.5.2
    seaborn   : 0.11.2
    numpy     : 1.23.1




```python

```
