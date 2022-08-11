# Inspect the single-lineage model run on the prostate data (009)

 Model attributes:

- sgRNA | gene varying intercept
- RNA and CN varying effects per gene
- correlation between gene varying effects modeled using the multivariate normal and Cholesky decomposition (non-centered parameterization)
- target gene mutation variable and cancer gene comutation variable.
- varying intercept and slope for copy number for chromosome nested with in cell line


```python
%load_ext autoreload
%autoreload 2
```


```python
from math import ceil
from time import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
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
LINEAGE = "prostate"
```


```python
saved_model_dir = models_dir() / f"hnb-single-lineage-{LINEAGE}-009_PYMC_NUMPYRO"
```


```python
with open(saved_model_dir / "description.txt") as f:
    model_description = "".join(list(f))

print(model_description)
```

    config. name: 'hnb-single-lineage-prostate-009'
    model name: 'LineageHierNegBinomModel'
    model version: '0.0.3'
    model description: A hierarchical negative binomial generalized linear model for one lineage.
    fit method: 'PYMC_NUMPYRO'

    --------------------------------------------------------------------------------

    CONFIGURATION

    {
        "name": "hnb-single-lineage-prostate-009",
        "description": " Single lineage hierarchical negative binomial model for prostate data from the Broad.\nVarying effects for each chromosome of each cell line has been nested under the existing varying effects for cell line. ",
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
                "tune": 2000,
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
                                    delta_cells_dim_0: 2, delta_cells_dim_1: 5,
                                    cell_chrom: 115, genes_chol_cov_dim_0: 15,
                                    cells_chol_cov_dim_0: 3,
                                    genes_chol_cov_corr_dim_0: 5,
                                    genes_chol_cov_corr_dim_1: 5,
                                    genes_chol_cov_stds_dim_0: 5, cancer_gene: 1,
                                    gene: 18119, cells_chol_cov_corr_dim_0: 2,
                                    cells_chol_cov_corr_dim_1: 2,
                                    cells_chol_cov_stds_dim_0: 2, cell_line: 5)
    Coordinates: (12/19)
      * chain                      (chain) int64 0 1 2 3
      * draw                       (draw) int64 0 1 2 3 4 5 ... 995 996 997 998 999
      * delta_genes_dim_0          (delta_genes_dim_0) int64 0 1 2 3 4
      * delta_genes_dim_1          (delta_genes_dim_1) int64 0 1 2 ... 18117 18118
      * sgrna                      (sgrna) object 'AAAAAAATCCAGCAATGCAG' ... 'TTT...
      * delta_cells_dim_0          (delta_cells_dim_0) int64 0 1
        ...                         ...
      * cancer_gene                (cancer_gene) object 'ZFHX3'
      * gene                       (gene) object 'A1BG' 'A1CF' ... 'ZZEF1' 'ZZZ3'
      * cells_chol_cov_corr_dim_0  (cells_chol_cov_corr_dim_0) int64 0 1
      * cells_chol_cov_corr_dim_1  (cells_chol_cov_corr_dim_1) int64 0 1
      * cells_chol_cov_stds_dim_0  (cells_chol_cov_stds_dim_0) int64 0 1
      * cell_line                  (cell_line) object 'ACH-000115' ... 'ACH-001648'
    Data variables: (12/35)
        mu_mu_a                    (chain, draw) float64 ...
        mu_b                       (chain, draw) float64 ...
        delta_genes                (chain, draw, delta_genes_dim_0, delta_genes_dim_1) float64 ...
        delta_a                    (chain, draw, sgrna) float64 ...
        mu_mu_m                    (chain, draw) float64 ...
        delta_cells                (chain, draw, delta_cells_dim_0, delta_cells_dim_1) float64 ...
        ...                         ...
        sigma_mu_k                 (chain, draw) float64 ...
        sigma_mu_m                 (chain, draw) float64 ...
        mu_k                       (chain, draw, cell_line) float64 ...
        mu_m                       (chain, draw, cell_line) float64 ...
        k                          (chain, draw, cell_chrom) float64 ...
        m                          (chain, draw, cell_chrom) float64 ...
    Attributes:
        created_at:           2022-08-08 13:59:39.887469
        arviz_version:        0.12.1
        model_name:           LineageHierNegBinomModel
        model_version:        0.0.3
        model_doc:            A hierarchical negative binomial generalized linear...
        previous_created_at:  ['2022-08-08 13:59:39.887469', '2022-08-08T17:55:38...

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
        created_at:           2022-08-08 13:59:39.887469
        arviz_version:        0.12.1
        previous_created_at:  ['2022-08-08 13:59:39.887469', '2022-08-08T17:55:38...

    --------------------------------------------------------------------------------

    MCMC DESCRIPTION

    date created: 2022-08-08 13:59
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.591, 0.639, 0.536, 0.64
    avg. step size: 0.009, 0.01, 0.008, 0.009
    avg. accept prob.: 0.975, 0.974, 0.98, 0.975
    avg. tree depth: 9.0, 9.0, 9.0, 9.0


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
      <td>0.068</td>
      <td>0.022</td>
      <td>0.036</td>
      <td>0.101</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>1474.0</td>
      <td>1846.0</td>
      <td>1.0</td>
      <td>mu_mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_b</td>
      <td>0.004</td>
      <td>0.001</td>
      <td>0.003</td>
      <td>0.005</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3432.0</td>
      <td>2961.0</td>
      <td>1.0</td>
      <td>mu_b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_mu_m</td>
      <td>-0.252</td>
      <td>0.063</td>
      <td>-0.345</td>
      <td>-0.151</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2777.0</td>
      <td>2899.0</td>
      <td>1.0</td>
      <td>mu_mu_m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sigma_a</td>
      <td>0.211</td>
      <td>0.001</td>
      <td>0.210</td>
      <td>0.213</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1330.0</td>
      <td>2143.0</td>
      <td>1.0</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sigma_k</td>
      <td>0.033</td>
      <td>0.003</td>
      <td>0.028</td>
      <td>0.037</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1031.0</td>
      <td>1788.0</td>
      <td>1.0</td>
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
    modeling_data_dir()
    / "lineage-modeling-data"
    / f"depmap-modeling-data_{LINEAGE}.csv",
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
prostate_model = LineageHierNegBinomModel(lineage=LINEAGE)
```


```python
valid_prostate_data = prostate_model.data_processing_pipeline(prostate_data.copy())
prostate_mdl_data = prostate_model.make_data_structure(valid_prostate_data)
```

    [INFO] 2022-08-08 15:36:51 [(lineage_hierarchical_nb.py:data_processing_pipeline:317] Processing data for modeling.
    [INFO] 2022-08-08 15:36:51 [(lineage_hierarchical_nb.py:data_processing_pipeline:318] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-08-08 15:38:09 [(lineage_hierarchical_nb.py:data_processing_pipeline:376] number of data points dropped: 2
    [INFO] 2022-08-08 15:38:09 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:597] number of genes mutated in all cells lines: 0
    [DEBUG] 2022-08-08 15:38:09 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:600] Genes always mutated:
    [DEBUG] 2022-08-08 15:38:10 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:68] all_mut: {}
    [INFO] 2022-08-08 15:38:10 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:77] Dropping 8 cancer genes.
    [DEBUG] 2022-08-08 15:38:10 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:79] Dropped cancer genes: ['AR', 'AXIN1', 'FOXA1', 'KLF6', 'NCOR2', 'PTEN', 'SALL4', 'SPOP']


## Analysis


```python
sns.histplot(x=prostate_post_summary["r_hat"], binwidth=0.01, stat="proportion");
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_21_0.png)




```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=prostate_post_summary, x="var_name", y="r_hat", ax=ax)
ax.tick_params(rotation=90)
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_22_0.png)




```python
az.plot_energy(trace);
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_23_0.png)




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



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_24_0.png)




```python
hmc_energy = (
    trace.sample_stats.get("energy")
    .to_dataframe()
    .reset_index(drop=False)
    .astype({"chain": "category"})
)
ax = sns.lineplot(
    data=hmc_energy, x="draw", y="energy", hue="chain", palette="Set1", alpha=0.5
)
ax.set_xlim(0, hmc_energy["draw"].max())
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_25_0.png)




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
      <td>0.008994</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.975439</td>
      <td>2.454373e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.009703</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.974011</td>
      <td>2.454306e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.008403</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.979780</td>
      <td>2.454294e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.008710</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.974836</td>
      <td>2.454288e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(
    trace, var_names=["mu_mu_a", "mu_b", "mu_k", "mu_mu_m", "mu_m"], compact=True
)
plt.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_27_0.png)




```python
az.plot_trace(trace, var_names=["^sigma_*"], filter_vars="regex", compact=False)
plt.tight_layout()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_28_0.png)




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
      <td>0.230241</td>
      <td>0.056882</td>
      <td>0.322394</td>
      <td>0.129311</td>
      <td>0.033118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.230464</td>
      <td>0.056938</td>
      <td>0.321099</td>
      <td>0.129494</td>
      <td>0.033278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.230367</td>
      <td>0.056981</td>
      <td>0.321000</td>
      <td>0.129721</td>
      <td>0.032857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.230338</td>
      <td>0.056893</td>
      <td>0.321615</td>
      <td>0.129138</td>
      <td>0.032885</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(trace, var_names=["alpha"], compact=False)
plt.tight_layout()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_30_0.png)




```python
az.plot_forest(
    trace, var_names=["^sigma_*"], filter_vars="regex", combined=False, figsize=(5, 5)
)
plt.tight_layout()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_31_0.png)




```python
axes = az.plot_forest(
    trace,
    var_names=["sigma_mu_a", "sigma_b", "sigma_d", "sigma_f", "sigma_h"],
    figsize=(5, 3),
    combined=True,
)
axes[0].axvline(0, color="k", zorder=1)
plt.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_32_0.png)




```python
axes = az.plot_forest(
    trace,
    var_names=["sigma_mu_k", "mu_k", "sigma_k", "mu_mu_m", "sigma_mu_m", "mu_m"],
    figsize=(5, 6),
    combined=True,
)
axes[0].axvline(0, color="k", zorder=1)
plt.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_33_0.png)




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



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_34_0.png)




```python
sgrna_to_gene_map = (
    prostate_data.copy()[["hugo_symbol", "sgrna"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
```


```python
for v in ["mu_a", "b", "d", "f", "h", "k", "m"]:
    display(Markdown(f"variable: **{v}**"))
    top = (
        prostate_post_summary.query(f"var_name == '{v}'")
        .sort_values("mean")
        .reset_index(drop=True)
        .pipe(head_tail, 5)
    )
    display(top)
```


variable: **mu_a**



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
      <td>mu_a[KIF11]</td>
      <td>-1.211</td>
      <td>0.103</td>
      <td>-1.366</td>
      <td>-1.044</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3081.0</td>
      <td>3146.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_a[HSPE1]</td>
      <td>-1.032</td>
      <td>0.096</td>
      <td>-1.191</td>
      <td>-0.882</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2615.0</td>
      <td>2424.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_a[SPC24]</td>
      <td>-1.019</td>
      <td>0.102</td>
      <td>-1.185</td>
      <td>-0.862</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2819.0</td>
      <td>2917.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mu_a[RAN]</td>
      <td>-1.003</td>
      <td>0.097</td>
      <td>-1.150</td>
      <td>-0.843</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2907.0</td>
      <td>2986.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mu_a[DUX4]</td>
      <td>-0.998</td>
      <td>0.096</td>
      <td>-1.156</td>
      <td>-0.847</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2752.0</td>
      <td>3114.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>mu_a[PRAMEF4]</td>
      <td>0.402</td>
      <td>0.094</td>
      <td>0.244</td>
      <td>0.542</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3040.0</td>
      <td>2942.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>mu_a[ARHGAP44]</td>
      <td>0.403</td>
      <td>0.094</td>
      <td>0.264</td>
      <td>0.564</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2764.0</td>
      <td>2970.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>mu_a[HLA-DQB1]</td>
      <td>0.413</td>
      <td>0.105</td>
      <td>0.250</td>
      <td>0.585</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3195.0</td>
      <td>2995.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>mu_a[TP53]</td>
      <td>0.460</td>
      <td>0.094</td>
      <td>0.317</td>
      <td>0.610</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3008.0</td>
      <td>3057.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>mu_a[CCNF]</td>
      <td>0.468</td>
      <td>0.116</td>
      <td>0.277</td>
      <td>0.646</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3081.0</td>
      <td>3276.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
  </tbody>
</table>
</div>



variable: **b**



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
      <td>b[EP300]</td>
      <td>-0.334</td>
      <td>0.043</td>
      <td>-0.395</td>
      <td>-0.260</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>6200.0</td>
      <td>2790.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b[TP63]</td>
      <td>-0.186</td>
      <td>0.040</td>
      <td>-0.254</td>
      <td>-0.126</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>8945.0</td>
      <td>2779.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b[TADA1]</td>
      <td>-0.180</td>
      <td>0.041</td>
      <td>-0.243</td>
      <td>-0.115</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7273.0</td>
      <td>3033.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b[STAG2]</td>
      <td>-0.175</td>
      <td>0.039</td>
      <td>-0.234</td>
      <td>-0.112</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>6958.0</td>
      <td>2888.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b[HOXB13]</td>
      <td>-0.170</td>
      <td>0.044</td>
      <td>-0.240</td>
      <td>-0.099</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>7460.0</td>
      <td>2962.0</td>
      <td>1.01</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>b[NDUFB11]</td>
      <td>0.226</td>
      <td>0.042</td>
      <td>0.161</td>
      <td>0.294</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7899.0</td>
      <td>3028.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>b[ATP6V1F]</td>
      <td>0.228</td>
      <td>0.043</td>
      <td>0.164</td>
      <td>0.301</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>9207.0</td>
      <td>2798.0</td>
      <td>1.01</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>b[GPI]</td>
      <td>0.238</td>
      <td>0.044</td>
      <td>0.167</td>
      <td>0.311</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>7846.0</td>
      <td>3156.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>b[NARS2]</td>
      <td>0.245</td>
      <td>0.041</td>
      <td>0.182</td>
      <td>0.312</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>8483.0</td>
      <td>2982.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>b[AIFM1]</td>
      <td>0.272</td>
      <td>0.043</td>
      <td>0.204</td>
      <td>0.342</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7843.0</td>
      <td>2445.0</td>
      <td>1.00</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



variable: **d**



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
      <td>d[UBE2N]</td>
      <td>-1.014</td>
      <td>0.239</td>
      <td>-1.370</td>
      <td>-0.613</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>6817.0</td>
      <td>2824.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d[CNOT1]</td>
      <td>-0.938</td>
      <td>0.270</td>
      <td>-1.357</td>
      <td>-0.510</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>6855.0</td>
      <td>2881.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d[KMT2D]</td>
      <td>-0.896</td>
      <td>0.245</td>
      <td>-1.298</td>
      <td>-0.520</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5900.0</td>
      <td>2701.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d[EARS2]</td>
      <td>-0.894</td>
      <td>0.273</td>
      <td>-1.329</td>
      <td>-0.483</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>6316.0</td>
      <td>3080.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d[KLF5]</td>
      <td>-0.857</td>
      <td>0.234</td>
      <td>-1.221</td>
      <td>-0.474</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5845.0</td>
      <td>3357.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>d[MRPL39]</td>
      <td>0.832</td>
      <td>0.267</td>
      <td>0.424</td>
      <td>1.263</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>6060.0</td>
      <td>3295.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>d[TOP2A]</td>
      <td>0.838</td>
      <td>0.244</td>
      <td>0.478</td>
      <td>1.262</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5721.0</td>
      <td>3045.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>d[PWP2]</td>
      <td>0.849</td>
      <td>0.237</td>
      <td>0.444</td>
      <td>1.210</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5171.0</td>
      <td>2491.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>d[ITGB1]</td>
      <td>0.883</td>
      <td>0.249</td>
      <td>0.507</td>
      <td>1.284</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>7228.0</td>
      <td>2893.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>d[DMAC1]</td>
      <td>1.059</td>
      <td>0.252</td>
      <td>0.665</td>
      <td>1.459</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>6592.0</td>
      <td>2869.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>



variable: **f**



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
      <td>f[EP300]</td>
      <td>-0.485</td>
      <td>0.106</td>
      <td>-0.657</td>
      <td>-0.321</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5006.0</td>
      <td>3000.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f[STAG2]</td>
      <td>-0.290</td>
      <td>0.107</td>
      <td>-0.464</td>
      <td>-0.127</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6046.0</td>
      <td>2677.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>2</th>
      <td>f[AR]</td>
      <td>-0.228</td>
      <td>0.104</td>
      <td>-0.386</td>
      <td>-0.056</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5433.0</td>
      <td>3316.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>3</th>
      <td>f[MED13]</td>
      <td>-0.216</td>
      <td>0.101</td>
      <td>-0.381</td>
      <td>-0.058</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>8523.0</td>
      <td>2481.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>4</th>
      <td>f[KIF18B]</td>
      <td>-0.215</td>
      <td>0.109</td>
      <td>-0.395</td>
      <td>-0.049</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7377.0</td>
      <td>2943.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>f[PRIM1]</td>
      <td>0.393</td>
      <td>0.103</td>
      <td>0.232</td>
      <td>0.557</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7997.0</td>
      <td>3030.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>f[NXT1]</td>
      <td>0.394</td>
      <td>0.112</td>
      <td>0.221</td>
      <td>0.577</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>4088.0</td>
      <td>2810.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>f[LSM2]</td>
      <td>0.411</td>
      <td>0.101</td>
      <td>0.253</td>
      <td>0.578</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6457.0</td>
      <td>3196.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>f[SNAP23]</td>
      <td>0.421</td>
      <td>0.107</td>
      <td>0.260</td>
      <td>0.598</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6090.0</td>
      <td>2845.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>f[RAB6A]</td>
      <td>0.472</td>
      <td>0.105</td>
      <td>0.295</td>
      <td>0.632</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6058.0</td>
      <td>3093.0</td>
      <td>1.0</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>



variable: **h**



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
      <td>h[KIF11, ZFHX3]</td>
      <td>-0.787</td>
      <td>0.089</td>
      <td>-0.937</td>
      <td>-0.655</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5933.0</td>
      <td>3340.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>1</th>
      <td>h[ELL, ZFHX3]</td>
      <td>-0.765</td>
      <td>0.083</td>
      <td>-0.903</td>
      <td>-0.638</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5850.0</td>
      <td>3155.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>h[LONP1, ZFHX3]</td>
      <td>-0.753</td>
      <td>0.086</td>
      <td>-0.882</td>
      <td>-0.611</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6362.0</td>
      <td>3423.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>h[RPS4X, ZFHX3]</td>
      <td>-0.742</td>
      <td>0.076</td>
      <td>-0.851</td>
      <td>-0.611</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6171.0</td>
      <td>3632.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>4</th>
      <td>h[TRNT1, ZFHX3]</td>
      <td>-0.737</td>
      <td>0.081</td>
      <td>-0.881</td>
      <td>-0.622</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4999.0</td>
      <td>3322.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>h[CCNF, ZFHX3]</td>
      <td>0.256</td>
      <td>0.100</td>
      <td>0.102</td>
      <td>0.419</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5309.0</td>
      <td>2922.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>h[NKD1, ZFHX3]</td>
      <td>0.259</td>
      <td>0.074</td>
      <td>0.141</td>
      <td>0.379</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5773.0</td>
      <td>3234.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>h[ELOA, ZFHX3]</td>
      <td>0.269</td>
      <td>0.073</td>
      <td>0.149</td>
      <td>0.378</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5592.0</td>
      <td>3067.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>h[TP53, ZFHX3]</td>
      <td>0.337</td>
      <td>0.072</td>
      <td>0.221</td>
      <td>0.451</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6037.0</td>
      <td>3412.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>h[EP300, ZFHX3]</td>
      <td>0.528</td>
      <td>0.074</td>
      <td>0.413</td>
      <td>0.650</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3935.0</td>
      <td>3143.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
  </tbody>
</table>
</div>



variable: **k**



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
      <td>k[ACH-001453__12]</td>
      <td>-0.113</td>
      <td>0.023</td>
      <td>-0.146</td>
      <td>-0.076</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1575.0</td>
      <td>1892.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>k[ACH-001627__16]</td>
      <td>-0.095</td>
      <td>0.025</td>
      <td>-0.134</td>
      <td>-0.057</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1451.0</td>
      <td>2017.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>k[ACH-000977__16]</td>
      <td>-0.089</td>
      <td>0.025</td>
      <td>-0.125</td>
      <td>-0.049</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1442.0</td>
      <td>2011.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>k[ACH-000977__19]</td>
      <td>-0.077</td>
      <td>0.024</td>
      <td>-0.114</td>
      <td>-0.042</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1364.0</td>
      <td>1795.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>k[ACH-001627__17]</td>
      <td>-0.072</td>
      <td>0.024</td>
      <td>-0.109</td>
      <td>-0.034</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1458.0</td>
      <td>1981.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>110</th>
      <td>k[ACH-000115__10]</td>
      <td>0.069</td>
      <td>0.023</td>
      <td>0.035</td>
      <td>0.106</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1469.0</td>
      <td>1973.0</td>
      <td>1.01</td>
      <td>k</td>
    </tr>
    <tr>
      <th>111</th>
      <td>k[ACH-000115__11]</td>
      <td>0.069</td>
      <td>0.023</td>
      <td>0.033</td>
      <td>0.104</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1482.0</td>
      <td>1717.0</td>
      <td>1.01</td>
      <td>k</td>
    </tr>
    <tr>
      <th>112</th>
      <td>k[ACH-000115__4]</td>
      <td>0.072</td>
      <td>0.024</td>
      <td>0.036</td>
      <td>0.107</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1443.0</td>
      <td>2142.0</td>
      <td>1.01</td>
      <td>k</td>
    </tr>
    <tr>
      <th>113</th>
      <td>k[ACH-001648__4]</td>
      <td>0.079</td>
      <td>0.024</td>
      <td>0.043</td>
      <td>0.115</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1500.0</td>
      <td>1919.0</td>
      <td>1.01</td>
      <td>k</td>
    </tr>
    <tr>
      <th>114</th>
      <td>k[ACH-000115__X]</td>
      <td>0.080</td>
      <td>0.026</td>
      <td>0.041</td>
      <td>0.124</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1848.0</td>
      <td>2648.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
  </tbody>
</table>
</div>



variable: **m**



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
      <td>m[ACH-001453__16]</td>
      <td>-0.729</td>
      <td>0.070</td>
      <td>-0.839</td>
      <td>-0.620</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5945.0</td>
      <td>3237.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>1</th>
      <td>m[ACH-000977__7]</td>
      <td>-0.712</td>
      <td>0.188</td>
      <td>-1.015</td>
      <td>-0.427</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>5418.0</td>
      <td>3326.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>m[ACH-000977__4]</td>
      <td>-0.672</td>
      <td>0.168</td>
      <td>-0.944</td>
      <td>-0.412</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5703.0</td>
      <td>3213.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>m[ACH-001453__12]</td>
      <td>-0.631</td>
      <td>0.033</td>
      <td>-0.683</td>
      <td>-0.578</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5085.0</td>
      <td>3457.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>4</th>
      <td>m[ACH-000977__13]</td>
      <td>-0.613</td>
      <td>0.124</td>
      <td>-0.809</td>
      <td>-0.412</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>4295.0</td>
      <td>2756.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>110</th>
      <td>m[ACH-001627__12]</td>
      <td>-0.002</td>
      <td>0.171</td>
      <td>-0.264</td>
      <td>0.278</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>5390.0</td>
      <td>2942.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>111</th>
      <td>m[ACH-001453__11]</td>
      <td>0.008</td>
      <td>0.135</td>
      <td>-0.195</td>
      <td>0.232</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3828.0</td>
      <td>2942.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>112</th>
      <td>m[ACH-001453__20]</td>
      <td>0.036</td>
      <td>0.108</td>
      <td>-0.131</td>
      <td>0.211</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3205.0</td>
      <td>2561.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>113</th>
      <td>m[ACH-001627__21]</td>
      <td>0.053</td>
      <td>0.174</td>
      <td>-0.236</td>
      <td>0.313</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>4811.0</td>
      <td>3064.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>114</th>
      <td>m[ACH-000115__10]</td>
      <td>0.056</td>
      <td>0.089</td>
      <td>-0.079</td>
      <td>0.204</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5208.0</td>
      <td>3478.0</td>
      <td>1.0</td>
      <td>m</td>
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
    compact=True,
)
plt.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_37_0.png)




```python
sgrnas_sample = trace.posterior.coords["sgrna"].values[:5]

az.plot_trace(trace, var_names="a", coords={"sgrna": sgrnas_sample}, compact=False)
plt.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_38_0.png)




```python
cell_chromosome_map = (
    valid_prostate_data[["depmap_id", "sgrna_target_chr", "cell_chrom"]]
    .drop_duplicates()
    .sort_values("cell_chrom")
    .reset_index(drop=True)
)
```


```python
chromosome_effect_post = (
    az.summary(trace, var_names=["k", "m"], kind="stats")
    .pipe(extract_coords_param_names, names="cell_chrom")
    .assign(var_name=lambda d: [p[0] for p in d.index.values])
    .merge(cell_chromosome_map, on="cell_chrom")
)

cell_effect_post = (
    az.summary(trace, var_names=["mu_k", "mu_m"], kind="stats")
    .pipe(extract_coords_param_names, names="depmap_id")
    .assign(var_name=lambda d: [p[:4] for p in d.index.values])
    .reset_index(drop=True)
    .pivot_wider(
        index="depmap_id",
        names_from="var_name",
        values_from=["mean", "hdi_5.5%", "hdi_94.5%"],
    )
)
```


```python
ncells = chromosome_effect_post["depmap_id"].nunique()
ncols = 3
nrows = ceil(ncells / 3)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, nrows * 3))
for ax, (cell, data_c) in zip(
    axes.flatten(), chromosome_effect_post.groupby("depmap_id")
):
    plot_df = data_c.pivot_wider(
        index="sgrna_target_chr",
        names_from="var_name",
        values_from=["mean", "hdi_5.5%", "hdi_94.5%"],
    )
    cell_eff = (
        cell_effect_post.copy().query(f"depmap_id == '{cell}'").reset_index(drop=True)
    )

    ax.axhline(0, color="k", zorder=1)
    ax.axvline(0, color="k", zorder=1)

    ax.vlines(
        x=plot_df["mean_k"],
        ymin=plot_df["hdi_5.5%_m"],
        ymax=plot_df["hdi_94.5%_m"],
        alpha=0.5,
        zorder=5,
    )
    ax.hlines(
        y=plot_df["mean_m"],
        xmin=plot_df["hdi_5.5%_k"],
        xmax=plot_df["hdi_94.5%_k"],
        alpha=0.5,
        zorder=5,
    )
    sns.scatterplot(data=plot_df, x="mean_k", y="mean_m", ax=ax, zorder=10)

    ax.vlines(
        x=cell_eff["mean_mu_k"],
        ymin=cell_eff["hdi_5.5%_mu_m"],
        ymax=cell_eff["hdi_94.5%_mu_m"],
        alpha=0.8,
        zorder=15,
        color="red",
    )
    ax.hlines(
        y=cell_eff["mean_mu_m"],
        xmin=cell_eff["hdi_5.5%_mu_k"],
        xmax=cell_eff["hdi_94.5%_mu_k"],
        alpha=0.8,
        zorder=15,
        color="red",
    )
    sns.scatterplot(
        data=cell_eff, x="mean_mu_k", y="mean_mu_m", ax=ax, zorder=20, color="red"
    )
    ax.set_title(cell)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

for ax in axes[-1, :]:
    ax.set_xlabel("$k$")
for ax in axes[:, 0]:
    ax.set_ylabel("$m$")
for ax in axes.flatten()[ncells:]:
    ax.axis("off")

fig.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_41_0.png)




```python
for v, data_v in chromosome_effect_post.groupby("var_name"):
    df = (
        data_v.copy()
        .reset_index(drop=True)
        .pivot_wider(
            index="depmap_id", names_from="sgrna_target_chr", values_from="mean"
        )
        .set_index("depmap_id")
    )
    cg = sns.clustermap(df, figsize=(6, 4))
    cg.ax_col_dendrogram.set_title(f"variable: ${v}$")
    plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_42_0.png)





![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_42_1.png)




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
```


```python
plot_df = gene_corr_post.pivot_wider("p1", "p2", "mean").set_index("p1")
sns.heatmap(plot_df, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_44_0.png)




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



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_46_0.png)




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

ax = sns.kdeplot(data=h_post_summary, x="mean", hue="cancer_gene")
ax.set_xlabel(r"$\bar{h}_g$ posterior")
ax.set_ylabel("density")
ax.get_legend().set_title("cancer gene\ncomut.")
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_47_0.png)




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
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), title=cg)

plt.tight_layout()
plt.show()
```



![png](034_single-lineage-prostate-inspection_009_files/034_single-lineage-prostate-inspection_009_48_0.png)



---

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 2.36 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-08-08

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

    Hostname: compute-a-16-170.o2.rc.hms.harvard.edu

    Git branch: varying-chromosome

    matplotlib: 3.5.2
    pandas    : 1.4.3
    numpy     : 1.23.1
    seaborn   : 0.11.2
    arviz     : 0.12.1




```python

```
