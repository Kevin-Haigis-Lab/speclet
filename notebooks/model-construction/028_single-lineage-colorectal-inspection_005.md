# Inspect the single-lineage model run on the colorectal data (005)

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
import seaborn as sns
from IPython.display import Markdown, display
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
saved_model_dir = models_dir() / "hnb-single-lineage-colorectal-005_PYMC_NUMPYRO"
```


```python
with open(saved_model_dir / "description.txt") as f:
    model_description = "".join(list(f))

print(model_description)
```

    name: 'hnb-single-lineage-colorectal-005'
    fit method: 'PYMC_NUMPYRO'

    --------------------------------------------------------------------------------

    CONFIGURATION

    {
        "name": "hnb-single-lineage-colorectal-005",
        "description": " Single lineage hierarchical negative binomial model for colorectal data from the Broad. The number of cancer genes has been limited by having to be mutated in at least 25% of cell lines. I have also increased the number of tuning steps from 1,000 to 2,000 and increased the target acceptance probability to 0.99. ",
        "active": true,
        "model": "LINEAGE_HIERARCHICAL_NB",
        "data_file": "modeling_data/lineage-modeling-data/depmap-modeling-data_colorectal.csv",
        "model_kwargs": {
            "lineage": "colorectal",
            "min_frac_cancer_genes": 0.2
        },
        "sampling_kwargs": {
            "pymc_mcmc": null,
            "pymc_advi": null,
            "pymc_numpyro": {
                "draws": 1000,
                "tune": 2000,
                "chains": 4,
                "target_accept": 0.99,
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
    Dimensions:                    (chain: 4, draw: 1000, delta_genes_dim_0: 13,
                                    delta_genes_dim_1: 18119, sgrna: 71062,
                                    genes_chol_cov_dim_0: 91,
                                    genes_chol_cov_corr_dim_0: 13,
                                    genes_chol_cov_corr_dim_1: 13,
                                    genes_chol_cov_stds_dim_0: 13, cancer_gene: 9,
                                    gene: 18119)
    Coordinates:
      * chain                      (chain) int64 0 1 2 3
      * draw                       (draw) int64 0 1 2 3 4 5 ... 995 996 997 998 999
      * delta_genes_dim_0          (delta_genes_dim_0) int64 0 1 2 3 ... 9 10 11 12
      * delta_genes_dim_1          (delta_genes_dim_1) int64 0 1 2 ... 18117 18118
      * sgrna                      (sgrna) object 'AAAAAAATCCAGCAATGCAG' ... 'TTT...
      * genes_chol_cov_dim_0       (genes_chol_cov_dim_0) int64 0 1 2 3 ... 88 89 90
      * genes_chol_cov_corr_dim_0  (genes_chol_cov_corr_dim_0) int64 0 1 2 ... 11 12
      * genes_chol_cov_corr_dim_1  (genes_chol_cov_corr_dim_1) int64 0 1 2 ... 11 12
      * genes_chol_cov_stds_dim_0  (genes_chol_cov_stds_dim_0) int64 0 1 2 ... 11 12
      * cancer_gene                (cancer_gene) object 'APC' 'AXIN2' ... 'UBR5'
      * gene                       (gene) object 'A1BG' 'A1CF' ... 'ZZEF1' 'ZZZ3'
    Data variables: (12/21)
        mu_mu_a                    (chain, draw) float64 ...
        mu_b                       (chain, draw) float64 ...
        mu_d                       (chain, draw) float64 ...
        delta_genes                (chain, draw, delta_genes_dim_0, delta_genes_dim_1) float64 ...
        delta_a                    (chain, draw, sgrna) float64 ...
        genes_chol_cov             (chain, draw, genes_chol_cov_dim_0) float64 ...
        ...                         ...
        mu_a                       (chain, draw, gene) float64 ...
        b                          (chain, draw, gene) float64 ...
        d                          (chain, draw, gene) float64 ...
        f                          (chain, draw, gene) float64 ...
        h                          (chain, draw, gene, cancer_gene) float64 ...
        a                          (chain, draw, sgrna) float64 ...
    Attributes:
        created_at:           2022-08-03 03:07:25.920215
        arviz_version:        0.12.1
        previous_created_at:  ['2022-08-03 03:07:25.920215', '2022-08-03T06:06:14...

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
        created_at:           2022-08-03 03:07:25.920215
        arviz_version:        0.12.1
        previous_created_at:  ['2022-08-03 03:07:25.920215', '2022-08-03T06:06:14...

    --------------------------------------------------------------------------------

    MCMC DESCRIPTION

    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.738, 0.731, 0.759, 0.698
    avg. step size: 0.009, 0.01, 0.009, 0.01


### Load posterior summary


```python
crc_post_summary = pd.read_csv(saved_model_dir / "posterior-summary.csv").assign(
    var_name=lambda d: [x.split("[")[0] for x in d["parameter"]]
)
crc_post_summary.head()
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
      <td>0.119</td>
      <td>0.001</td>
      <td>0.118</td>
      <td>0.121</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2583.0</td>
      <td>2932.0</td>
      <td>1.0</td>
      <td>mu_mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_b</td>
      <td>-0.001</td>
      <td>0.000</td>
      <td>-0.001</td>
      <td>-0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4301.0</td>
      <td>3589.0</td>
      <td>1.0</td>
      <td>mu_b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_d</td>
      <td>-0.022</td>
      <td>0.000</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4271.0</td>
      <td>3631.0</td>
      <td>1.0</td>
      <td>mu_d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sigma_a</td>
      <td>0.182</td>
      <td>0.001</td>
      <td>0.181</td>
      <td>0.183</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1168.0</td>
      <td>2194.0</td>
      <td>1.0</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alpha</td>
      <td>7.858</td>
      <td>0.007</td>
      <td>7.848</td>
      <td>7.870</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3100.0</td>
      <td>2713.0</td>
      <td>1.0</td>
      <td>alpha</td>
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

### Colorectal data


```python
def _broad_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["screen"] == "broad"].reset_index(drop=True)


crc_dm = CrisprScreenDataManager(
    modeling_data_dir()
    / "lineage-modeling-data"
    / "depmap-modeling-data_colorectal.csv",
    transformations=[_broad_only],
)
```


```python
crc_data = crc_dm.get_data(read_kwargs={"low_memory": False})
crc_data.head()
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
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>0.594321</td>
      <td>2</td>
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
      <td>0.951337</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAATCAGAGAAACCTGAACG</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.363633</td>
      <td>2</td>
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
      <td>0.945508</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACGTCTTTGAAGAAAGCTG</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.001343</td>
      <td>2</td>
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
      <td>1.428663</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACGTCTTTGAAGGAAGCTG</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>0.367220</td>
      <td>2</td>
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
      <td>1.428663</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAGAGGTTCCAGACTACTTA</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-1.180029</td>
      <td>2</td>
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
      <td>0.466222</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



### Single lineage model


```python
crc_model = LineageHierNegBinomModel(lineage="colorectal", min_frac_cancer_genes=0.2)
```


```python
valid_crc_data = crc_model.data_processing_pipeline(crc_data.copy())
crc_model_data = crc_model.make_data_structure(valid_crc_data)
```

    [INFO] 2022-08-03 06:59:50 [(lineage_hierarchical_nb.py:data_processing_pipeline:274] Processing data for modeling.
    [INFO] 2022-08-03 06:59:50 [(lineage_hierarchical_nb.py:data_processing_pipeline:275] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-08-03 07:01:33 [(lineage_hierarchical_nb.py:data_processing_pipeline:326] number of data points dropped: 25
    [INFO] 2022-08-03 07:01:36 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:493] number of genes mutated in all cells lines: 0
    [DEBUG] 2022-08-03 07:01:36 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:496] Genes always mutated:
    [INFO] 2022-08-03 07:01:39 [(lineage_hierarchical_nb.py:_trim_cancer_genes:549] Dropping 21 cancer genes.
    [DEBUG] 2022-08-03 07:01:39 [(lineage_hierarchical_nb.py:_trim_cancer_genes:550] Dropped cancer genes: ['AKT1', 'AXIN1', 'BAX', 'ERBB3', 'GRIN2A', 'HIF1A', 'MAP2K1', 'MAX', 'MDM2', 'MLH1', 'MSH2', 'PIK3R1', 'POLE', 'PTPRT', 'SALL4', 'SFRP4', 'SMAD2', 'SMAD3', 'SMAD4', 'SRC', 'TGFBR2']



```python
crc_model_data.coords["cancer_gene"]
```




    ['APC', 'AXIN2', 'B2M', 'FBXW7', 'KRAS', 'MSH6', 'PIK3CA', 'POLD1', 'UBR5']



## Analysis


```python
sns.histplot(x=crc_post_summary["r_hat"], binwidth=0.01, stat="proportion");
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_21_0.png)




```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=crc_post_summary, x="var_name", y="r_hat", ax=ax)
ax.tick_params(rotation=90)
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_22_0.png)




```python
az.plot_energy(trace);
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_23_0.png)




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



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_24_0.png)




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
      <td>0.008668</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.988654</td>
      <td>1.912181e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010030</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.985516</td>
      <td>1.912168e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.009039</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.987882</td>
      <td>1.912167e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.010094</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.984214</td>
      <td>1.912171e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
# HAPPY_CHAINS = [0, 1, 2]
# SAD_CHAINS = [3]
# trace.posterior = trace.posterior.sel({"chain": HAPPY_CHAINS})
# trace.posterior_predictive = trace.posterior_predictive.sel({"chain": HAPPY_CHAINS})
```


```python
az.plot_trace(trace, var_names=["mu_mu_a", "mu_b", "mu_d"], compact=False)
plt.tight_layout()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_27_0.png)




```python
az.summary(trace, var_names=["mu_mu_a", "mu_b", "mu_d"])
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
      <th>mu_mu_a</th>
      <td>0.119</td>
      <td>0.001</td>
      <td>0.118</td>
      <td>0.121</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2583.0</td>
      <td>2932.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_b</th>
      <td>-0.001</td>
      <td>0.000</td>
      <td>-0.001</td>
      <td>-0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4301.0</td>
      <td>3589.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_d</th>
      <td>-0.022</td>
      <td>0.000</td>
      <td>-0.023</td>
      <td>-0.022</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4271.0</td>
      <td>3631.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(
    trace, var_names=[f"sigma_{x}" for x in ["a", "mu_a", "b", "d", "f"]], compact=False
)
plt.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_29_0.png)




```python
az.plot_trace(trace, var_names=["sigma_h"], compact=True)
plt.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_30_0.png)




```python
az.plot_trace(trace, var_names=["alpha"], compact=False)
plt.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_31_0.png)




```python
az.plot_forest(
    trace, var_names=["^sigma_*"], filter_vars="regex", combined=False, figsize=(5, 8)
)
plt.tight_layout()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_32_0.png)




```python
var_names = ["a", "mu_a", "b", "d", "f", "h"]
_, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True)
for ax, var_name in zip(axes.flatten(), var_names):
    x = crc_post_summary.query(f"var_name == '{var_name}'")["mean"]
    sns.kdeplot(x=x, ax=ax)
    ax.set_title(var_name)
    ax.set_xlim(-2, 1)

plt.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_33_0.png)




```python
sgrna_to_gene_map = (
    crc_data.copy()[["hugo_symbol", "sgrna"]].drop_duplicates().reset_index(drop=True)
)
```


```python
crc_post_summary.query("var_name == 'mu_a'").sort_values("mean").pipe(head_tail, 5)
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
      <th>12660</th>
      <td>mu_a[RAN]</td>
      <td>-1.183</td>
      <td>0.075</td>
      <td>-1.304</td>
      <td>-1.061</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2563.0</td>
      <td>2617.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>12332</th>
      <td>mu_a[PSMD7]</td>
      <td>-1.062</td>
      <td>0.077</td>
      <td>-1.187</td>
      <td>-0.943</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2712.0</td>
      <td>2935.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>7792</th>
      <td>mu_a[KIF11]</td>
      <td>-1.023</td>
      <td>0.074</td>
      <td>-1.147</td>
      <td>-0.913</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2302.0</td>
      <td>2828.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>11849</th>
      <td>mu_a[POLR2L]</td>
      <td>-1.020</td>
      <td>0.074</td>
      <td>-1.143</td>
      <td>-0.904</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2806.0</td>
      <td>2531.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>13322</th>
      <td>mu_a[RPSA]</td>
      <td>-1.006</td>
      <td>0.075</td>
      <td>-1.135</td>
      <td>-0.893</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2512.0</td>
      <td>2916.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>13177</th>
      <td>mu_a[ROCK2]</td>
      <td>0.389</td>
      <td>0.074</td>
      <td>0.275</td>
      <td>0.510</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2718.0</td>
      <td>2902.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>16407</th>
      <td>mu_a[TRIQK]</td>
      <td>0.394</td>
      <td>0.073</td>
      <td>0.280</td>
      <td>0.512</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2724.0</td>
      <td>2752.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>10472</th>
      <td>mu_a[OFD1]</td>
      <td>0.400</td>
      <td>0.071</td>
      <td>0.295</td>
      <td>0.519</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2491.0</td>
      <td>3037.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>17905</th>
      <td>mu_a[ZNF611]</td>
      <td>0.413</td>
      <td>0.072</td>
      <td>0.304</td>
      <td>0.529</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2441.0</td>
      <td>2944.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>8819</th>
      <td>mu_a[MAPK9]</td>
      <td>0.423</td>
      <td>0.083</td>
      <td>0.283</td>
      <td>0.549</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2879.0</td>
      <td>2834.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
  </tbody>
</table>
</div>




```python
crc_post_summary.query("var_name == 'b'").sort_values("mean").pipe(head_tail, 5)
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
      <th>31284</th>
      <td>b[RNGTT]</td>
      <td>-0.090</td>
      <td>0.015</td>
      <td>-0.115</td>
      <td>-0.066</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8053.0</td>
      <td>2945.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18806</th>
      <td>b[ANKLE2]</td>
      <td>-0.090</td>
      <td>0.015</td>
      <td>-0.112</td>
      <td>-0.064</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9072.0</td>
      <td>2635.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>30419</th>
      <td>b[PSMA6]</td>
      <td>-0.084</td>
      <td>0.016</td>
      <td>-0.109</td>
      <td>-0.056</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8557.0</td>
      <td>2802.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>30883</th>
      <td>b[RBM22]</td>
      <td>-0.084</td>
      <td>0.016</td>
      <td>-0.107</td>
      <td>-0.057</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7176.0</td>
      <td>2830.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>30451</th>
      <td>b[PSMD7]</td>
      <td>-0.082</td>
      <td>0.017</td>
      <td>-0.107</td>
      <td>-0.055</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8109.0</td>
      <td>3167.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>24038</th>
      <td>b[GET4]</td>
      <td>0.034</td>
      <td>0.015</td>
      <td>0.011</td>
      <td>0.059</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8017.0</td>
      <td>2648.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>33183</th>
      <td>b[STON1]</td>
      <td>0.035</td>
      <td>0.014</td>
      <td>0.012</td>
      <td>0.056</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10819.0</td>
      <td>3014.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18748</th>
      <td>b[AMIGO2]</td>
      <td>0.035</td>
      <td>0.014</td>
      <td>0.015</td>
      <td>0.059</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9299.0</td>
      <td>2655.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>20819</th>
      <td>b[CDKN1C]</td>
      <td>0.045</td>
      <td>0.015</td>
      <td>0.021</td>
      <td>0.068</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7445.0</td>
      <td>2837.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>20817</th>
      <td>b[CDKN1A]</td>
      <td>0.057</td>
      <td>0.015</td>
      <td>0.032</td>
      <td>0.080</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8310.0</td>
      <td>2461.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
crc_post_summary.query("var_name == 'd'").sort_values("mean").pipe(head_tail, 5)
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
      <th>50218</th>
      <td>d[SIK3]</td>
      <td>-0.113</td>
      <td>0.021</td>
      <td>-0.147</td>
      <td>-0.080</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9276.0</td>
      <td>2842.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>38694</th>
      <td>d[CCND1]</td>
      <td>-0.112</td>
      <td>0.020</td>
      <td>-0.143</td>
      <td>-0.077</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10973.0</td>
      <td>2728.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>41087</th>
      <td>d[ERBB2]</td>
      <td>-0.110</td>
      <td>0.023</td>
      <td>-0.147</td>
      <td>-0.073</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8812.0</td>
      <td>2709.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>48547</th>
      <td>d[PSMB5]</td>
      <td>-0.104</td>
      <td>0.021</td>
      <td>-0.142</td>
      <td>-0.073</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6334.0</td>
      <td>3126.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>50106</th>
      <td>d[SGK1]</td>
      <td>-0.093</td>
      <td>0.022</td>
      <td>-0.132</td>
      <td>-0.061</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10640.0</td>
      <td>2608.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>43071</th>
      <td>d[HNRNPK]</td>
      <td>0.115</td>
      <td>0.021</td>
      <td>0.080</td>
      <td>0.147</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10445.0</td>
      <td>2433.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>48080</th>
      <td>d[POLR2D]</td>
      <td>0.116</td>
      <td>0.020</td>
      <td>0.086</td>
      <td>0.151</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8419.0</td>
      <td>2863.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>39248</th>
      <td>d[CIAO3]</td>
      <td>0.118</td>
      <td>0.020</td>
      <td>0.086</td>
      <td>0.149</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7065.0</td>
      <td>3107.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>42214</th>
      <td>d[GINS2]</td>
      <td>0.129</td>
      <td>0.021</td>
      <td>0.094</td>
      <td>0.159</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9683.0</td>
      <td>2769.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
    <tr>
      <th>47943</th>
      <td>d[PLK1]</td>
      <td>0.160</td>
      <td>0.021</td>
      <td>0.125</td>
      <td>0.190</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10820.0</td>
      <td>2655.0</td>
      <td>1.0</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>




```python
crc_post_summary.query("var_name == 'h'").sort_values("mean").pipe(head_tail, 5)
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
      <th>186277</th>
      <td>h[RAN, MSH6]</td>
      <td>-0.935</td>
      <td>0.052</td>
      <td>-1.019</td>
      <td>-0.852</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2690.0</td>
      <td>2749.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>111172</th>
      <td>h[DONSON, MSH6]</td>
      <td>-0.855</td>
      <td>0.053</td>
      <td>-0.944</td>
      <td>-0.772</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2946.0</td>
      <td>3117.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>192235</th>
      <td>h[RPSA, MSH6]</td>
      <td>-0.846</td>
      <td>0.054</td>
      <td>-0.938</td>
      <td>-0.765</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2618.0</td>
      <td>2965.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>183325</th>
      <td>h[PSMD7, MSH6]</td>
      <td>-0.838</td>
      <td>0.054</td>
      <td>-0.921</td>
      <td>-0.751</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2952.0</td>
      <td>3137.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>178978</th>
      <td>h[POLR2L, MSH6]</td>
      <td>-0.833</td>
      <td>0.053</td>
      <td>-0.915</td>
      <td>-0.745</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2937.0</td>
      <td>2991.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>111173</th>
      <td>h[DONSON, PIK3CA]</td>
      <td>0.624</td>
      <td>0.045</td>
      <td>0.556</td>
      <td>0.700</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3509.0</td>
      <td>3278.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>192236</th>
      <td>h[RPSA, PIK3CA]</td>
      <td>0.624</td>
      <td>0.045</td>
      <td>0.559</td>
      <td>0.700</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3784.0</td>
      <td>3236.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>183326</th>
      <td>h[PSMD7, PIK3CA]</td>
      <td>0.627</td>
      <td>0.045</td>
      <td>0.554</td>
      <td>0.697</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3773.0</td>
      <td>3040.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>178979</th>
      <td>h[POLR2L, PIK3CA]</td>
      <td>0.631</td>
      <td>0.044</td>
      <td>0.563</td>
      <td>0.704</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>3964.0</td>
      <td>3424.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>186278</th>
      <td>h[RAN, PIK3CA]</td>
      <td>0.727</td>
      <td>0.045</td>
      <td>0.655</td>
      <td>0.798</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3450.0</td>
      <td>3049.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
  </tbody>
</table>
</div>




```python
def _md_bold(s: str) -> None:
    display(Markdown(f"**{s}**"))
```


```python
example_genes = ["KRAS", "APC"]
for gene in example_genes:
    _md_bold(gene)
    az.plot_trace(
        trace,
        var_names=["mu_a", "b", "d", "f", "h"],
        coords={"gene": [gene]},
        compact=False,
    )
    plt.tight_layout()
    plt.show()
```


**KRAS**




![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_40_1.png)




**APC**




![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_40_3.png)




```python
example_genes = ["KRAS", "APC"]
for gene in example_genes:
    _md_bold(gene)
    az.plot_forest(
        trace,
        var_names=["mu_a", "b", "d", "f", "h"],
        coords={"gene": [gene]},
        combined=True,
    )
    plt.tight_layout()
    plt.show()
```


**KRAS**




![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_41_1.png)




**APC**




![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_41_3.png)




```python
sgrnas_sample = trace.posterior.coords["sgrna"].values[:5]

az.plot_trace(trace, var_names="a", coords={"sgrna": sgrnas_sample}, compact=False)
plt.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_42_0.png)




```python
az.summary(trace, var_names="a", coords={"sgrna": sgrnas_sample})
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
      <th>a[AAAAAAATCCAGCAATGCAG]</th>
      <td>0.054</td>
      <td>0.061</td>
      <td>-0.046</td>
      <td>0.146</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6608.0</td>
      <td>3466.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[AAAAAACCCGTAGATAGCCT]</th>
      <td>0.105</td>
      <td>0.059</td>
      <td>0.016</td>
      <td>0.201</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4887.0</td>
      <td>2986.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[AAAAAAGAAGAAAAAACCAG]</th>
      <td>-0.312</td>
      <td>0.063</td>
      <td>-0.413</td>
      <td>-0.211</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7356.0</td>
      <td>3314.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[AAAAAAGCTCAAGAAGGAGG]</th>
      <td>0.124</td>
      <td>0.061</td>
      <td>0.031</td>
      <td>0.224</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5363.0</td>
      <td>3289.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>a[AAAAAAGGCTGTAAAAGCGT]</th>
      <td>-0.078</td>
      <td>0.061</td>
      <td>-0.175</td>
      <td>0.018</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4864.0</td>
      <td>2978.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
crc_post_summary.filter_string("var_name", "^sigma_*")
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
      <td>0.182</td>
      <td>0.001</td>
      <td>0.181</td>
      <td>0.183</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1168.0</td>
      <td>2194.0</td>
      <td>1.00</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sigma_mu_a</td>
      <td>0.204</td>
      <td>0.001</td>
      <td>0.202</td>
      <td>0.207</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1308.0</td>
      <td>2271.0</td>
      <td>1.00</td>
      <td>sigma_mu_a</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sigma_b</td>
      <td>0.020</td>
      <td>0.000</td>
      <td>0.019</td>
      <td>0.021</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1755.0</td>
      <td>2824.0</td>
      <td>1.00</td>
      <td>sigma_b</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sigma_d</td>
      <td>0.029</td>
      <td>0.000</td>
      <td>0.029</td>
      <td>0.030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1472.0</td>
      <td>2580.0</td>
      <td>1.00</td>
      <td>sigma_d</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sigma_f</td>
      <td>0.015</td>
      <td>0.002</td>
      <td>0.012</td>
      <td>0.018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>493.0</td>
      <td>2163.0</td>
      <td>1.01</td>
      <td>sigma_f</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sigma_h[APC]</td>
      <td>0.030</td>
      <td>0.001</td>
      <td>0.028</td>
      <td>0.031</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1501.0</td>
      <td>2552.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sigma_h[AXIN2]</td>
      <td>0.032</td>
      <td>0.001</td>
      <td>0.031</td>
      <td>0.034</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2096.0</td>
      <td>3146.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sigma_h[B2M]</td>
      <td>0.079</td>
      <td>0.001</td>
      <td>0.077</td>
      <td>0.081</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1533.0</td>
      <td>2644.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sigma_h[FBXW7]</td>
      <td>0.049</td>
      <td>0.001</td>
      <td>0.047</td>
      <td>0.050</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1851.0</td>
      <td>3096.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sigma_h[KRAS]</td>
      <td>0.048</td>
      <td>0.001</td>
      <td>0.047</td>
      <td>0.049</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1731.0</td>
      <td>2370.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sigma_h[MSH6]</td>
      <td>0.142</td>
      <td>0.001</td>
      <td>0.140</td>
      <td>0.144</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2087.0</td>
      <td>3039.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>15</th>
      <td>sigma_h[PIK3CA]</td>
      <td>0.107</td>
      <td>0.001</td>
      <td>0.105</td>
      <td>0.108</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2191.0</td>
      <td>3228.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>16</th>
      <td>sigma_h[POLD1]</td>
      <td>0.094</td>
      <td>0.002</td>
      <td>0.092</td>
      <td>0.097</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3197.0</td>
      <td>2727.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
    <tr>
      <th>17</th>
      <td>sigma_h[UBR5]</td>
      <td>0.047</td>
      <td>0.002</td>
      <td>0.044</td>
      <td>0.050</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2557.0</td>
      <td>2971.0</td>
      <td>1.00</td>
      <td>sigma_h</td>
    </tr>
  </tbody>
</table>
</div>




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
      <td>0.508</td>
      <td>0.016</td>
      <td>0.484</td>
      <td>0.535</td>
      <td>0</td>
      <td>1</td>
      <td>mu_a</td>
      <td>b</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 2]</th>
      <td>-0.371</td>
      <td>0.012</td>
      <td>-0.391</td>
      <td>-0.352</td>
      <td>0</td>
      <td>2</td>
      <td>mu_a</td>
      <td>d</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 3]</th>
      <td>-0.237</td>
      <td>0.076</td>
      <td>-0.353</td>
      <td>-0.114</td>
      <td>0</td>
      <td>3</td>
      <td>mu_a</td>
      <td>f</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[0, 4]</th>
      <td>-0.331</td>
      <td>0.022</td>
      <td>-0.367</td>
      <td>-0.296</td>
      <td>0</td>
      <td>4</td>
      <td>mu_a</td>
      <td>h[APC]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[12, 8]</th>
      <td>-0.427</td>
      <td>0.038</td>
      <td>-0.484</td>
      <td>-0.365</td>
      <td>12</td>
      <td>8</td>
      <td>h[UBR5]</td>
      <td>h[KRAS]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[12, 9]</th>
      <td>-0.339</td>
      <td>0.032</td>
      <td>-0.389</td>
      <td>-0.289</td>
      <td>12</td>
      <td>9</td>
      <td>h[UBR5]</td>
      <td>h[MSH6]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[12, 10]</th>
      <td>0.452</td>
      <td>0.032</td>
      <td>0.403</td>
      <td>0.503</td>
      <td>12</td>
      <td>10</td>
      <td>h[UBR5]</td>
      <td>h[PIK3CA]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[12, 11]</th>
      <td>0.176</td>
      <td>0.041</td>
      <td>0.110</td>
      <td>0.239</td>
      <td>12</td>
      <td>11</td>
      <td>h[UBR5]</td>
      <td>h[POLD1]</td>
    </tr>
    <tr>
      <th>genes_chol_cov_corr[12, 12]</th>
      <td>1.000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>12</td>
      <td>12</td>
      <td>h[UBR5]</td>
      <td>h[UBR5]</td>
    </tr>
  </tbody>
</table>
<p>169 rows × 8 columns</p>
</div>




```python
plot_df = gene_corr_post.pivot_wider("p1", "p2", "mean").set_index("p1")
sns.heatmap(plot_df, cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_46_0.png)




```python
plot_df = gene_corr_post.pivot_wider("p1", "p2", "mean").set_index("p1")
sns.clustermap(plot_df, cmap="seismic", vmin=-1, vmax=1, figsize=(6, 6))
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_47_0.png)




```python
cancer_genes = trace.posterior.coords["cancer_gene"].values.tolist()
cancer_gene_mutants = (
    valid_crc_data.filter_column_isin("hugo_symbol", cancer_genes)[
        ["hugo_symbol", "depmap_id", "is_mutated"]
    ]
    .drop_duplicates()
    .pivot_wider("depmap_id", names_from="hugo_symbol", values_from="is_mutated")
    .set_index("depmap_id")
)
cg = sns.clustermap(
    cancer_gene_mutants, cmap="Greys", xticklabels=1, yticklabels=1, figsize=(4, 7)
)
cg.ax_heatmap.tick_params("both", length=0)
cg.ax_cbar.remove()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_48_0.png)




```python
h_post_summary = (
    crc_post_summary.query("var_name == 'h'")
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
      <td>h[A1BG, APC]</td>
      <td>-0.004</td>
      <td>0.024</td>
      <td>-0.042</td>
      <td>0.036</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>8417.0</td>
      <td>2974.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>A1BG</td>
      <td>APC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>h[A1BG, AXIN2]</td>
      <td>0.008</td>
      <td>0.027</td>
      <td>-0.033</td>
      <td>0.053</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>9075.0</td>
      <td>2864.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>A1BG</td>
      <td>AXIN2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>h[A1BG, B2M]</td>
      <td>-0.020</td>
      <td>0.058</td>
      <td>-0.113</td>
      <td>0.070</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>11295.0</td>
      <td>2671.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>A1BG</td>
      <td>B2M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>h[A1BG, FBXW7]</td>
      <td>0.026</td>
      <td>0.037</td>
      <td>-0.035</td>
      <td>0.084</td>
      <td>0.000</td>
      <td>0.001</td>
      <td>11959.0</td>
      <td>2767.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>A1BG</td>
      <td>FBXW7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>h[A1BG, KRAS]</td>
      <td>0.013</td>
      <td>0.025</td>
      <td>-0.027</td>
      <td>0.053</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5381.0</td>
      <td>3441.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>A1BG</td>
      <td>KRAS</td>
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
      <th>113783</th>
      <td>h[RAN, MSH6]</td>
      <td>-0.935</td>
      <td>0.052</td>
      <td>-1.019</td>
      <td>-0.852</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2690.0</td>
      <td>2749.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>RAN</td>
      <td>MSH6</td>
    </tr>
    <tr>
      <th>38678</th>
      <td>h[DONSON, MSH6]</td>
      <td>-0.855</td>
      <td>0.053</td>
      <td>-0.944</td>
      <td>-0.772</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2946.0</td>
      <td>3117.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>DONSON</td>
      <td>MSH6</td>
    </tr>
    <tr>
      <th>119741</th>
      <td>h[RPSA, MSH6]</td>
      <td>-0.846</td>
      <td>0.054</td>
      <td>-0.938</td>
      <td>-0.765</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2618.0</td>
      <td>2965.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>RPSA</td>
      <td>MSH6</td>
    </tr>
    <tr>
      <th>110831</th>
      <td>h[PSMD7, MSH6]</td>
      <td>-0.838</td>
      <td>0.054</td>
      <td>-0.921</td>
      <td>-0.751</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2952.0</td>
      <td>3137.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>PSMD7</td>
      <td>MSH6</td>
    </tr>
    <tr>
      <th>106484</th>
      <td>h[POLR2L, MSH6]</td>
      <td>-0.833</td>
      <td>0.053</td>
      <td>-0.915</td>
      <td>-0.745</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2937.0</td>
      <td>2991.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>POLR2L</td>
      <td>MSH6</td>
    </tr>
    <tr>
      <th>38679</th>
      <td>h[DONSON, PIK3CA]</td>
      <td>0.624</td>
      <td>0.045</td>
      <td>0.556</td>
      <td>0.700</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3509.0</td>
      <td>3278.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>DONSON</td>
      <td>PIK3CA</td>
    </tr>
    <tr>
      <th>119742</th>
      <td>h[RPSA, PIK3CA]</td>
      <td>0.624</td>
      <td>0.045</td>
      <td>0.559</td>
      <td>0.700</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3784.0</td>
      <td>3236.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>RPSA</td>
      <td>PIK3CA</td>
    </tr>
    <tr>
      <th>110832</th>
      <td>h[PSMD7, PIK3CA]</td>
      <td>0.627</td>
      <td>0.045</td>
      <td>0.554</td>
      <td>0.697</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3773.0</td>
      <td>3040.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>PSMD7</td>
      <td>PIK3CA</td>
    </tr>
    <tr>
      <th>106485</th>
      <td>h[POLR2L, PIK3CA]</td>
      <td>0.631</td>
      <td>0.044</td>
      <td>0.563</td>
      <td>0.704</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>3964.0</td>
      <td>3424.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>POLR2L</td>
      <td>PIK3CA</td>
    </tr>
    <tr>
      <th>113784</th>
      <td>h[RAN, PIK3CA]</td>
      <td>0.727</td>
      <td>0.045</td>
      <td>0.655</td>
      <td>0.798</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3450.0</td>
      <td>3049.0</td>
      <td>1.0</td>
      <td>h</td>
      <td>RAN</td>
      <td>PIK3CA</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.kdeplot(data=h_post_summary, x="mean", hue="cancer_gene")
ax.set_xlabel(r"$\bar{h}_g$ posterior")
ax.set_ylabel("density")
ax.get_legend().remove()  # set_title("cancer gene\ncomut.")
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_51_0.png)




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
        valid_crc_data.filter_column_isin("hugo_symbol", h_hits)
        .merge(cancer_gene_mutants.reset_index(), on="depmap_id")
        .reset_index()
        .astype({"hugo_symbol": str})
        .assign(
            hugo_symbol=lambda d: pd.Categorical(d["hugo_symbol"], categories=h_hits),
            _cg_mut=lambda d: d[cg].map({True: "mut.", False: "WT"}),
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



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_52_0.png)




```python
top_n = 5
top_b_hits = (
    crc_post_summary.query("var_name == 'b'")
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
        obs_data = crc_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(data=obs_data, x="rna_expr", y="lfc", ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("log RNA expression")
fig.supylabel("log-fold change")

fig.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_53_0.png)




```python
top_n = 5
top_d_hits = (
    crc_post_summary.query("var_name == 'd'")
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
        obs_data = crc_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(data=obs_data, x="copy_number", y="lfc", ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("copy number")
fig.supylabel("log-fold change")
fig.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_54_0.png)



## PPC


```python
np.random.seed(99)
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




    (40, 2842455)




```python
fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharex=False, sharey=False)
ax1 = axes[0]
ax2 = axes[1]
hist_kwargs = {"stat": "count", "element": "step", "fill": False}

pp_avg = trace.posterior_predictive["ct_final"].mean(axis=(0, 1))

bw = 0.1
for i in range(example_ppc_draws.shape[0]):
    sns.histplot(
        x=np.log10(example_ppc_draws[i, :] + 1),
        alpha=0.2,
        binwidth=bw,
        color="tab:blue",
        ax=ax1,
        **hist_kwargs,
    )

sns.histplot(
    x=np.log10(pp_avg + 1),
    color="tab:orange",
    binwidth=bw,
    ax=ax1,
    alpha=0.5,
    **hist_kwargs,
)
sns.histplot(
    x=np.log10(valid_crc_data["counts_final"] + 1),
    color="k",
    binwidth=bw,
    ax=ax1,
    **hist_kwargs,
)
ax1.set_xlabel("log10(counts final + 1)")
ax1.set_ylabel("density")

bw = 20
for i in range(example_ppc_draws.shape[0]):
    sns.histplot(
        x=example_ppc_draws[i, :],
        alpha=0.2,
        binwidth=bw,
        color="tab:blue",
        ax=ax2,
        **hist_kwargs,
    )

sns.histplot(
    x=pp_avg, color="tab:orange", alpha=0.5, ax=ax2, binwidth=bw, **hist_kwargs
)
sns.histplot(
    x=valid_crc_data["counts_final"], color="k", binwidth=bw, ax=ax2, **hist_kwargs
)
ax2.set_xlabel("counts final")
ax2.set_ylabel("density")
ax2.set_xlim(0, 1000)

fig.suptitle("PPC")
fig.tight_layout()
plt.show()
```



![png](028_single-lineage-colorectal-inspection_005_files/028_single-lineage-colorectal-inspection_005_57_0.png)



---

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 17.87 minutes



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

    pandas    : 1.4.3
    seaborn   : 0.11.2
    numpy     : 1.23.1
    matplotlib: 3.5.2
    arviz     : 0.12.1




```python

```
