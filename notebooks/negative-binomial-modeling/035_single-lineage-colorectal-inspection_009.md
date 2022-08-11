# Inspect the single-lineage model run on the colorectal data (009)

 Model attributes:

- sgRNA | gene varying intercept
- RNA and CN varying effects per gene
- correlation between gene varying effects modeled using the multivariate normal and Cholesky decomposition (non-centered parameterization)
- target gene mutation variable and cancer gene comutation variable.
- varying intercept copy number effect for chromosome nested under cell line


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
from matplotlib.lines import Line2D
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
saved_model_dir = models_dir() / "hnb-single-lineage-colorectal-009_PYMC_NUMPYRO"
```


```python
with open(saved_model_dir / "description.txt") as f:
    model_description = "".join(list(f))

print(model_description)
```

    config. name: 'hnb-single-lineage-colorectal-009'
    model name: 'LineageHierNegBinomModel'
    model version: '0.1.3'
    model description: A hierarchical negative binomial generalized linear model for one lineage.
    fit method: 'PYMC_NUMPYRO'

    --------------------------------------------------------------------------------

    CONFIGURATION

    {
        "name": "hnb-single-lineage-colorectal-009",
        "description": " Single lineage hierarchical negative binomial model for colorectal data from the Broad. Varying effects for each chromosome of each cell line has been nested under the existing varying effects for cell line. ",
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
                                    delta_cells_dim_0: 2, delta_cells_dim_1: 40,
                                    cell_chrom: 920, genes_chol_cov_dim_0: 91,
                                    cells_chol_cov_dim_0: 3,
                                    genes_chol_cov_corr_dim_0: 13,
                                    genes_chol_cov_corr_dim_1: 13,
                                    genes_chol_cov_stds_dim_0: 13, cancer_gene: 9,
                                    gene: 18119, cells_chol_cov_corr_dim_0: 2,
                                    cells_chol_cov_corr_dim_1: 2,
                                    cells_chol_cov_stds_dim_0: 2, cell_line: 40)
    Coordinates: (12/19)
      * chain                      (chain) int64 0 1 2 3
      * draw                       (draw) int64 0 1 2 3 4 5 ... 995 996 997 998 999
      * delta_genes_dim_0          (delta_genes_dim_0) int64 0 1 2 3 ... 9 10 11 12
      * delta_genes_dim_1          (delta_genes_dim_1) int64 0 1 2 ... 18117 18118
      * sgrna                      (sgrna) object 'AAAAAAATCCAGCAATGCAG' ... 'TTT...
      * delta_cells_dim_0          (delta_cells_dim_0) int64 0 1
        ...                         ...
      * cancer_gene                (cancer_gene) object 'APC' 'AXIN2' ... 'UBR5'
      * gene                       (gene) object 'A1BG' 'A1CF' ... 'ZZEF1' 'ZZZ3'
      * cells_chol_cov_corr_dim_0  (cells_chol_cov_corr_dim_0) int64 0 1
      * cells_chol_cov_corr_dim_1  (cells_chol_cov_corr_dim_1) int64 0 1
      * cells_chol_cov_stds_dim_0  (cells_chol_cov_stds_dim_0) int64 0 1
      * cell_line                  (cell_line) object 'ACH-000007' ... 'ACH-002025'
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
        created_at:              2022-08-10 16:56:36.110920
        arviz_version:           0.12.1
        model_name:              LineageHierNegBinomModel
        model_version:           0.1.3
        model_doc:               A hierarchical negative binomial generalized lin...
        previous_created_at:     ['2022-08-10 16:56:36.110920', '2022-08-09T10:34...
        combined_model_version:  ['0.1.3', '0.1.2', '0.1.2', '0.1.2']

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
        created_at:           2022-08-10 16:56:36.110920
        arviz_version:        0.12.1
        previous_created_at:  ['2022-08-10 16:56:36.110920', '2022-08-09T10:34:51...

    --------------------------------------------------------------------------------

    MCMC DESCRIPTION

    date created: 2022-08-10 16:56
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.673, 0.71, 0.772, 0.802
    avg. step size: 0.008, 0.01, 0.009, 0.004
    avg. accept prob.: 0.99, 0.985, 0.988, 0.993
    avg. tree depth: 9.0, 9.0, 9.0, 10.0


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
      <td>0.090</td>
      <td>0.004</td>
      <td>0.084</td>
      <td>0.096</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>344.0</td>
      <td>578.0</td>
      <td>1.02</td>
      <td>mu_mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_b</td>
      <td>-0.001</td>
      <td>0.000</td>
      <td>-0.001</td>
      <td>-0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3297.0</td>
      <td>3353.0</td>
      <td>1.00</td>
      <td>mu_b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_mu_m</td>
      <td>-0.184</td>
      <td>0.025</td>
      <td>-0.226</td>
      <td>-0.146</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>606.0</td>
      <td>1205.0</td>
      <td>1.01</td>
      <td>mu_mu_m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sigma_a</td>
      <td>0.183</td>
      <td>0.001</td>
      <td>0.182</td>
      <td>0.184</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>994.0</td>
      <td>1990.0</td>
      <td>1.00</td>
      <td>sigma_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sigma_k</td>
      <td>0.031</td>
      <td>0.001</td>
      <td>0.029</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1059.0</td>
      <td>1950.0</td>
      <td>1.00</td>
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

### CRC data


```python
crc_dm = CrisprScreenDataManager(
    modeling_data_dir()
    / "lineage-modeling-data"
    / "depmap-modeling-data_colorectal.csv",
    transformations=[broad_only],
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
crc_mdl_data = crc_model.make_data_structure(valid_crc_data)
```

    [INFO] 2022-08-11 06:14:09 [(lineage_hierarchical_nb.py:data_processing_pipeline:329] Processing data for modeling.
    [INFO] 2022-08-11 06:14:09 [(lineage_hierarchical_nb.py:data_processing_pipeline:330] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-08-11 06:15:58 [(lineage_hierarchical_nb.py:data_processing_pipeline:388] number of data points dropped: 25
    [INFO] 2022-08-11 06:16:00 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:627] number of genes mutated in all cells lines: 0
    [DEBUG] 2022-08-11 06:16:00 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:630] Genes always mutated:
    [DEBUG] 2022-08-11 06:16:03 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:68] all_mut: {}
    [INFO] 2022-08-11 06:16:04 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:77] Dropping 21 cancer genes.
    [DEBUG] 2022-08-11 06:16:04 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:79] Dropped cancer genes: ['AKT1', 'AXIN1', 'BAX', 'ERBB3', 'GRIN2A', 'HIF1A', 'MAP2K1', 'MAX', 'MDM2', 'MLH1', 'MSH2', 'PIK3R1', 'POLE', 'PTPRT', 'SALL4', 'SFRP4', 'SMAD2', 'SMAD3', 'SMAD4', 'SRC', 'TGFBR2']



```python
crc_mdl_data.coords["cancer_gene"]
```




    ['APC', 'AXIN2', 'B2M', 'FBXW7', 'KRAS', 'MSH6', 'PIK3CA', 'POLD1', 'UBR5']



## Analysis


```python
sns.histplot(x=crc_post_summary["r_hat"], binwidth=0.01, stat="proportion");
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_21_0.png)




```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=crc_post_summary, x="var_name", y="r_hat", ax=ax)
ax.tick_params(rotation=90)
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_22_0.png)




```python
sns.histplot(data=crc_post_summary, x="ess_bulk", binwidth=500);
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_23_0.png)




```python
crc_post_summary.query("ess_bulk < 400 and ess_tail < 400")
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
  </tbody>
</table>
</div>




```python
az.plot_energy(trace);
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_25_0.png)




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



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_26_0.png)




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
      <td>0.008000</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.989615</td>
      <td>1.910692e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.010022</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.984676</td>
      <td>1.910693e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.008725</td>
      <td>511.0</td>
      <td>9.0</td>
      <td>0.988467</td>
      <td>1.910689e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.004137</td>
      <td>1023.0</td>
      <td>10.0</td>
      <td>0.992916</td>
      <td>1.910687e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(trace, var_names=["mu_mu_a", "mu_b", "mu_mu_m"], compact=False)
plt.tight_layout()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_28_0.png)




```python
az.plot_trace(trace, var_names=["^sigma_*"], filter_vars="regex", compact=True)
plt.tight_layout()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_29_0.png)




```python
az.summary(trace, var_names=["^sigma_.*$"], filter_vars="regex")
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
      <th>sigma_a</th>
      <td>0.183</td>
      <td>0.001</td>
      <td>0.182</td>
      <td>0.184</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>994.0</td>
      <td>1990.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_k</th>
      <td>0.031</td>
      <td>0.001</td>
      <td>0.029</td>
      <td>0.032</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1059.0</td>
      <td>1950.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_m</th>
      <td>0.163</td>
      <td>0.007</td>
      <td>0.152</td>
      <td>0.173</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1124.0</td>
      <td>1546.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_mu_a</th>
      <td>0.200</td>
      <td>0.001</td>
      <td>0.197</td>
      <td>0.202</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>988.0</td>
      <td>1696.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_b</th>
      <td>0.019</td>
      <td>0.000</td>
      <td>0.018</td>
      <td>0.019</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1539.0</td>
      <td>2329.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_d</th>
      <td>0.217</td>
      <td>0.003</td>
      <td>0.212</td>
      <td>0.222</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1223.0</td>
      <td>2639.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_f</th>
      <td>0.016</td>
      <td>0.002</td>
      <td>0.013</td>
      <td>0.019</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1478.0</td>
      <td>2563.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[APC]</th>
      <td>0.028</td>
      <td>0.001</td>
      <td>0.027</td>
      <td>0.030</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1243.0</td>
      <td>2082.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[AXIN2]</th>
      <td>0.032</td>
      <td>0.001</td>
      <td>0.031</td>
      <td>0.034</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1932.0</td>
      <td>2965.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[B2M]</th>
      <td>0.082</td>
      <td>0.001</td>
      <td>0.080</td>
      <td>0.084</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1819.0</td>
      <td>2877.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[FBXW7]</th>
      <td>0.050</td>
      <td>0.001</td>
      <td>0.048</td>
      <td>0.051</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2123.0</td>
      <td>3080.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[KRAS]</th>
      <td>0.047</td>
      <td>0.001</td>
      <td>0.046</td>
      <td>0.048</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1602.0</td>
      <td>2792.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[MSH6]</th>
      <td>0.138</td>
      <td>0.001</td>
      <td>0.136</td>
      <td>0.140</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2077.0</td>
      <td>3115.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[PIK3CA]</th>
      <td>0.105</td>
      <td>0.001</td>
      <td>0.104</td>
      <td>0.106</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1838.0</td>
      <td>3305.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[POLD1]</th>
      <td>0.092</td>
      <td>0.002</td>
      <td>0.090</td>
      <td>0.095</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2569.0</td>
      <td>2777.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_h[UBR5]</th>
      <td>0.048</td>
      <td>0.002</td>
      <td>0.045</td>
      <td>0.050</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2069.0</td>
      <td>3179.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_mu_k</th>
      <td>0.020</td>
      <td>0.003</td>
      <td>0.016</td>
      <td>0.024</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>757.0</td>
      <td>1389.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma_mu_m</th>
      <td>0.152</td>
      <td>0.019</td>
      <td>0.123</td>
      <td>0.183</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1514.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




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
      <td>0.199535</td>
      <td>0.018873</td>
      <td>0.217379</td>
      <td>0.016489</td>
      <td>0.030676</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.199652</td>
      <td>0.018876</td>
      <td>0.217482</td>
      <td>0.016022</td>
      <td>0.030665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.199614</td>
      <td>0.018891</td>
      <td>0.217424</td>
      <td>0.016242</td>
      <td>0.030736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.199555</td>
      <td>0.018858</td>
      <td>0.217409</td>
      <td>0.016153</td>
      <td>0.030736</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.plot_trace(trace, var_names=["alpha"], compact=False)
plt.tight_layout()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_32_0.png)




```python
az.plot_forest(
    trace, var_names=["^sigma_*"], filter_vars="regex", combined=False, figsize=(5, 5)
)
plt.tight_layout()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_33_0.png)




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



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_34_0.png)




```python
sgrna_to_gene_map = (
    crc_data.copy()[["hugo_symbol", "sgrna"]].drop_duplicates().reset_index(drop=True)
)
```


```python
for v in ["mu_a", "b", "d", "f", "h", "k", "m"]:
    display(Markdown(f"variable: **{v}**"))
    top = (
        crc_post_summary.query(f"var_name == '{v}'")
        .sort_values("mean")
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
      <th>12662</th>
      <td>mu_a[RAN]</td>
      <td>-1.182</td>
      <td>0.074</td>
      <td>-1.305</td>
      <td>-1.069</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2085.0</td>
      <td>2795.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>12334</th>
      <td>mu_a[PSMD7]</td>
      <td>-1.045</td>
      <td>0.078</td>
      <td>-1.167</td>
      <td>-0.920</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2327.0</td>
      <td>2711.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>11851</th>
      <td>mu_a[POLR2L]</td>
      <td>-1.029</td>
      <td>0.075</td>
      <td>-1.145</td>
      <td>-0.905</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2279.0</td>
      <td>2894.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>13324</th>
      <td>mu_a[RPSA]</td>
      <td>-1.025</td>
      <td>0.074</td>
      <td>-1.141</td>
      <td>-0.908</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1957.0</td>
      <td>2750.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>7794</th>
      <td>mu_a[KIF11]</td>
      <td>-1.023</td>
      <td>0.076</td>
      <td>-1.136</td>
      <td>-0.891</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2114.0</td>
      <td>2531.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>16409</th>
      <td>mu_a[TRIQK]</td>
      <td>0.390</td>
      <td>0.076</td>
      <td>0.268</td>
      <td>0.510</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1529.0</td>
      <td>2075.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>3838</th>
      <td>mu_a[DAB2IP]</td>
      <td>0.394</td>
      <td>0.075</td>
      <td>0.276</td>
      <td>0.512</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1343.0</td>
      <td>1617.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>13179</th>
      <td>mu_a[ROCK2]</td>
      <td>0.394</td>
      <td>0.077</td>
      <td>0.273</td>
      <td>0.522</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>1167.0</td>
      <td>1813.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>8821</th>
      <td>mu_a[MAPK9]</td>
      <td>0.407</td>
      <td>0.084</td>
      <td>0.275</td>
      <td>0.541</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>1495.0</td>
      <td>2360.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>17907</th>
      <td>mu_a[ZNF611]</td>
      <td>0.412</td>
      <td>0.072</td>
      <td>0.297</td>
      <td>0.525</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1737.0</td>
      <td>2207.0</td>
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
      <th>31286</th>
      <td>b[RNGTT]</td>
      <td>-0.085</td>
      <td>0.015</td>
      <td>-0.108</td>
      <td>-0.060</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5630.0</td>
      <td>2761.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18808</th>
      <td>b[ANKLE2]</td>
      <td>-0.080</td>
      <td>0.014</td>
      <td>-0.104</td>
      <td>-0.058</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6631.0</td>
      <td>2840.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>30885</th>
      <td>b[RBM22]</td>
      <td>-0.079</td>
      <td>0.015</td>
      <td>-0.103</td>
      <td>-0.055</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6820.0</td>
      <td>3305.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>30421</th>
      <td>b[PSMA6]</td>
      <td>-0.076</td>
      <td>0.015</td>
      <td>-0.099</td>
      <td>-0.052</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8107.0</td>
      <td>2756.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>31945</th>
      <td>b[SF1]</td>
      <td>-0.072</td>
      <td>0.014</td>
      <td>-0.094</td>
      <td>-0.049</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5565.0</td>
      <td>2923.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>24040</th>
      <td>b[GET4]</td>
      <td>0.034</td>
      <td>0.015</td>
      <td>0.011</td>
      <td>0.059</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6545.0</td>
      <td>3058.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>33739</th>
      <td>b[TGFB1]</td>
      <td>0.035</td>
      <td>0.014</td>
      <td>0.013</td>
      <td>0.058</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8724.0</td>
      <td>3118.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18750</th>
      <td>b[AMIGO2]</td>
      <td>0.040</td>
      <td>0.014</td>
      <td>0.018</td>
      <td>0.063</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7791.0</td>
      <td>2925.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>20821</th>
      <td>b[CDKN1C]</td>
      <td>0.043</td>
      <td>0.015</td>
      <td>0.020</td>
      <td>0.067</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8461.0</td>
      <td>2371.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>20819</th>
      <td>b[CDKN1A]</td>
      <td>0.053</td>
      <td>0.014</td>
      <td>0.030</td>
      <td>0.075</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7400.0</td>
      <td>2915.0</td>
      <td>1.0</td>
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
      <th>50220</th>
      <td>d[SIK3]</td>
      <td>-0.643</td>
      <td>0.149</td>
      <td>-0.885</td>
      <td>-0.403</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>6385.0</td>
      <td>2872.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>38696</th>
      <td>d[CCND1]</td>
      <td>-0.613</td>
      <td>0.157</td>
      <td>-0.876</td>
      <td>-0.377</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>5985.0</td>
      <td>2781.0</td>
      <td>1.01</td>
      <td>d</td>
    </tr>
    <tr>
      <th>41089</th>
      <td>d[ERBB2]</td>
      <td>-0.535</td>
      <td>0.163</td>
      <td>-0.788</td>
      <td>-0.271</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6651.0</td>
      <td>2954.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>48549</th>
      <td>d[PSMB5]</td>
      <td>-0.534</td>
      <td>0.166</td>
      <td>-0.796</td>
      <td>-0.269</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>4051.0</td>
      <td>2865.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>40046</th>
      <td>d[CYP4F11]</td>
      <td>-0.431</td>
      <td>0.152</td>
      <td>-0.691</td>
      <td>-0.205</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>7046.0</td>
      <td>2631.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>39250</th>
      <td>d[CIAO3]</td>
      <td>0.983</td>
      <td>0.149</td>
      <td>0.749</td>
      <td>1.220</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>6702.0</td>
      <td>2896.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>53246</th>
      <td>d[VCP]</td>
      <td>0.992</td>
      <td>0.165</td>
      <td>0.701</td>
      <td>1.231</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6143.0</td>
      <td>2574.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>50766</th>
      <td>d[SMU1]</td>
      <td>0.993</td>
      <td>0.163</td>
      <td>0.724</td>
      <td>1.239</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5133.0</td>
      <td>3011.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>42216</th>
      <td>d[GINS2]</td>
      <td>1.014</td>
      <td>0.170</td>
      <td>0.750</td>
      <td>1.287</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>6242.0</td>
      <td>2990.0</td>
      <td>1.00</td>
      <td>d</td>
    </tr>
    <tr>
      <th>47945</th>
      <td>d[PLK1]</td>
      <td>1.216</td>
      <td>0.176</td>
      <td>0.945</td>
      <td>1.502</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5677.0</td>
      <td>2910.0</td>
      <td>1.00</td>
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
      <th>63996</th>
      <td>f[MYH9]</td>
      <td>-0.039</td>
      <td>0.015</td>
      <td>-0.063</td>
      <td>-0.016</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1353.0</td>
      <td>2688.0</td>
      <td>1.01</td>
      <td>f</td>
    </tr>
    <tr>
      <th>55741</th>
      <td>f[BAG3]</td>
      <td>-0.036</td>
      <td>0.014</td>
      <td>-0.059</td>
      <td>-0.013</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1109.0</td>
      <td>2860.0</td>
      <td>1.01</td>
      <td>f</td>
    </tr>
    <tr>
      <th>68416</th>
      <td>f[SLC13A4]</td>
      <td>-0.034</td>
      <td>0.014</td>
      <td>-0.056</td>
      <td>-0.012</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2659.0</td>
      <td>2838.0</td>
      <td>1.00</td>
      <td>f</td>
    </tr>
    <tr>
      <th>62331</th>
      <td>f[KRAS]</td>
      <td>-0.034</td>
      <td>0.017</td>
      <td>-0.059</td>
      <td>-0.007</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>515.0</td>
      <td>2735.0</td>
      <td>1.01</td>
      <td>f</td>
    </tr>
    <tr>
      <th>67261</th>
      <td>f[RFFL]</td>
      <td>-0.029</td>
      <td>0.014</td>
      <td>-0.052</td>
      <td>-0.008</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2375.0</td>
      <td>2571.0</td>
      <td>1.00</td>
      <td>f</td>
    </tr>
    <tr>
      <th>58201</th>
      <td>f[DAD1]</td>
      <td>0.051</td>
      <td>0.016</td>
      <td>0.025</td>
      <td>0.075</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>839.0</td>
      <td>2691.0</td>
      <td>1.01</td>
      <td>f</td>
    </tr>
    <tr>
      <th>68932</th>
      <td>f[SNRPD1]</td>
      <td>0.051</td>
      <td>0.016</td>
      <td>0.026</td>
      <td>0.076</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2009.0</td>
      <td>3190.0</td>
      <td>1.01</td>
      <td>f</td>
    </tr>
    <tr>
      <th>69278</th>
      <td>f[SS18L2]</td>
      <td>0.053</td>
      <td>0.016</td>
      <td>0.031</td>
      <td>0.080</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3236.0</td>
      <td>2603.0</td>
      <td>1.00</td>
      <td>f</td>
    </tr>
    <tr>
      <th>56041</th>
      <td>f[BUB3]</td>
      <td>0.053</td>
      <td>0.016</td>
      <td>0.027</td>
      <td>0.078</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4769.0</td>
      <td>3114.0</td>
      <td>1.00</td>
      <td>f</td>
    </tr>
    <tr>
      <th>71021</th>
      <td>f[TXNL4A]</td>
      <td>0.058</td>
      <td>0.017</td>
      <td>0.030</td>
      <td>0.084</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>1104.0</td>
      <td>2926.0</td>
      <td>1.01</td>
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
      <th>257341</th>
      <td>h[RAN, MSH6]</td>
      <td>-0.904</td>
      <td>0.052</td>
      <td>-0.981</td>
      <td>-0.816</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2149.0</td>
      <td>2975.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>182236</th>
      <td>h[DONSON, MSH6]</td>
      <td>-0.819</td>
      <td>0.053</td>
      <td>-0.907</td>
      <td>-0.738</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2139.0</td>
      <td>2780.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>263299</th>
      <td>h[RPSA, MSH6]</td>
      <td>-0.813</td>
      <td>0.052</td>
      <td>-0.894</td>
      <td>-0.731</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1986.0</td>
      <td>2675.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>250042</th>
      <td>h[POLR2L, MSH6]</td>
      <td>-0.804</td>
      <td>0.053</td>
      <td>-0.885</td>
      <td>-0.719</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2464.0</td>
      <td>3011.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>254389</th>
      <td>h[PSMD7, MSH6]</td>
      <td>-0.795</td>
      <td>0.054</td>
      <td>-0.883</td>
      <td>-0.711</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2438.0</td>
      <td>2637.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>213530</th>
      <td>h[KIF11, PIK3CA]</td>
      <td>0.605</td>
      <td>0.045</td>
      <td>0.531</td>
      <td>0.677</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3089.0</td>
      <td>2939.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>182237</th>
      <td>h[DONSON, PIK3CA]</td>
      <td>0.609</td>
      <td>0.045</td>
      <td>0.538</td>
      <td>0.681</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3111.0</td>
      <td>3042.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>263300</th>
      <td>h[RPSA, PIK3CA]</td>
      <td>0.612</td>
      <td>0.043</td>
      <td>0.544</td>
      <td>0.681</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2850.0</td>
      <td>2891.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>250043</th>
      <td>h[POLR2L, PIK3CA]</td>
      <td>0.622</td>
      <td>0.044</td>
      <td>0.554</td>
      <td>0.690</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3221.0</td>
      <td>2932.0</td>
      <td>1.0</td>
      <td>h</td>
    </tr>
    <tr>
      <th>257342</th>
      <td>h[RAN, PIK3CA]</td>
      <td>0.711</td>
      <td>0.045</td>
      <td>0.645</td>
      <td>0.787</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>2979.0</td>
      <td>3242.0</td>
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
      <th>307509</th>
      <td>k[ACH-001460__17]</td>
      <td>-0.175</td>
      <td>0.009</td>
      <td>-0.188</td>
      <td>-0.161</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1190.0</td>
      <td>2286.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307102</th>
      <td>k[ACH-000683__1]</td>
      <td>-0.168</td>
      <td>0.008</td>
      <td>-0.180</td>
      <td>-0.155</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>810.0</td>
      <td>1547.0</td>
      <td>1.01</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307508</th>
      <td>k[ACH-001460__16]</td>
      <td>-0.136</td>
      <td>0.018</td>
      <td>-0.165</td>
      <td>-0.106</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2500.0</td>
      <td>3015.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307578</th>
      <td>k[ACH-002023__17]</td>
      <td>-0.130</td>
      <td>0.008</td>
      <td>-0.143</td>
      <td>-0.117</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1438.0</td>
      <td>2194.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307250</th>
      <td>k[ACH-000957__11]</td>
      <td>-0.116</td>
      <td>0.007</td>
      <td>-0.128</td>
      <td>-0.105</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1242.0</td>
      <td>2602.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>306906</th>
      <td>k[ACH-000381__12]</td>
      <td>0.094</td>
      <td>0.013</td>
      <td>0.073</td>
      <td>0.114</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3099.0</td>
      <td>2886.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307568</th>
      <td>k[ACH-002023__7]</td>
      <td>0.094</td>
      <td>0.009</td>
      <td>0.078</td>
      <td>0.108</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1706.0</td>
      <td>2707.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307427</th>
      <td>k[ACH-001454__4]</td>
      <td>0.097</td>
      <td>0.009</td>
      <td>0.083</td>
      <td>0.111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1489.0</td>
      <td>2386.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>306753</th>
      <td>k[ACH-000009__20]</td>
      <td>0.099</td>
      <td>0.015</td>
      <td>0.074</td>
      <td>0.122</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2604.0</td>
      <td>3032.0</td>
      <td>1.00</td>
      <td>k</td>
    </tr>
    <tr>
      <th>307565</th>
      <td>k[ACH-002023__4]</td>
      <td>0.137</td>
      <td>0.009</td>
      <td>0.124</td>
      <td>0.151</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1616.0</td>
      <td>2853.0</td>
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
      <th>308360</th>
      <td>m[ACH-001454__17]</td>
      <td>-1.313</td>
      <td>0.058</td>
      <td>-1.412</td>
      <td>-1.225</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4554.0</td>
      <td>2817.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>307991</th>
      <td>m[ACH-000651__16]</td>
      <td>-1.274</td>
      <td>0.034</td>
      <td>-1.330</td>
      <td>-1.222</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4995.0</td>
      <td>3783.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>308345</th>
      <td>m[ACH-001454__2]</td>
      <td>-1.107</td>
      <td>0.046</td>
      <td>-1.181</td>
      <td>-1.036</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5357.0</td>
      <td>2661.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>308354</th>
      <td>m[ACH-001454__11]</td>
      <td>-1.050</td>
      <td>0.033</td>
      <td>-1.106</td>
      <td>-1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5683.0</td>
      <td>3329.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>308357</th>
      <td>m[ACH-001454__14]</td>
      <td>-1.015</td>
      <td>0.098</td>
      <td>-1.173</td>
      <td>-0.862</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3655.0</td>
      <td>2527.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>307779</th>
      <td>m[ACH-000296__11]</td>
      <td>0.187</td>
      <td>0.132</td>
      <td>-0.024</td>
      <td>0.395</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>6623.0</td>
      <td>2838.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>308486</th>
      <td>m[ACH-002023__5]</td>
      <td>0.188</td>
      <td>0.050</td>
      <td>0.109</td>
      <td>0.266</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5230.0</td>
      <td>3358.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>307775</th>
      <td>m[ACH-000296__7]</td>
      <td>0.200</td>
      <td>0.073</td>
      <td>0.086</td>
      <td>0.318</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>7100.0</td>
      <td>3062.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>307783</th>
      <td>m[ACH-000296__15]</td>
      <td>0.228</td>
      <td>0.126</td>
      <td>0.030</td>
      <td>0.430</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>2264.0</td>
      <td>2287.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
    <tr>
      <th>308482</th>
      <td>m[ACH-002023__1]</td>
      <td>0.361</td>
      <td>0.041</td>
      <td>0.297</td>
      <td>0.430</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>6699.0</td>
      <td>3580.0</td>
      <td>1.0</td>
      <td>m</td>
    </tr>
  </tbody>
</table>
</div>



```python
varnames = ["mu_a", "b", "d", "f"]
example_genes = ["AXIN1", "CTNNB1", "TP53", "PTEN"]
gene_effects_post = trace.posterior.get(varnames).sel(gene=example_genes).to_dataframe()

plot_df = (
    gene_effects_post.reset_index(drop=False)
    .pivot_longer(
        index=["chain", "draw", "gene"], names_to="var_name", values_to="value"
    )
    .astype({"chain": "category"})
)

fig, axes = plt.subplots(len(varnames), 2, figsize=(9, len(varnames) * 2))
gene_pal = {
    g: c for g, c in zip(example_genes, sns.mpl_palette("Set1", len(example_genes)))
}

for i, (var_name, data_v) in enumerate(plot_df.groupby("var_name")):
    for c, data_c in data_v.groupby("chain"):
        sns.kdeplot(
            data=data_c,
            x="value",
            hue="gene",
            palette=gene_pal,
            fill=False,
            ax=axes[i, 0],
            alpha=0.8,
        )
        axes[i, 0].set_xlabel(None)
        axes[i, 0].set_ylabel(var_name)
        sns.lineplot(
            data=data_c,
            x="draw",
            y="value",
            hue="gene",
            palette=gene_pal,
            ax=axes[i, 1],
            alpha=0.6,
            lw=0.25,
        )
        axes[i, 1].set_xlim(0, data_c["draw"].max())
        axes[i, 1].set_xlabel(None)

axes[-1, 1].set_xlabel("draw")
for ax in axes.flatten():
    if (leg := ax.get_legend()) is not None:
        leg.remove()

leg_handles = []
for g, c in gene_pal.items():
    leg_handles.append(Line2D([0], [0], color=c, label=g))
fig.legend(handles=leg_handles, title="gene", loc="upper left", bbox_to_anchor=(1, 1))

fig.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_37_0.png)




```python
az.plot_trace(
    trace,
    var_names=["h"],
    coords={"gene": example_genes},
    compact=True,
    legend=False,
)
plt.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_38_0.png)




```python
sgrnas_sample = trace.posterior.coords["sgrna"].values[:5]

az.plot_trace(trace, var_names="a", coords={"sgrna": sgrnas_sample}, compact=False)
plt.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_39_0.png)




```python
example_genes = ["KRAS", "BRAF", "CTNNB1", "TP53", "PTEN"]
for example_gene in example_genes:
    display(Markdown(f"🧬 target gene: *{example_gene}*"))
    example_gene_sgrna = sgrna_to_gene_map.query(f"hugo_symbol == '{example_gene}'")[
        "sgrna"
    ].tolist()
    axes = az.plot_forest(
        trace,
        var_names=[
            "mu_mu_a",
            "mu_a",
            "a",
            "mu_b",
            "b",
            "d",
            "f",
            "h",
        ],
        coords={"gene": [example_gene], "sgrna": example_gene_sgrna},
        combined=True,
        figsize=(6, 7),
    )
    axes[0].axvline(0, color="k", lw=0.8, zorder=1)
    plt.show()
```


🧬 target gene: *KRAS*




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_40_1.png)




🧬 target gene: *BRAF*




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_40_3.png)




🧬 target gene: *CTNNB1*




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_40_5.png)




🧬 target gene: *TP53*




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_40_7.png)




🧬 target gene: *PTEN*




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_40_9.png)




```python
with az.rc_context({"plot.max_subplots": 64}):
    for gene in ["KIF11", "SNRNP200"]:
        axes = az.plot_pair(
            trace,
            var_names=["mu_a", "b", "d", "f", "h"],
            coords={"gene": [gene]},
            figsize=(13, 13),
            scatter_kwargs={"alpha": 0.1, "markersize": 2},
        )
        for ax in axes.flatten():
            ax.axhline(0, color="k")
            ax.axvline(0, color="k")
        plt.tight_layout()
        plt.show()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/plots/backends/matplotlib/pairplot.py:232: UserWarning: rcParams['plot.max_subplots'] (64) is smaller than the number of resulting pair plots with these variables, generating only a 10x10 grid
      warnings.warn(




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_41_1.png)



    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/arviz/plots/backends/matplotlib/pairplot.py:232: UserWarning: rcParams['plot.max_subplots'] (64) is smaller than the number of resulting pair plots with these variables, generating only a 10x10 grid
      warnings.warn(




![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_41_3.png)




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



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_42_0.png)




```python
tp53_muts = (
    valid_crc_data.query("hugo_symbol == 'TP53'")[["depmap_id", "is_mutated"]]
    .copy()
    .drop_duplicates()
    .rename(columns={"is_mutated": "TP53 mut."})
    .reset_index(drop=True)
)
```


```python
cell_line_vars = ["mu_k", "mu_m"]
cell_line_effects = (
    az.summary(trace, var_names=cell_line_vars, kind="stats")
    .pipe(extract_coords_param_names, names=["depmap_id"])
    .assign(var_name=lambda d: [x.split("[")[0] for x in d.index.values])
    .pivot_wider(
        index="depmap_id",
        names_from="var_name",
        values_from=["mean", "hdi_5.5%", "hdi_94.5%"],
    )
    .merge(tp53_muts, on="depmap_id")
)

fig, ax = plt.subplots(figsize=(6, 6))

ax.axhline(0, color="k", zorder=1)
ax.axvline(0, color="k", zorder=1)

ax.vlines(
    x=cell_line_effects["mean_mu_k"],
    ymin=cell_line_effects["hdi_5.5%_mu_m"],
    ymax=cell_line_effects["hdi_94.5%_mu_m"],
    alpha=0.5,
    color="gray",
)
ax.hlines(
    y=cell_line_effects["mean_mu_m"],
    xmin=cell_line_effects["hdi_5.5%_mu_k"],
    xmax=cell_line_effects["hdi_94.5%_mu_k"],
    alpha=0.5,
    color="gray",
)
mut_pal = {True: "tab:red", False: "tab:blue"}
sns.scatterplot(
    data=cell_line_effects,
    x="mean_mu_k",
    y="mean_mu_m",
    hue="TP53 mut.",
    palette=mut_pal,
    ax=ax,
    zorder=10,
)

annotate_df = cell_line_effects.query("mean_mu_k < -0.05 or mean_mu_m < -0.4")
for _, info in annotate_df.iterrows():
    ax.text(
        x=info["mean_mu_k"] + 0.002, y=info["mean_mu_m"] + 0.01, s=info["depmap_id"]
    )

sns.move_legend(ax, loc="upper left")
fig.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_44_0.png)




```python
cell_chromosome_map = (
    valid_crc_data[["depmap_id", "sgrna_target_chr", "cell_chrom"]]
    .drop_duplicates()
    .sort_values("cell_chrom")
    .reset_index(drop=True)
)
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



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_45_0.png)




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
ax = sns.heatmap(
    plot_df,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
ax.set_ylabel(None)
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_47_0.png)




```python
cells_var_names = ["mu_k", "mu_m"]
cells_corr_post = (
    az.summary(trace, "cells_chol_cov_corr", kind="stats")
    .pipe(extract_coords_param_names, names=["d1", "d2"])
    .astype({"d1": int, "d2": int})
    .assign(
        p1=lambda d: [cells_var_names[i] for i in d["d1"]],
        p2=lambda d: [cells_var_names[i] for i in d["d2"]],
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
_, ax = plt.subplots(figsize=(2, 2))
plot_df = cells_corr_post.pivot_wider("p1", "p2", "mean").set_index("p1")
sns.heatmap(plot_df, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_ylabel(None)
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_49_0.png)




```python
cancer_genes = trace.posterior.coords["cancer_gene"].values.tolist()
cancer_gene_mutants = (
    valid_crc_data.filter_column_isin("hugo_symbol", cancer_genes)[
        ["hugo_symbol", "depmap_id", "is_mutated"]
    ]
    .drop_duplicates()
    # .assign(is_mutated=lambda d: d["is_mutated"].map({True: "X", False: ""}))
    .sort_values(["hugo_symbol", "depmap_id"])
    .pivot_wider("depmap_id", names_from="hugo_symbol", values_from="is_mutated")
    .set_index("depmap_id")
)

sns.clustermap(
    cancer_gene_mutants, cmap="gray_r", xticklabels=1, yticklabels=1, figsize=(3, 9)
);
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_50_0.png)




```python
sns.clustermap(
    cancer_gene_mutants.corr(),
    cmap="seismic",
    center=0,
    vmin=-1,
    vmax=1,
    figsize=(4, 4),
);
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_51_0.png)




```python
az.plot_trace(trace, var_names=["sigma_h"], compact=True)
plt.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_52_0.png)




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

_, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=h_post_summary, x="mean", hue="cancer_gene", ax=ax)
ax.set_xlabel(r"$\bar{h}_g$ posterior")
ax.set_ylabel("density")
ax.get_legend().set_title("cancer gene\ncomut.")
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_53_0.png)




```python
fig, axes = plt.subplots(
    len(cancer_genes), 1, squeeze=False, figsize=(10, len(cancer_genes) * 4)
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
    ax.axhline(0, color="k", lw=0.8)
    cg_mut_pal = {"mut.": "tab:red", "WT": "gray"}
    boxes = sns.boxplot(
        data=h_hits_data,
        x="hugo_symbol",
        y="lfc",
        hue="_cg_mut",
        palette=cg_mut_pal,
        ax=ax,
        showfliers=False,
        boxprops={"alpha": 0.5},
    )
    points = sns.stripplot(
        data=h_hits_data,
        x="hugo_symbol",
        y="lfc",
        hue="_cg_mut",
        dodge=True,
        palette=cg_mut_pal,
        s=4,
        alpha=0.7,
        ax=ax,
    )
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), title=cg)
    ax.set_xlabel(None)
    ax.set_ylabel("log-fold change")

axes[-1, 0].set_xlabel("target gene")
plt.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_54_0.png)




```python
h_post_df = (
    crc_post_summary.query("var_name== 'h'")
    .reset_index(drop=True)
    .pipe(
        extract_coords_param_names,
        names=["target_gene", "cancer_gene"],
        col="parameter",
    )
    .pivot_wider("target_gene", names_from="cancer_gene", values_from="mean")
    .set_index("target_gene")
)

h_gene_var = h_post_df.values.var(axis=1)
idx = h_gene_var >= np.quantile(h_gene_var, q=0.8)
h_post_df_topvar = h_post_df.loc[idx, :]

sns.clustermap(h_post_df_topvar, cmap="seismic", center=0, figsize=(5, 12))
plt.show()
```

    /home/jc604/.conda/envs/speclet/lib/python3.10/site-packages/seaborn/matrix.py:654: UserWarning: Clustering large matrix with scipy. Installing `fastcluster` may give better performance.
      warnings.warn(msg)





    <seaborn.matrix.ClusterGrid at 0x7ff8a65da5c0>





![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_55_2.png)




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
        obs_data = valid_crc_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(
            data=obs_data, x="m_rna_gene", y="lfc", ax=ax, edgecolor=None, s=20
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("log RNA expression")
fig.supylabel("log-fold change")

fig.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_56_0.png)




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
        obs_data = valid_crc_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(data=obs_data, x="cn_gene", y="lfc", ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("copy number")
fig.supylabel("log-fold change")
fig.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_57_0.png)



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




    (40, 2842455)




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
sns.kdeplot(x=np.log10(valid_crc_data["counts_final"] + 1), color="k", ax=ax1)
ax1.set_xlabel("log10(counts final + 1)")
ax1.set_ylabel("density")

x_max = 1000
x_cut = x_max * 5
for i in range(example_ppc_draws.shape[0]):
    x = example_ppc_draws[i, :]
    x = x[x < x_cut]
    sns.kdeplot(x=x, alpha=0.2, color="tab:blue", ax=ax2)

sns.kdeplot(x=pp_avg[pp_avg < x_cut], color="tab:orange", ax=ax2)
_obs = valid_crc_data["counts_final"].values
_obs = _obs[_obs < x_cut]
sns.kdeplot(x=_obs, color="k", ax=ax2)
ax2.set_xlabel("counts final")
ax2.set_ylabel("density")
ax2.set_xlim(0, x_max)

fig.suptitle("PPC")
fig.tight_layout()
plt.show()
```



![png](035_single-lineage-colorectal-inspection_009_files/035_single-lineage-colorectal-inspection_009_60_0.png)



---

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```


```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```


```python

```
