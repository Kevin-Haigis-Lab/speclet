# Inspect the single-lineage model run on the prostate data


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

### Load posterior summary


```python
prostate_post_summary = pd.read_csv(
    models_dir() / "hnb-single-lineage-prostate_PYMC_NUMPYRO" / "posterior-summary.csv"
).assign(var_name=lambda d: [x.split("[")[0] for x in d["parameter"]])
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
      <td>mu_b</td>
      <td>-0.000</td>
      <td>0.008</td>
      <td>-0.012</td>
      <td>0.012</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5836.0</td>
      <td>3091.0</td>
      <td>1.0</td>
      <td>mu_b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b[A1BG]</td>
      <td>-0.037</td>
      <td>0.067</td>
      <td>-0.148</td>
      <td>0.064</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5267.0</td>
      <td>2895.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b[A1CF]</td>
      <td>-0.008</td>
      <td>0.067</td>
      <td>-0.109</td>
      <td>0.101</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4757.0</td>
      <td>2877.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b[A2M]</td>
      <td>0.038</td>
      <td>0.070</td>
      <td>-0.076</td>
      <td>0.150</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5439.0</td>
      <td>2917.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b[A2ML1]</td>
      <td>0.032</td>
      <td>0.068</td>
      <td>-0.078</td>
      <td>0.134</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6742.0</td>
      <td>3014.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



### Load trace object


```python
trace_file = (
    models_dir() / "hnb-single-lineage-prostate_PYMC_NUMPYRO" / "posterior.netcdf"
)
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


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/23/22 03:19:08] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Processing data for modeling.     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#267" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">267</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> LFC limits: <span style="font-weight: bold">(</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-5.0</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5.0</span><span style="font-weight: bold">)</span>           <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#268" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">268</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[07/23/22 03:20:09] </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> number of data points dropped: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>  <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">lineage_hierarchical_nb.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///n/data1/hms/dbmi/park/Cook/speclet/speclet/bayesian_models/lineage_hierarchical_nb.py#319" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">319</span></a>
</pre>



## Analysis


```python
sns.histplot(x=prostate_post_summary["r_hat"], binwidth=0.01, stat="proportion");
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_17_0.png)




```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=prostate_post_summary, x="var_name", y="r_hat", ax=ax)
ax.tick_params(rotation=90)
plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_18_0.png)




```python
az.plot_energy(trace);
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_19_0.png)




```python
sgrna_to_gene_map = (
    prostate_data.copy()[["hugo_symbol", "sgrna"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
```


```python
(
    prostate_post_summary.query("var_name == 'mu_a'")
    .sort_values("mean")
    .reset_index(drop=True)
    .pipe(head_tail, n=5)
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
      <td>mu_a[RAN]</td>
      <td>-1.404</td>
      <td>0.447</td>
      <td>-2.118</td>
      <td>-0.693</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>5525.0</td>
      <td>2968.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mu_a[KIF11]</td>
      <td>-1.383</td>
      <td>0.439</td>
      <td>-2.092</td>
      <td>-0.695</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>5079.0</td>
      <td>3191.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mu_a[HSPE1]</td>
      <td>-1.357</td>
      <td>0.458</td>
      <td>-2.056</td>
      <td>-0.589</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>5068.0</td>
      <td>2798.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mu_a[CENPW]</td>
      <td>-1.344</td>
      <td>0.452</td>
      <td>-2.061</td>
      <td>-0.624</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>5257.0</td>
      <td>3046.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mu_a[RPL9]</td>
      <td>-1.273</td>
      <td>0.444</td>
      <td>-1.992</td>
      <td>-0.583</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>5068.0</td>
      <td>2750.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>mu_a[FOPNL]</td>
      <td>0.455</td>
      <td>0.451</td>
      <td>-0.269</td>
      <td>1.154</td>
      <td>0.007</td>
      <td>0.006</td>
      <td>4467.0</td>
      <td>2811.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>mu_a[ZNF611]</td>
      <td>0.458</td>
      <td>0.458</td>
      <td>-0.292</td>
      <td>1.174</td>
      <td>0.007</td>
      <td>0.006</td>
      <td>4558.0</td>
      <td>2941.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>mu_a[TMPRSS11F]</td>
      <td>0.471</td>
      <td>0.492</td>
      <td>-0.270</td>
      <td>1.270</td>
      <td>0.007</td>
      <td>0.006</td>
      <td>5616.0</td>
      <td>3391.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>mu_a[EPHA2]</td>
      <td>0.503</td>
      <td>0.457</td>
      <td>-0.226</td>
      <td>1.217</td>
      <td>0.006</td>
      <td>0.006</td>
      <td>4994.0</td>
      <td>2721.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>mu_a[NF2]</td>
      <td>0.536</td>
      <td>0.442</td>
      <td>-0.207</td>
      <td>1.213</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>5824.0</td>
      <td>2857.0</td>
      <td>1.0</td>
      <td>mu_a</td>
    </tr>
  </tbody>
</table>
</div>




```python
(
    prostate_post_summary.query("var_name == 'b'")
    .sort_values("mean")
    .reset_index(drop=True)
    .pipe(head_tail, n=5)
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
      <td>-0.944</td>
      <td>0.066</td>
      <td>-1.043</td>
      <td>-0.833</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4953.0</td>
      <td>3081.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b[ZC3H11A]</td>
      <td>-0.692</td>
      <td>0.157</td>
      <td>-0.930</td>
      <td>-0.441</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4386.0</td>
      <td>2714.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b[HOXB13]</td>
      <td>-0.681</td>
      <td>0.083</td>
      <td>-0.817</td>
      <td>-0.555</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5387.0</td>
      <td>3107.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b[EBP]</td>
      <td>-0.661</td>
      <td>0.080</td>
      <td>-0.792</td>
      <td>-0.537</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5285.0</td>
      <td>3038.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b[IRS2]</td>
      <td>-0.604</td>
      <td>0.076</td>
      <td>-0.722</td>
      <td>-0.480</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5076.0</td>
      <td>2756.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18114</th>
      <td>b[NDUFB11]</td>
      <td>0.604</td>
      <td>0.079</td>
      <td>0.479</td>
      <td>0.733</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4840.0</td>
      <td>3061.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18115</th>
      <td>b[NDUFA10]</td>
      <td>0.606</td>
      <td>0.072</td>
      <td>0.493</td>
      <td>0.723</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4985.0</td>
      <td>3061.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18116</th>
      <td>b[GRB2]</td>
      <td>0.613</td>
      <td>0.072</td>
      <td>0.501</td>
      <td>0.730</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4620.0</td>
      <td>2955.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18117</th>
      <td>b[NARS2]</td>
      <td>0.644</td>
      <td>0.071</td>
      <td>0.527</td>
      <td>0.754</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5381.0</td>
      <td>3110.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>18118</th>
      <td>b[AIFM1]</td>
      <td>0.697</td>
      <td>0.071</td>
      <td>0.587</td>
      <td>0.811</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4395.0</td>
      <td>3063.0</td>
      <td>1.0</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>




```python
example_genes = ["KIF11", "AR", "NF2"]
az.plot_trace(
    trace, var_names=["mu_a", "b"], coords={"gene": example_genes}, compact=False
)
plt.tight_layout()
plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_23_0.png)




```python
sgrnas_sample = trace.posterior.coords["sgrna"].values[:5]

az.plot_trace(trace, var_names="a", coords={"sgrna": sgrnas_sample}, compact=False)
plt.tight_layout()
plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_24_0.png)




```python
example_gene = "KIF11"
sgrna_AR = sgrna_to_gene_map.query(f"hugo_symbol == '{example_gene}'")["sgrna"].tolist()
az.plot_forest(
    trace,
    var_names=["mu_mu_a", "sigma_mu_a", "mu_a", "sigma_a", "a", "mu_b", "b"],
    coords={"gene": [example_gene], "sgrna": sgrna_AR},
    combined=False,
    figsize=(6, 5),
)
plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_25_0.png)




```python
az.plot_trace(trace, var_names=["mu_mu_a", "mu_b"], compact=False)
plt.tight_layout()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_26_0.png)




```python
az.plot_trace(trace, var_names=["^sigma_*"], filter_vars="regex", compact=False)
plt.tight_layout()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_27_0.png)




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
      <th>107302</th>
      <td>sigma_b</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>0.0</td>
      <td>0.012</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3095.0</td>
      <td>1663.0</td>
      <td>1.0</td>
      <td>sigma_b</td>
    </tr>
    <tr>
      <th>107303</th>
      <td>sigma_mu_a</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>0.0</td>
      <td>0.014</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2321.0</td>
      <td>1220.0</td>
      <td>1.0</td>
      <td>sigma_mu_a</td>
    </tr>
    <tr>
      <th>107304</th>
      <td>sigma_a</td>
      <td>0.003</td>
      <td>0.003</td>
      <td>0.0</td>
      <td>0.007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2553.0</td>
      <td>1581.0</td>
      <td>1.0</td>
      <td>sigma_a</td>
    </tr>
  </tbody>
</table>
</div>




```python
var_names = ["a", "mu_a", "b"]
_, axes = plt.subplots(2, 2, figsize=(8, 4), sharex=True)
for ax, var_name in zip(axes.flatten(), var_names):
    x = prostate_post_summary.query(f"var_name == '{var_name}'")["mean"]
    sns.kdeplot(x=x, ax=ax)
    ax.set_title(f"${var_name}$")
    ax.set_xlim(-2, 1)

plt.tight_layout()
plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_29_0.png)




```python
eg_gene = trace.posterior.coords["gene"].values[0]

for gene in [eg_gene, "KIF11"]:
    az.plot_pair(
        trace,
        var_names=["mu_a", "b"],
        coords={"gene": [gene]},
    )
    plt.tight_layout()
    plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_30_0.png)





![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_30_1.png)




```python
mu_a_post_avg = trace.posterior["mu_a"].mean(axis=(0, 1))
b_post_avg = trace.posterior["b"].mean(axis=(0, 1))

ax = sns.scatterplot(x=mu_a_post_avg, y=b_post_avg, alpha=0.1, linewidth=0)
ax.axhline(color="black")
ax.axvline(color="black")
ax.set_xlabel(r"$\mu_a$")
ax.set_ylabel(r"$b$")
ax.set_title("Joint posterior distribution")
plt.show()
```



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_31_0.png)




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



![png](022_single-lineage-prostate-inspection_files/022_single-lineage-prostate-inspection_33_0.png)




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

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 5.48 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-07-23

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
    qnorm     : 0.8.1
    pandas    : 1.4.3
    matplotlib: 3.5.2
    numpy     : 1.22.4
    seaborn   : 0.11.2
