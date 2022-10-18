# Model diagnostics

## Setup

### Imports


```python
%load_ext autoreload
%autoreload 2
```


```python
from time import time

import pandas as pd
```


```python
from speclet.analysis.sublineage_model_analysis import load_sublineage_model_posteriors
from speclet.managers.posterior_data_manager import PosteriorDataManager as PostDataMan
```


```python
# Notebook execution timer.
notebook_tic = time()
```

### Data

#### Model posteriors


```python
postmen = load_sublineage_model_posteriors()
```


```python
len(postmen)
```




    43



## Analysis


```python
def posterior_dims_to_dataframe(pm: PostDataMan) -> pd.DataFrame:
    return pd.DataFrame(dict(pm.trace.posterior.dims), index=[0]).assign(
        lineage_subtype=pm.id
    )
```


```python
posterior_dims = pd.concat(
    [posterior_dims_to_dataframe(pm) for pm in postmen.posteriors]
).reset_index(drop=True)
```


```python
posterior_dims.describe().round(1)
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
      <th>chain</th>
      <th>draw</th>
      <th>delta_genes_dim_0</th>
      <th>delta_genes_dim_1</th>
      <th>sgrna</th>
      <th>delta_cells_dim_0</th>
      <th>delta_cells_dim_1</th>
      <th>cell_chrom</th>
      <th>genes_chol_cov_dim_0</th>
      <th>cells_chol_cov_dim_0</th>
      <th>genes_chol_cov_corr_dim_0</th>
      <th>genes_chol_cov_corr_dim_1</th>
      <th>genes_chol_cov_stds_dim_0</th>
      <th>gene</th>
      <th>cancer_gene</th>
      <th>cells_chol_cov_corr_dim_0</th>
      <th>cells_chol_cov_corr_dim_1</th>
      <th>cells_chol_cov_stds_dim_0</th>
      <th>cell_line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>19.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
      <td>43.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.0</td>
      <td>1000.0</td>
      <td>6.3</td>
      <td>18119.0</td>
      <td>71062.0</td>
      <td>2.0</td>
      <td>19.4</td>
      <td>446.1</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>6.3</td>
      <td>6.3</td>
      <td>6.3</td>
      <td>18119.0</td>
      <td>5.1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.5</td>
      <td>403.5</td>
      <td>34.2</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>3.6</td>
      <td>3.6</td>
      <td>0.0</td>
      <td>3.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.0</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>18119.0</td>
      <td>71062.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>92.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>18119.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.0</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>18119.0</td>
      <td>71062.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>115.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>18119.0</td>
      <td>1.5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.0</td>
      <td>1000.0</td>
      <td>4.0</td>
      <td>18119.0</td>
      <td>71062.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>345.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>18119.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.0</td>
      <td>1000.0</td>
      <td>8.0</td>
      <td>18119.0</td>
      <td>71062.0</td>
      <td>2.0</td>
      <td>26.5</td>
      <td>609.5</td>
      <td>36.5</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>18119.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>26.5</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.0</td>
      <td>1000.0</td>
      <td>16.0</td>
      <td>18119.0</td>
      <td>71062.0</td>
      <td>2.0</td>
      <td>83.0</td>
      <td>1909.0</td>
      <td>136.0</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>18119.0</td>
      <td>12.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>83.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
n_cell_lines = 20
n_chromosomes = 23
n_cancer_genes = 5
n_sgrnas = 72000
n_genes = 18000

n_cell_vars = (n_cell_lines * 2) + (n_cell_lines * n_chromosomes * 2)
n_gene_vars = (n_genes * 4) + n_sgrnas
n_comut_vars = n_genes * n_cancer_genes
total = n_cell_vars + n_gene_vars + n_comut_vars

n_cell_vars, n_gene_vars, n_comut_vars, total
```




    (960, 144000, 90000, 234960)



---


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 0.39 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-10-18

    Python implementation: CPython
    Python version       : 3.10.6
    IPython version      : 8.5.0

    Compiler    : GCC 10.4.0
    OS          : Linux
    Release     : 3.10.0-1160.76.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-231.o2.rc.hms.harvard.edu

    Git branch: figures

    pandas: 1.4.4




```python

```
