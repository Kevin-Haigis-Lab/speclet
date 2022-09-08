# Exploring lineages and sublineages

## Setup

### Imports


```python
%load_ext autoreload
%autoreload 2
```


```python
from pathlib import Path
from time import time

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
```


```python
from speclet.io import DataFile, data_path, notebook_table_dir
from speclet.plot import set_speclet_theme
```


```python
# Notebook execution timer.
notebook_tic = time()

# Plotting setup.
set_speclet_theme()
%config InlineBackend.figure_format = "retina"

# Constants
RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)
```


```python
output_dir = notebook_table_dir("001_020_lineage-exploration")
```

### Data


```python
modeling_data_file = data_path(DataFile.DEPMAP_DATA)
modeling_data_file
```




    PosixPath('/n/data1/hms/dbmi/park/Cook/speclet/modeling_data/depmap-modeling-data.csv')




```python
modeling_data_dd = dd.read_csv(
    modeling_data_file, low_memory=False, dtype={"age": "float64"}
)
modeling_data_dd.head()
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
      <td>AAACCTGCGGCGGTCGCCA</td>
      <td>OVR3_c905R1</td>
      <td>-0.299958</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr8_66505451_-</td>
      <td>VXN</td>
      <td>sanger</td>
      <td>True</td>
      <td>8</td>
      <td>66505451</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.139595</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AACAGCACACCGGCCCCGT</td>
      <td>OVR3_c905R1</td>
      <td>0.267092</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_156009834_-</td>
      <td>IL9R</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>156009834</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACCTCCGGACTCCTCAGC</td>
      <td>OVR3_c905R1</td>
      <td>0.550477</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr7_39609658_-</td>
      <td>YAE1</td>
      <td>sanger</td>
      <td>True</td>
      <td>7</td>
      <td>39609658</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.923715</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACTCAAACTGACGCCGAA</td>
      <td>OVR3_c905R1</td>
      <td>-0.391922</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr1_117623388_-</td>
      <td>TENT5C</td>
      <td>sanger</td>
      <td>True</td>
      <td>1</td>
      <td>117623388</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AACTGACCTTGAAACGCTG</td>
      <td>OVR3_c905R1</td>
      <td>-1.562577</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr16_66933623_+</td>
      <td>CIAO2B</td>
      <td>sanger</td>
      <td>True</td>
      <td>16</td>
      <td>66933623</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## Analysis


```python
lineage_data = (
    modeling_data_dd.query("screen == 'broad'")[
        ["depmap_id", "lineage", "lineage_subtype"]
    ]
    .drop_duplicates()
    .compute()
)
lineage_data
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
      <th>lineage</th>
      <th>lineage_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86746</th>
      <td>ACH-000004</td>
      <td>blood</td>
      <td>AML</td>
    </tr>
    <tr>
      <th>157808</th>
      <td>ACH-000005</td>
      <td>blood</td>
      <td>AML</td>
    </tr>
    <tr>
      <th>228871</th>
      <td>ACH-000007</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
    </tr>
    <tr>
      <th>117813</th>
      <td>ACH-000009</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
    </tr>
    <tr>
      <th>25860</th>
      <td>ACH-000011</td>
      <td>urinary_tract</td>
      <td>bladder_carcinoma</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36493</th>
      <td>ACH-002460</td>
      <td>skin</td>
      <td>melanoma</td>
    </tr>
    <tr>
      <th>107555</th>
      <td>ACH-002508</td>
      <td>skin</td>
      <td>melanoma</td>
    </tr>
    <tr>
      <th>178617</th>
      <td>ACH-002510</td>
      <td>skin</td>
      <td>melanoma</td>
    </tr>
    <tr>
      <th>249679</th>
      <td>ACH-002512</td>
      <td>skin</td>
      <td>melanoma</td>
    </tr>
    <tr>
      <th>222928</th>
      <td>ACH-002875</td>
      <td>skin</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>895 rows × 3 columns</p>
</div>




```python
(
    lineage_data.copy()
    .assign(missing_subtype=lambda d: d["lineage_subtype"].isna())
    .groupby("lineage")["missing_subtype"]
    .sum()
    .reset_index()
    .set_index("lineage")
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
      <th>missing_subtype</th>
    </tr>
    <tr>
      <th>lineage</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bile_duct</th>
      <td>0</td>
    </tr>
    <tr>
      <th>blood</th>
      <td>0</td>
    </tr>
    <tr>
      <th>bone</th>
      <td>0</td>
    </tr>
    <tr>
      <th>breast</th>
      <td>1</td>
    </tr>
    <tr>
      <th>central_nervous_system</th>
      <td>0</td>
    </tr>
    <tr>
      <th>cervix</th>
      <td>0</td>
    </tr>
    <tr>
      <th>colorectal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>epidermoid_carcinoma</th>
      <td>0</td>
    </tr>
    <tr>
      <th>esophagus</th>
      <td>0</td>
    </tr>
    <tr>
      <th>eye</th>
      <td>0</td>
    </tr>
    <tr>
      <th>fibroblast</th>
      <td>0</td>
    </tr>
    <tr>
      <th>gastric</th>
      <td>0</td>
    </tr>
    <tr>
      <th>kidney</th>
      <td>2</td>
    </tr>
    <tr>
      <th>liver</th>
      <td>1</td>
    </tr>
    <tr>
      <th>lung</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lymphocyte</th>
      <td>0</td>
    </tr>
    <tr>
      <th>ovary</th>
      <td>0</td>
    </tr>
    <tr>
      <th>pancreas</th>
      <td>0</td>
    </tr>
    <tr>
      <th>peripheral_nervous_system</th>
      <td>0</td>
    </tr>
    <tr>
      <th>plasma_cell</th>
      <td>0</td>
    </tr>
    <tr>
      <th>prostate</th>
      <td>0</td>
    </tr>
    <tr>
      <th>skin</th>
      <td>1</td>
    </tr>
    <tr>
      <th>soft_tissue</th>
      <td>0</td>
    </tr>
    <tr>
      <th>thyroid</th>
      <td>0</td>
    </tr>
    <tr>
      <th>upper_aerodigestive</th>
      <td>0</td>
    </tr>
    <tr>
      <th>urinary_tract</th>
      <td>0</td>
    </tr>
    <tr>
      <th>uterus</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
lineage_counts = (
    lineage_data.copy()
    .reset_index(drop=True)
    .fillna({"lineage_subtype": "NA"})
    .groupby(["lineage", "lineage_subtype"])["depmap_id"]
    .count()
    .reset_index()
    .sort_values(["lineage", "depmap_id"], ascending=(True, False))
)

for lineage, data in lineage_counts.groupby("lineage"):
    total = data["depmap_id"].sum()
    display(Markdown(f"**{lineage}** ({total} cell lines)"))
    display(data.reset_index(drop=True))
    display(Markdown("---"))
```


**bile_duct** (37 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bile_duct</td>
      <td>cholangiocarcinoma</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bile_duct</td>
      <td>gallbladder_adenocarcinoma</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



---



**blood** (55 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>blood</td>
      <td>AML</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>blood</td>
      <td>ALL</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>blood</td>
      <td>CML</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>blood</td>
      <td>CLL</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>blood</td>
      <td>unspecified_leukemia</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



---



**bone** (30 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bone</td>
      <td>Ewing_sarcoma</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bone</td>
      <td>osteosarcoma</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bone</td>
      <td>chordoma</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bone</td>
      <td>chondrosarcoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**breast** (40 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>breast</td>
      <td>breast_carcinoma</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>breast</td>
      <td>breast_ductal_carcinoma</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>breast</td>
      <td>NA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>breast</td>
      <td>breast_adenocarcinoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**central_nervous_system** (62 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>central_nervous_system</td>
      <td>glioma</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>central_nervous_system</td>
      <td>medulloblastoma</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>central_nervous_system</td>
      <td>meningioma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>central_nervous_system</td>
      <td>PNET</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**cervix** (14 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cervix</td>
      <td>cervical_carcinoma</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cervix</td>
      <td>cervical_squamous</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cervix</td>
      <td>cervical_adenocarcinoma</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cervix</td>
      <td>glassy_cell_carcinoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**colorectal** (40 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



---



**epidermoid_carcinoma** (1 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>epidermoid_carcinoma</td>
      <td>skin_squamous</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**esophagus** (25 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>esophagus</td>
      <td>esophagus_squamous</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>esophagus</td>
      <td>esophagus_adenocarcinoma</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



---



**eye** (7 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>eye</td>
      <td>uveal_melanoma</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>eye</td>
      <td>retinoblastoma</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



---



**fibroblast** (1 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fibroblast</td>
      <td>fibroblast_soft_tissue</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**gastric** (28 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gastric</td>
      <td>gastric_adenocarcinoma</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



---



**kidney** (28 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kidney</td>
      <td>renal_cell_carcinoma</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kidney</td>
      <td>NA</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kidney</td>
      <td>malignant_rhabdoid_tumor</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



---



**liver** (22 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>liver</td>
      <td>hepatocellular_carcinoma</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>liver</td>
      <td>NA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>liver</td>
      <td>hepatoblastoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**lung** (116 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lung</td>
      <td>NSCLC</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lung</td>
      <td>SCLC</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lung</td>
      <td>mesothelioma</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lung</td>
      <td>lung_carcinoid</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**lymphocyte** (30 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lymphocyte</td>
      <td>non_hodgkin_lymphoma</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lymphocyte</td>
      <td>lymphoma_unspecified</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lymphocyte</td>
      <td>hodgkin_lymphoma</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lymphocyte</td>
      <td>ATL</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**ovary** (47 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ovary</td>
      <td>SCCOHT</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ovary</td>
      <td>brenner_tumor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ovary</td>
      <td>mixed_germ_cell</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ovary</td>
      <td>ovary_carcinoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**pancreas** (38 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pancreas</td>
      <td>exocrine</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>



---



**peripheral_nervous_system** (21 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>peripheral_nervous_system</td>
      <td>neuroblastoma</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>peripheral_nervous_system</td>
      <td>PNET</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**plasma_cell** (21 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>plasma_cell</td>
      <td>multiple_myeloma</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



---



**prostate** (5 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>prostate</td>
      <td>prostate_adenocarcinoma</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>prostate</td>
      <td>prostate_hyperplasia</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**skin** (65 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>skin</td>
      <td>melanoma</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skin</td>
      <td>skin_squamous</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>skin</td>
      <td>merkel_cell_carcinoma</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>skin</td>
      <td>NA</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**soft_tissue** (44 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>soft_tissue</td>
      <td>rhabdomyosarcoma</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>soft_tissue</td>
      <td>liposarcoma</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>soft_tissue</td>
      <td>malignant_rhabdoid_tumor</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>soft_tissue</td>
      <td>ATRT</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>soft_tissue</td>
      <td>synovial_sarcoma</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>soft_tissue</td>
      <td>epithelioid_sarcoma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>soft_tissue</td>
      <td>leiomyosarcoma</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>soft_tissue</td>
      <td>fibrosarcoma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>soft_tissue</td>
      <td>pleomorphic_sarcoma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>soft_tissue</td>
      <td>thyroid_sarcoma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>soft_tissue</td>
      <td>undifferentiated_sarcoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



**thyroid** (11 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>thyroid</td>
      <td>thyroid_carcinoma</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>thyroid</td>
      <td>thyroid_squamous</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



---



**upper_aerodigestive** (46 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>upper_aerodigestive</td>
      <td>upper_aerodigestive_squamous</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>



---



**urinary_tract** (30 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>urinary_tract</td>
      <td>bladder_carcinoma</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



---



**uterus** (31 cell lines)



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
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>uterus</td>
      <td>endometrial_adenocarcinoma</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uterus</td>
      <td>choriocarcinoma</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>uterus</td>
      <td>endometrial_squamous</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>uterus</td>
      <td>MMMT</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>uterus</td>
      <td>clear_cell_carcinoma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>uterus</td>
      <td>endometrial_adenosquamous</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>uterus</td>
      <td>endometrial_stromal_sarcoma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>uterus</td>
      <td>mullerian_carcinoma</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>uterus</td>
      <td>uterine_carcinosarcoma</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



---



```python
lineage_counts.to_csv(output_dir / "broad-lineage-subtype-counts.csv", index=False)
```

---

## Session info


```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 7.08 minutes



```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-08-23

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

    Hostname: compute-e-16-231.o2.rc.hms.harvard.edu

    Git branch: expand-lineages

    dask      : 2022.7.1
    matplotlib: 3.5.2
    seaborn   : 0.11.2
    pandas    : 1.4.3
    numpy     : 1.23.1




```python

```
