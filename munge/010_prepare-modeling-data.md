# Prepare data for modeling


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import janitor
from pathlib import Path
import re
```


```python
data_dir = Path('../data')
save_dir = Path('../modeling_data')
```

## Select cell lines


```python
sample_info = pd.read_csv(save_dir / 'sample_info.csv')
```


```python
sample_info.head()
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
      <th>stripped_cell_line_name</th>
      <th>ccle_name</th>
      <th>sex</th>
      <th>cas9_activity</th>
      <th>primary_or_metastasis</th>
      <th>primary_disease</th>
      <th>subtype</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACH-000001</td>
      <td>NIHOVCAR3</td>
      <td>NIHOVCAR3_OVARY</td>
      <td>Female</td>
      <td>NaN</td>
      <td>Metastasis</td>
      <td>Ovarian Cancer</td>
      <td>Adenocarcinoma, high grade serous</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACH-000002</td>
      <td>HL60</td>
      <td>HL60_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</td>
      <td>Female</td>
      <td>NaN</td>
      <td>Primary</td>
      <td>Leukemia</td>
      <td>Acute Myelogenous Leukemia (AML), M3 (Promyelo...</td>
      <td>blood</td>
      <td>AML</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACH-000003</td>
      <td>CACO2</td>
      <td>CACO2_LARGE_INTESTINE</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Colon/Colorectal Cancer</td>
      <td>Adenocarcinoma</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACH-000004</td>
      <td>HEL</td>
      <td>HEL_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</td>
      <td>Male</td>
      <td>47.6</td>
      <td>NaN</td>
      <td>Leukemia</td>
      <td>Acute Myelogenous Leukemia (AML), M6 (Erythrol...</td>
      <td>blood</td>
      <td>AML</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACH-000005</td>
      <td>HEL9217</td>
      <td>HEL9217_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE</td>
      <td>Male</td>
      <td>13.4</td>
      <td>NaN</td>
      <td>Leukemia</td>
      <td>Acute Myelogenous Leukemia (AML), M6 (Erythrol...</td>
      <td>blood</td>
      <td>AML</td>
    </tr>
  </tbody>
</table>
</div>




```python
def show_counts(df, col):
    return df[[col, 'depmap_id']] \
        .drop_duplicates() \
        .groupby(col) \
        .count() \
        .sort_values('depmap_id', ascending=False)
```


```python
show_counts(sample_info, 'primary_disease')
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
    </tr>
    <tr>
      <th>primary_disease</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lung Cancer</th>
      <td>273</td>
    </tr>
    <tr>
      <th>Leukemia</th>
      <td>132</td>
    </tr>
    <tr>
      <th>Skin Cancer</th>
      <td>113</td>
    </tr>
    <tr>
      <th>Lymphoma</th>
      <td>109</td>
    </tr>
    <tr>
      <th>Brain Cancer</th>
      <td>107</td>
    </tr>
    <tr>
      <th>Colon/Colorectal Cancer</th>
      <td>83</td>
    </tr>
    <tr>
      <th>Breast Cancer</th>
      <td>82</td>
    </tr>
    <tr>
      <th>Head and Neck Cancer</th>
      <td>76</td>
    </tr>
    <tr>
      <th>Bone Cancer</th>
      <td>75</td>
    </tr>
    <tr>
      <th>Ovarian Cancer</th>
      <td>74</td>
    </tr>
    <tr>
      <th>Pancreatic Cancer</th>
      <td>59</td>
    </tr>
    <tr>
      <th>Kidney Cancer</th>
      <td>56</td>
    </tr>
    <tr>
      <th>Gastric Cancer</th>
      <td>49</td>
    </tr>
    <tr>
      <th>Neuroblastoma</th>
      <td>46</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Fibroblast</th>
      <td>43</td>
    </tr>
    <tr>
      <th>Sarcoma</th>
      <td>42</td>
    </tr>
    <tr>
      <th>Bladder Cancer</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Endometrial/Uterine Cancer</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Esophageal Cancer</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Bile Duct Cancer</th>
      <td>36</td>
    </tr>
    <tr>
      <th>Myeloma</th>
      <td>34</td>
    </tr>
    <tr>
      <th>Liver Cancer</th>
      <td>27</td>
    </tr>
    <tr>
      <th>Cervical Cancer</th>
      <td>22</td>
    </tr>
    <tr>
      <th>Thyroid Cancer</th>
      <td>21</td>
    </tr>
    <tr>
      <th>Rhabdoid</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Engineered</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Prostate Cancer</th>
      <td>13</td>
    </tr>
    <tr>
      <th>Liposarcoma</th>
      <td>11</td>
    </tr>
    <tr>
      <th>Eye Cancer</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Gallbladder Cancer</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Non-Cancerous</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Embryonal Cancer</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Teratoma</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Adrenal Cancer</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
show_counts(
    sample_info[sample_info.primary_disease == 'Colon/Colorectal Cancer'],
    'subtype'
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
      <th>depmap_id</th>
    </tr>
    <tr>
      <th>subtype</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adenocarcinoma</th>
      <td>73</td>
    </tr>
    <tr>
      <th>Colorectal Carcinoma</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Adenocarcinoma, mucinous</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Adenocarcinoma, papillotubular</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Caecum Adenocarcinoma</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Colon Adenocarcinoma</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Colon Carcinoma</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
show_counts(
    sample_info[sample_info.primary_disease == 'Colon/Colorectal Cancer'],
    'sex'
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
      <th>depmap_id</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>42</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>27</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
show_counts(
    sample_info[sample_info.primary_disease == 'Colon/Colorectal Cancer'], 
    'primary_or_metastasis'
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
      <th>depmap_id</th>
    </tr>
    <tr>
      <th>primary_or_metastasis</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Primary</th>
      <td>43</td>
    </tr>
    <tr>
      <th>Metastasis</th>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
