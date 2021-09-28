# Check of modeling data

```python
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import seaborn as sns
```

```python
import dask.dataframe as dd
from dask.distributed import Client, progress

client = Client(n_workers=4, threads_per_worker=2, memory_limit="16GB")
client
```

<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://127.0.0.1:40063</li>
  <li><b>Dashboard: </b><a href='http://127.0.0.1:8787/status' target='_blank'>http://127.0.0.1:8787/status</a></li>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Cluster</h3>
<ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
  <li><b>Workers: </b>4</li>
  <li><b>Cores: </b>8</li>
  <li><b>Memory: </b>59.60 GiB</li>
</ul>
</td>
</tr>
</table>

```python
depmap_modeling_df_path = Path("../modeling_data/depmap_modeling_dataframe.csv")
if not depmap_modeling_df_path.exists():
    raise FileNotFoundError(f"Could not find '{depmap_modeling_df_path.as_posix()}'")
```

```python
pd.read_csv(depmap_modeling_df_path, low_memory=False, nrows=200)
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
      <th>num_mutations</th>
      <th>any_deleterious</th>
      <th>any_tcga_hotspot</th>
      <th>any_cosmic_hotspot</th>
      <th>is_mutated</th>
      <th>copy_number</th>
      <th>lineage</th>
      <th>primary_or_metastasis</th>
      <th>is_male</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAGCCCAGGAGTATGGGAG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.858827</td>
      <td>3</td>
      <td>chr2_130522105_-</td>
      <td>CFC1B</td>
      <td>broad</td>
      <td>True</td>
      <td>2</td>
      <td>130522105</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.109223</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAATCAGAGAAACCTGAACG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>-0.397664</td>
      <td>3</td>
      <td>chr11_89916950_-</td>
      <td>TRIM49D1</td>
      <td>broad</td>
      <td>True</td>
      <td>11</td>
      <td>89916950</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.155134</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACGTCTTTGAAGAAAGCTG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.102909</td>
      <td>3</td>
      <td>chr5_71055421_-</td>
      <td>GTF2H2</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>71055421</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.757424</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACGTCTTTGAAGGAAGCTG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>-0.434218</td>
      <td>3</td>
      <td>chr5_69572480_+</td>
      <td>GTF2H2C</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>69572480</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.757424</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAGAGGTTCCAGACTACTTA</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.590026</td>
      <td>3</td>
      <td>chrX_155898173_+</td>
      <td>VAMP7</td>
      <td>broad</td>
      <td>True</td>
      <td>X</td>
      <td>155898173</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.345761</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>195</th>
      <td>TGCTGGTGTGAATAAACAGT</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.294152</td>
      <td>3</td>
      <td>chr5_79619732_-</td>
      <td>TENT2</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>79619732</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.757424</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>196</th>
      <td>TGGCCTTAGGAAGCAGTGCG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.110017</td>
      <td>3</td>
      <td>chr18_62187703_+</td>
      <td>RELCH</td>
      <td>broad</td>
      <td>True</td>
      <td>18</td>
      <td>62187703</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.064841</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>197</th>
      <td>TGGCGAAGATGTAGACGGCG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>-0.013850</td>
      <td>3</td>
      <td>chr22_23961084_+</td>
      <td>GSTT2B</td>
      <td>broad</td>
      <td>True</td>
      <td>22</td>
      <td>23961084</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.794994</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>198</th>
      <td>TGGCTGGTGTTCAGGATCCA</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>-0.038889</td>
      <td>3</td>
      <td>chr3_50297333_+</td>
      <td>NAA80</td>
      <td>broad</td>
      <td>True</td>
      <td>3</td>
      <td>50297333</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.154458</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
    <tr>
      <th>199</th>
      <td>TGGTGTCGTAGTGAGCCAGG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.264638</td>
      <td>3</td>
      <td>chr21_43065249_+</td>
      <td>CBS</td>
      <td>broad</td>
      <td>True</td>
      <td>21</td>
      <td>43065249</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2.266272</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 24 columns</p>
</div>

```python
depmap_modeling_df = dd.read_csv(
    depmap_modeling_df_path,
    dtype={
        "age": "float64",
        "p_dna_batch": "object",
        "primary_or_metastasis": "object",
        "counts_final": "float64",
    },
    low_memory=False,
)
```

```python
depmap_modeling_df.head()
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
      <th>num_mutations</th>
      <th>any_deleterious</th>
      <th>any_tcga_hotspot</th>
      <th>any_cosmic_hotspot</th>
      <th>is_mutated</th>
      <th>copy_number</th>
      <th>lineage</th>
      <th>primary_or_metastasis</th>
      <th>is_male</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAGCCCAGGAGTATGGGAG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.858827</td>
      <td>3</td>
      <td>chr2_130522105_-</td>
      <td>CFC1B</td>
      <td>broad</td>
      <td>True</td>
      <td>2</td>
      <td>130522105</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.109223</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAATCAGAGAAACCTGAACG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>-0.397664</td>
      <td>3</td>
      <td>chr11_89916950_-</td>
      <td>TRIM49D1</td>
      <td>broad</td>
      <td>True</td>
      <td>11</td>
      <td>89916950</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.155134</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACGTCTTTGAAGAAAGCTG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.102909</td>
      <td>3</td>
      <td>chr5_71055421_-</td>
      <td>GTF2H2</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>71055421</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.757424</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACGTCTTTGAAGGAAGCTG</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>-0.434218</td>
      <td>3</td>
      <td>chr5_69572480_+</td>
      <td>GTF2H2C</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>69572480</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.757424</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAGAGGTTCCAGACTACTTA</td>
      <td>HEL-311Cas9_RepA_p4_batch3</td>
      <td>0.590026</td>
      <td>3</td>
      <td>chrX_155898173_+</td>
      <td>VAMP7</td>
      <td>broad</td>
      <td>True</td>
      <td>X</td>
      <td>155898173</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.345761</td>
      <td>blood</td>
      <td>NaN</td>
      <td>True</td>
      <td>30.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>

```python
depmap_modeling_df.columns
```

    Index(['sgrna', 'replicate_id', 'lfc', 'p_dna_batch', 'genome_alignment',
           'hugo_symbol', 'screen', 'multiple_hits_on_gene', 'sgrna_target_chr',
           'sgrna_target_pos', 'depmap_id', 'counts_final', 'counts_initial',
           'rna_expr', 'num_mutations', 'any_deleterious', 'any_tcga_hotspot',
           'any_cosmic_hotspot', 'is_mutated', 'copy_number', 'lineage',
           'primary_or_metastasis', 'is_male', 'age'],
          dtype='object')

## Basic checks

```python
FAILED_CHECKS = 0
```

Check that specific columns exist (prevents some really bonehead discoveries later on...).

```python
cols_that_should_exist = [
    "depmap_id",
    "sgrna",
    "hugo_symbol",
    "lfc",
    "screen",
    "num_mutations",
    "is_mutated",
    "lineage",
    "counts_final",
    "p_dna_batch",
    "primary_or_metastasis",
]

missing_cols = [
    col for col in cols_that_should_exist if col not in depmap_modeling_df.columns
]
if len(missing_cols) != 0:
    print(f"Some columns ({len(missing_cols)}) that should be present are not 😦")
    print("  missing columns: " + ", ".join(missing_cols))
    FAILED_CHECKS += 1
```

Check that specific columns have no missing (`NA`) values.

```python
cols_without_na = [
    "depmap_id",
    "sgrna",
    "hugo_symbol",
    "lfc",
    "screen",
    "num_mutations",
    "is_mutated",
    "lineage",
]

na_checks = depmap_modeling_df.isna()[cols_without_na].any().compute()
num_missed_checks = na_checks.sum()

if num_missed_checks > 0:
    FAILED_CHECKS += num_missed_checks
    print(na_checks[na_checks])
```

```python
na_checks
```

    depmap_id        False
    sgrna            False
    hugo_symbol      False
    lfc              False
    screen           False
    num_mutations    False
    is_mutated       False
    lineage          False
    dtype: bool

Check that all combinations of cell line, sgRNA, and experimental replicate only appear once.

```python
grp_cols = ["depmap_id", "sgrna", "replicate_id"]
ct_df = (
    depmap_modeling_df.assign(n=1)[grp_cols + ["n"]]
    .groupby(grp_cols)
    .count()
    .query("n > 1")
    .compute()
)

if not ct_df.shape[0] == 0:
    print("There are some sgRNA with multiple targets.")
    print(ct_df.head(20))
    FAILED_CHECKS += 1
```

    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.98 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.98 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.01 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.01 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.02 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.02 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.03 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.04 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.04 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.05 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.05 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.06 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.07 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.07 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.09 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.10 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.10 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.11 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.11 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.12 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.12 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.14 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.14 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.15 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.15 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.16 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.16 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.17 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.17 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.18 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.18 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.19 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.19 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.50 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.61 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.60 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.52 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.74 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.84 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.94 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.04 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.14 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.25 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.78 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.96 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.26 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.66 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.65 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.91 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.16 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.41 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.66 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.91 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 11.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 71% memory usage. Resuming worker. Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.92 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.22 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.46 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.69 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 11.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.12 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.47 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.73 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.78 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 63% memory usage. Resuming worker. Process memory: 9.45 GiB -- Worker memory limit: 14.90 GiB

```python
if FAILED_CHECKS > 0:
    raise Exception(f"There were {FAILED_CHECKS} failed checks.")
```

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-09-28

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.27.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-190.o2.rc.hms.harvard.edu

    Git branch: update-data

    seaborn   : 0.11.2
    plotnine  : 0.8.0
    pandas    : 1.2.3
    numpy     : 1.20.1
    dask      : 2021.5.1
    matplotlib: 3.3.4

```python

```
