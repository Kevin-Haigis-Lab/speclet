# Check of modeling data

```python
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, progress
```

```python
client = Client(n_workers=4, threads_per_worker=2, memory_limit="16GB")
client
```

<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://127.0.0.1:33827</li>
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

Papermill parameters:

`depmap_modeling_df`: The path to the full DepMap modeling data set.

```python
DEPMAP_MODELING_DF: str = ""
```

```python
# Parameters
DEPMAP_MODELING_DF = "../modeling_data/depmap-modeling-data.csv"
```

```python
assert DEPMAP_MODELING_DF != "", "No path provided for the modeling data."
```

```python
depmap_modeling_df_path = Path(DEPMAP_MODELING_DF)

if not depmap_modeling_df_path.exists():
    raise FileNotFoundError(f"Could not find '{str(depmap_modeling_df_path)}'")
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.139595</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.923715</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
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
      <td>TGAGCTGGCAATGCTAGAT</td>
      <td>OVR3_c905R1</td>
      <td>0.565344</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_155774116_-</td>
      <td>SPRY3</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>155774116</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>196</th>
      <td>TGATGGAGCGAATCAGATG</td>
      <td>OVR3_c905R1</td>
      <td>-0.204959</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr16_66934065_+</td>
      <td>CIAO2B</td>
      <td>sanger</td>
      <td>True</td>
      <td>16</td>
      <td>66934065</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>197</th>
      <td>TGCACTTATGTGTGCCGCC</td>
      <td>OVR3_c905R1</td>
      <td>0.650650</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chrX_156003692_-</td>
      <td>IL9R</td>
      <td>sanger</td>
      <td>True</td>
      <td>X</td>
      <td>156003692</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>198</th>
      <td>TGCTAGGACCCAACTGAGC</td>
      <td>OVR3_c905R1</td>
      <td>-0.517796</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr10_46580364_+</td>
      <td>SYT15</td>
      <td>sanger</td>
      <td>True</td>
      <td>10</td>
      <td>46580364</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.752471</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
    </tr>
    <tr>
      <th>199</th>
      <td>TGGAAAGTTGCCTCGTCCG</td>
      <td>OVR3_c905R1</td>
      <td>-0.218348</td>
      <td>CRISPR_C6596666.sample</td>
      <td>chr1_117622978_-</td>
      <td>TENT5C</td>
      <td>sanger</td>
      <td>True</td>
      <td>1</td>
      <td>117622978</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60</td>
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
        "counts_initial": "float64",
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.139595</td>
      <td>ovary</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.656377</td>
      <td>ovary</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.923715</td>
      <td>ovary</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.352975</td>
      <td>ovary</td>
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
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.157211</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
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

    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.82 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.82 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.83 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.83 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.84 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.84 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.84 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.85 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.85 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.86 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.86 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.87 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.87 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.88 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.88 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.89 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.89 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.90 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.91 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.91 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.92 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 11.92 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.92 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.93 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.93 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.94 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.94 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.96 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.96 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.96 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.97 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.97 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.97 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.98 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.98 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.98 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.01 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.01 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.02 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.02 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.02 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.03 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.03 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.04 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.04 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.05 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.05 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.06 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.06 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.07 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.07 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.09 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 61% memory usage. Resuming worker. Process memory: 9.09 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.61 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.11 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.18 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.24 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.30 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.38 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.47 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.55 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.63 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.72 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.17 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.24 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.53 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.29 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.56 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.82 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.91 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 12.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.09 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.18 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.27 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.35 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.44 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 79% memory usage. Resuming worker. Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.83 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.88 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.88 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 81% memory usage. Pausing worker.  Process memory: 12.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.28 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.49 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 75% memory usage. Resuming worker. Process memory: 11.23 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.23 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.39 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.54 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.67 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.76 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.76 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.11 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.43 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.78 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 11.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.99 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.19 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.38 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.55 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.72 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.89 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.05 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.18 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 79% memory usage. Resuming worker. Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.82 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 81% memory usage. Pausing worker.  Process memory: 12.10 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.10 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.38 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.61 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.76 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.23 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.35 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.48 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.94 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 14.03 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 14.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.58 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 70% memory usage. Resuming worker. Process memory: 10.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.65 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.90 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.64 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.89 GiB -- Worker memory limit: 14.90 GiB

```python
if FAILED_CHECKS > 0:
    raise Exception(f"There were {FAILED_CHECKS} failed checks.")
```

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-02-01

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.27.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1160.45.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-178.o2.rc.hms.harvard.edu

    Git branch: larger-subsample

    numpy : 1.20.1
    pandas: 1.2.3
    dask  : 2021.5.1

```python

```
