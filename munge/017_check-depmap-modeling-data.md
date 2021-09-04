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

    /n/data1/hms/dbmi/park/Cook/speclet/.snakemake/conda/6d81efe674885d090d2907cb94e4eefa/lib/python3.9/site-packages/distributed/node.py:160: UserWarning: Port 8787 is already in use.
    Perhaps you already have a cluster running?
    Hosting the HTTP server on port 45340 instead

<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://127.0.0.1:33315</li>
  <li><b>Dashboard: </b><a href='http://127.0.0.1:45340/status' target='_blank'>http://127.0.0.1:45340/status</a></li>
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
depmap_modeling_df = dd.read_csv(
    depmap_modeling_df_path, dtype={"age": "float64"}, low_memory=False
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
      <td>0.847995</td>
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
      <td>0.700605</td>
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
      <td>0.934918</td>
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
      <td>1.259171</td>
      <td>ovary</td>
      <td>metastasis</td>
      <td>False</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>

```python
depmap_modeling_df.shape
```

    (Delayed('int-f850eef8-d655-4aff-95f7-41ecb5e7b12c'), 23)

```python
depmap_modeling_df.columns
```

    Index(['sgrna', 'replicate_id', 'lfc', 'p_dna_batch', 'genome_alignment',
           'hugo_symbol', 'screen', 'multiple_hits_on_gene', 'sgrna_target_chr',
           'sgrna_target_pos', 'depmap_id', 'read_counts', 'rna_expr',
           'num_mutations', 'any_deleterious', 'any_tcga_hotspot',
           'any_cosmic_hotspot', 'is_mutated', 'copy_number', 'lineage',
           'primary_or_metastasis', 'is_male', 'age'],
          dtype='object')

## Basic checks

```python
FAILED_CHECKS = 0
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

    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.60 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.60 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.60 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.61 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.62 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.62 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.63 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.64 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.64 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.65 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.66 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.66 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.67 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.67 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.68 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.69 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.69 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.70 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.71 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.71 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.72 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.73 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.73 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.74 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.75 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.75 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.76 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.77 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.77 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.78 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.79 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.81 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.82 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.48 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.73 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.86 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.95 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.04 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.12 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.21 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.30 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.39 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.82 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.87 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.13 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.40 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.91 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.17 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.43 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.53 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.62 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.71 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.80 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.89 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 11.97 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.97 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.06 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 76% memory usage. Resuming worker. Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.50 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.57 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.83 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 81% memory usage. Pausing worker.  Process memory: 12.09 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.09 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 72% memory usage. Resuming worker. Process memory: 10.86 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.86 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.08 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.32 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.66 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.97 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.28 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.54 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.75 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 11.96 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.96 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.17 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.38 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.59 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.77 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 76% memory usage. Resuming worker. Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.45 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.46 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.46 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 11.73 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 80% memory usage. Pausing worker.  Process memory: 12.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.00 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.23 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.41 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.64 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.87 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.02 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.30 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.59 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 13.60 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 12.19 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Worker is at 68% memory usage. Resuming worker. Process memory: 10.14 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.51 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.74 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.48 GiB -- Worker memory limit: 14.90 GiB
    distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.  Perhaps some other process is leaking memory?  Process memory: 10.72 GiB -- Worker memory limit: 14.90 GiB

```python
if FAILED_CHECKS > 0:
    raise Exception(f"There were {FAILED_CHECKS} failed checks.")
```

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-09-04

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.27.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-h-17-54.o2.rc.hms.harvard.edu

    Git branch: read-count-data

    plotnine  : 0.8.0
    seaborn   : 0.11.2
    dask      : 2021.5.1
    numpy     : 1.20.1
    pandas    : 1.2.3
    matplotlib: 3.3.4

```python

```
