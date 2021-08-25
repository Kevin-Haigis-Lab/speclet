```python
%load_ext autoreload
%autoreload 2
```

```python
import warnings
from pathlib import Path
from time import time

import dask
import dask.dataframe as dd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import seaborn as sns
from dask.distributed import Client, progress
```

```python
from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.io import cache_io, data_io
from src.loggers import logger
from src.plot.color_pal import SeabornColor
```

```python
notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(
    gg.theme_bw()
    + gg.theme(
        figure_size=(4, 4),
        axis_ticks_major=gg.element_blank(),
        strip_background=gg.element_blank(),
    )
)
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)
```

## Setup

```python
client = Client(n_workers=2, threads_per_worker=2, memory_limit="10GB")
```

```python
data_path = data_io.data_path(data_io.DataFile.achilles_data)
if data_path.exists():
    print(f"data file: '{data_path.as_posix()}'")
else:
    raise FileNotFoundError(data_path)
```

    data file: '/n/data1/hms/dbmi/park/Cook/speclet/modeling_data/depmap_modeling_dataframe.csv'

```python
data_types: dict[str, str] = {"age": "float64"}
for a in ["p_dna_batch", "replicate_id", "primary_or_metastasis", "is_male"]:
    data_types[a] = "category"

achilles_data = dd.read_csv(data_path, dtype=data_types, low_memory=False)
```

## Basic statistics

```python
achilles_data.head()
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
      <td>FALSE</td>
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
      <td>FALSE</td>
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
      <td>FALSE</td>
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
      <td>FALSE</td>
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
      <td>FALSE</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>

```python
achilles_data.columns
```

    Index(['sgrna', 'replicate_id', 'lfc', 'p_dna_batch', 'genome_alignment',
           'hugo_symbol', 'screen', 'multiple_hits_on_gene', 'sgrna_target_chr',
           'sgrna_target_pos', 'depmap_id', 'rna_expr', 'num_mutations',
           'any_deleterious', 'any_tcga_hotspot', 'any_cosmic_hotspot',
           'is_mutated', 'copy_number', 'lineage', 'primary_or_metastasis',
           'is_male', 'age'],
          dtype='object')

```python
print("stats for all data:")

for col, lbl in {
    "sgrna": "sgRNA",
    "hugo_symbol": "genes",
    "depmap_id": "cell lines",
    "lineage": "lineages",
    "screen": "data sources",
}.items():
    count = len(achilles_data[col].unique().compute())
    print(f"  num {lbl}: {count:,}")
```

    stats for all data:
      num sgRNA: 157,808
      num genes: 18,182
      num cell lines: 984
      num lineages: 27
      num data sources: 2

```python
broad_achilles_data = achilles_data.query("screen == 'broad'")
```

```python
print("stats for broad data:")

for col, lbl in {
    "sgrna": "sgRNA",
    "hugo_symbol": "genes",
    "depmap_id": "cell lines",
    "lineage": "lineages",
}.items():
    count = len(broad_achilles_data[col].unique().compute())
    print(f"  num {lbl}: {count:,}")
```

    stats for broad data:
      num sgRNA: 71,062
      num genes: 18,119
      num cell lines: 853
      num lineages: 27

```python
gene_sgrna_map = (
    broad_achilles_data[["hugo_symbol", "sgrna"]].drop_duplicates().compute()
)
```

```python
gene_sgrna_map.groupby("hugo_symbol").count().reset_index(drop=False).describe()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>18119.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.921960</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.384721</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
sampled_dataset = broad_achilles_data.sample(frac=0.001).compute()
```

```python
sampled_dataset.shape
```

    (60620, 22)

```python
(
    gg.ggplot(sampled_dataset.sample(n=1000), gg.aes(x="lfc"))
    + gg.geom_density(
        size=1.2,
        alpha=0.2,
        fill=SeabornColor.BLUE,
        color=SeabornColor.BLUE,
    )
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.theme(figure_size=(4, 3))
    + gg.labs(x="log-fold change", y="density")
)
```

![png](001_001_basic-data-statistics-and-plots_files/001_001_basic-data-statistics-and-plots_18_0.png)

    <ggplot: (8731168961003)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 27.66 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-08-23

    Python implementation: CPython
    Python version       : 3.9.6
    IPython version      : 7.26.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-230.o2.rc.hms.harvard.edu

    Git branch: fix-theano-lock

    numpy     : 1.21.2
    dask      : 2021.8.0
    pandas    : 1.3.2
    seaborn   : 0.11.2
    matplotlib: 3.4.3
    plotnine  : 0.8.0
