```python
import re
import string
import warnings
from pathlib import Path
from time import time

import arviz as az
import common_data_processing as dphelp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import pymc3_helpers as pmhelp
import seaborn as sns
from theano import tensor as tt

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_minimal())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 245
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("pymc3_model_cache")
```

# Modeling with a hierarchical 2D covariate with additional varying levels

This notebook is to experiment with creating a hierarchical covariate with 2 dimensions: gene $\times$ cell line.
Each value in the matrix will come from a model of two varying covariables, one for the gene and one for the cell line.
The cell line covariate will then come from a distribution of the lineage as another level in the model.

## Data preparation

I will need a fair amount of data to fit a model with so many parameters, but I will still start with a subsample of the full data set.

```python
NUM_GENES = 20
NUM_LINEAGES = 5

data_path = Path("../modeling_data")
starting_dataset_path = data_path / "depmap_modeling_dataframe.csv"
final_dataset_path = (
    data_path / "subsample_covariate-matrix_depmap-modeling-dataframe.csv"
)

if final_dataset_path.exists() and False:
    # Read in data set that has already been created.
    print("Reading in existing file.")
    modeling_data = pd.read_csv(final_dataset_path)
else:
    # Read in full data set and subsample some genes and lineages.
    print(
        f"Subsampling {NUM_GENES} genes and {NUM_LINEAGES} lineages from full dataset."
    )
    full_modeling_data = pd.read_csv(starting_dataset_path, low_memory=False)

    GENES = np.random.choice(full_modeling_data.hugo_symbol.unique(), NUM_GENES)
    LINEAGES = np.random.choice(full_modeling_data.lineage.unique(), NUM_LINEAGES)

    modeling_data = full_modeling_data[full_modeling_data.hugo_symbol.isin(GENES)]
    modeling_data = modeling_data[modeling_data.lineage.isin(LINEAGES)]
    modeling_data = modeling_data.reset_index(drop=True)

    modeling_data.to_csv(final_dataset_path)
    del full_modeling_data
```

    Subsampling 20 genes and 5 lineages from full dataset.

```python
print("Genes:")
print(modeling_data.hugo_symbol.unique().tolist())

print("-" * 60)
print("Lineages:")
print(modeling_data.lineage.unique().tolist())
```

    Genes:
    ['C18orf54', 'ARG1', 'DHDH', 'RGPD6', 'H3F3A', 'STRC', 'GIPR', 'FAM206A', 'YPEL5', 'GTF2E1', 'AAED1', 'LGI3', 'ZNF175', 'ARHGAP26', 'IL17B', 'HERC1', 'PLEKHH3', 'NACC2', 'PRKAR1A', 'CT47A7']
    ------------------------------------------------------------
    Lineages:
    ['blood', 'upper_aerodigestive', 'colorectal', 'liver']

```python
modeling_data.shape
```

    (22960, 27)

```python
modeling_data.head(5)
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
      <th>pdna_batch</th>
      <th>passes_qc</th>
      <th>depmap_id</th>
      <th>primary_or_metastasis</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>kras_mutation</th>
      <th>...</th>
      <th>log2_gene_cn_p1</th>
      <th>gene_cn</th>
      <th>n_muts</th>
      <th>any_deleterious</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAGAAAAACAAGAAATGCCG</td>
      <td>697-311cas9_repa_p6_batch3</td>
      <td>-0.317445</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000070</td>
      <td>Primary</td>
      <td>blood</td>
      <td>ALL</td>
      <td>WT</td>
      <td>...</td>
      <td>0.985476</td>
      <td>1.679088</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2.629939</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAGCATCTCTGGCCATGCCA</td>
      <td>697-311cas9_repa_p6_batch3</td>
      <td>0.216901</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000070</td>
      <td>Primary</td>
      <td>blood</td>
      <td>ALL</td>
      <td>WT</td>
      <td>...</td>
      <td>0.971533</td>
      <td>1.641990</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACAGAAACACGACATCCCCA</td>
      <td>697-311cas9_repa_p6_batch3</td>
      <td>0.014907</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000070</td>
      <td>Primary</td>
      <td>blood</td>
      <td>ALL</td>
      <td>WT</td>
      <td>...</td>
      <td>1.009784</td>
      <td>1.745008</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.056584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACTCACCTGTCGAGGCGACG</td>
      <td>697-311cas9_repa_p6_batch3</td>
      <td>-0.987370</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000070</td>
      <td>Primary</td>
      <td>blood</td>
      <td>ALL</td>
      <td>WT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.454176</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACTGCCCGCAAATCGACCGG</td>
      <td>697-311cas9_repa_p6_batch3</td>
      <td>-0.712884</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000070</td>
      <td>Primary</td>
      <td>blood</td>
      <td>ALL</td>
      <td>WT</td>
      <td>...</td>
      <td>1.300406</td>
      <td>2.670787</td>
      <td>1</td>
      <td>False</td>
      <td>missense_mutation</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>TRUE</td>
      <td>False</td>
      <td>10.285865</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>

```python
for col in [
    "sgrna",
    "pdna_batch",
    "depmap_id",
    "lineage",
    "kras_mutation",
    "hugo_symbol",
]:
    modeling_data = dphelp.make_cat(modeling_data, col, ordered=True, sort_cats=True)
```

## Data visualization

```python
plot_data = (
    modeling_data[["lineage", "depmap_id"]]
    .drop_duplicates()
    .groupby("lineage")
    .count()
    .reset_index()
)

(
    gg.ggplot(plot_data, gg.aes("lineage", "depmap_id"))
    + gg.geom_col(gg.aes(fill="depmap_id"))
    + gg.geom_text(gg.aes(label="depmap_id"), va="bottom")
    + gg.scale_fill_gradient(guide=None)
    + gg.scale_y_continuous(expand=(0, 0, 0.05, 0))
    + gg.labs(x="lineage", y="number of cell lines")
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_9_0.png)

    <ggplot: (8765464161009)>

```python
plot_data = (
    modeling_data[["lineage", "hugo_symbol", "lfc"]]
    .groupby(["lineage", "hugo_symbol"])
    .mean()
    .reset_index()
)

(
    gg.ggplot(plot_data, gg.aes(x="lineage", y="hugo_symbol", fill="lfc"))
    + gg.geom_tile()
    + gg.scale_x_discrete(expand=(0.1, 0.1))
    + gg.labs(x="lineage", y="gene", fill="avg. LFC")
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_10_0.png)

    <ggplot: (8765464171945)>

```python
(
    gg.ggplot(modeling_data, gg.aes(x="hugo_symbol", y="lfc"))
    + gg.geom_boxplot(
        gg.aes(color="lineage", fill="lineage"),
        alpha=0.4,
        outlier_alpha=0.4,
        outlier_size=0.4,
    )
    + gg.theme(
        figure_size=(10, 5),
        axis_text_x=gg.element_text(angle=90),
        legend_position=(0.22, 0.25),
    )
    + gg.labs(x="gene", y="LFC", fill="lineage")
)
```

![png](010_015_gene-lineage-hierarchical-matrix_files/010_015_gene-lineage-hierarchical-matrix_11_0.png)

    <ggplot: (8765464034314)>

```python

```

```python

```

```python

```

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 28.69 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-01-27

    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.19.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-230.o2.rc.hms.harvard.edu

    Git branch: data-subset-model

    seaborn   : 0.11.1
    plotnine  : 0.7.1
    matplotlib: 3.3.3
    re        : 2.2.1
    numpy     : 1.19.5
    pymc3     : 3.9.3
    arviz     : 0.11.0
    theano    : 1.0.5
    pandas    : 1.2.0

```python

```
