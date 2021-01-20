# A hierarchcial linear regression to fit CRISPR-Cas9 screen results

This notebook is intended to experiment with various hierarchical model architectures on a sub-sample of the real DepMap data.

## Set-up

```python
import re
import string
import warnings
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

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
import string_functions as stringr
from theano import tensor as tt

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_minimal())

%config InlineBackend.figure_format = 'retina'

RANDOM_SEED = 824
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("pymc3_model_cache")
```

```python
mut_pal = {"False": "#429DD6", "True": "#B93174"}
```

## Data preparation

```python
data_path = Path("../modeling_data/depmap_modeling_dataframe_subsample.csv")
data = pd.read_csv(data_path)

data = data.sort_values(["hugo_symbol", "sgrna", "depmap_id"]).reset_index(drop=True)
for col in ("hugo_symbol", "depmap_id", "sgrna", "lineage", "chromosome"):
    data = dphelp.make_cat(data, col, ordered=True, sort_cats=False)

data = dphelp.zscale_cna_by_group(data, cn_max=10)

data.head(n=7)
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
      <th>gene_cn</th>
      <th>n_muts</th>
      <th>any_deleterious</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>gene_cn_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.029491</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>2.632957</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.632215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.426017</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>2.632957</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.632215</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>0.008626</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>1.594524</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.280821</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>1.594524</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.239815</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>1.594524</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>hcc827-311cas9 rep a p6_batch3</td>
      <td>-0.170583</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000012</td>
      <td>Primary</td>
      <td>lung</td>
      <td>NSCLC</td>
      <td>WT</td>
      <td>...</td>
      <td>1.741667</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.042644</td>
      <td>-0.082165</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>hcc827-311cas9 rep b p6_batch3</td>
      <td>-0.320019</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000012</td>
      <td>Primary</td>
      <td>lung</td>
      <td>NSCLC</td>
      <td>WT</td>
      <td>...</td>
      <td>1.741667</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.042644</td>
      <td>-0.082165</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 28 columns</p>
</div>

```python
data.describe()
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
      <th>lfc</th>
      <th>pdna_batch</th>
      <th>n_alignments</th>
      <th>chrom_pos</th>
      <th>segment_mean</th>
      <th>segment_cn</th>
      <th>log2_gene_cn_p1</th>
      <th>gene_cn</th>
      <th>n_muts</th>
      <th>rna_expr</th>
      <th>gene_cn_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>60049.000000</td>
      <td>60049.000000</td>
      <td>60049.000000</td>
      <td>6.004900e+04</td>
      <td>5.984300e+04</td>
      <td>5.984300e+04</td>
      <td>6.004900e+04</td>
      <td>6.004900e+04</td>
      <td>60049.000000</td>
      <td>60049.000000</td>
      <td>6.004900e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.110098</td>
      <td>2.861063</td>
      <td>1.029126</td>
      <td>8.199159e+07</td>
      <td>1.052517e+00</td>
      <td>3.673521e+12</td>
      <td>1.012354e+00</td>
      <td>1.894709e+00</td>
      <td>0.097654</td>
      <td>2.957374</td>
      <td>-3.975792e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.562585</td>
      <td>0.355663</td>
      <td>0.168162</td>
      <td>6.014320e+07</td>
      <td>8.296245e-01</td>
      <td>3.177012e+14</td>
      <td>2.294065e-01</td>
      <td>4.336833e+00</td>
      <td>0.354338</td>
      <td>2.292989</td>
      <td>1.000008e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-4.972325</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>7.675103e+06</td>
      <td>9.598765e-08</td>
      <td>1.000000e+00</td>
      <td>5.506197e-09</td>
      <td>5.506197e-09</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.065984e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.344279</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.201892e+07</td>
      <td>8.530658e-01</td>
      <td>1.806335e+00</td>
      <td>8.900191e-01</td>
      <td>1.435176e+00</td>
      <td>0.000000</td>
      <td>0.790772</td>
      <td>-5.710118e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.033168</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>5.977335e+07</td>
      <td>1.005954e+00</td>
      <td>2.008272e+00</td>
      <td>1.004289e+00</td>
      <td>1.729965e+00</td>
      <td>0.000000</td>
      <td>2.895303</td>
      <td>-1.340551e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.218036</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.375787e+08</td>
      <td>1.154679e+00</td>
      <td>2.226348e+00</td>
      <td>1.107473e+00</td>
      <td>2.026700e+00</td>
      <td>0.000000</td>
      <td>4.698218</td>
      <td>4.430254e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.598174</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>2.230033e+08</td>
      <td>5.460920e+01</td>
      <td>2.747932e+16</td>
      <td>5.797252e+00</td>
      <td>3.283930e+02</td>
      <td>8.000000</td>
      <td>11.496005</td>
      <td>1.055240e+01</td>
    </tr>
  </tbody>
</table>
</div>

```python
data.shape
```

    (60049, 28)

## EDA

```python
data.columns
```

    Index(['sgrna', 'replicate_id', 'lfc', 'pdna_batch', 'passes_qc', 'depmap_id',
           'primary_or_metastasis', 'lineage', 'lineage_subtype', 'kras_mutation',
           'genome_alignment', 'n_alignments', 'hugo_symbol', 'chromosome',
           'chrom_pos', 'segment_mean', 'segment_cn', 'log2_gene_cn_p1', 'gene_cn',
           'n_muts', 'any_deleterious', 'variant_classification', 'is_deleterious',
           'is_tcga_hotspot', 'is_cosmic_hotspot', 'mutated_at_guide_location',
           'rna_expr', 'gene_cn_z'],
          dtype='object')

```python
(
    gg.ggplot(data, gg.aes(x="hugo_symbol", y="lfc"))
    + gg.geom_boxplot(outlier_alpha=0.3, outlier_size=0.5)
    + gg.theme(axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1))
    + gg.labs(x=None, y="log fold change")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_11_0.png)

    <ggplot: (8771926491986)>

```python
p = (
    gg.ggplot(data, gg.aes(x="gene_cn_z", y="lfc"))
    + gg.geom_point(gg.aes(color="hugo_symbol"), alpha=0.1, size=0.5)
    + gg.geom_smooth()
    + gg.scale_color_discrete(
        guide=gg.guide_legend(override_aes={"size": 1, "alpha": 1})
    )
    + gg.labs(
        x="gene cn (z-scaled)",
        y="log fold change",
        title="Correlation between CN and LFC",
        color="gene",
    )
)
p
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_12_0.png)

    <ggplot: (8772018675559)>

The general trend is that gene CN is negatively correlated with logFC, but there is substantial variability per gene.

```python
(
    p
    + gg.geom_smooth(
        gg.aes(color="hugo_symbol"),
        linetype="--",
        alpha=0.5,
        se=False,
        show_legend=False,
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_14_0.png)

    <ggplot: (8772018121698)>

The varying effects per gene are more striking than the varying efferects per lineage.
The effect of lineage/cell line may need to be regulated with relatively strict priors.

```python
d = data.groupby(["lineage", "hugo_symbol"]).mean().reset_index(drop=False)
d["lineage"] = stringr.str_replace(d["lineage"], "_", " ")
d["lineage"] = stringr.str_wrap(d["lineage"], width=10)

(
    gg.ggplot(d, gg.aes(x="lineage", y="hugo_symbol", fill="lfc"))
    + gg.geom_tile()
    + gg.scale_x_discrete(expand=(0, 0.5, 0, 0.5))
    + gg.scale_y_discrete(expand=(0, 0.5, 0, 0.5))
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_16_0.png)

    <ggplot: (8771915270337)>

```python
data.head()
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
      <th>gene_cn</th>
      <th>n_muts</th>
      <th>any_deleterious</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>gene_cn_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.029491</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>2.632957</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.632215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.426017</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>2.632957</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.632215</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>0.008626</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>1.594524</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.280821</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>1.594524</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.239815</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>1.594524</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>

---

## Modeling

### Model 1. Hierarchical model by gene with no other variables

$
y \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_{g} \\
\alpha_g \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha) \\
\mu_\alpha \sim \mathcal{N}(0, 5) \\
\sigma_\alpha \sim \text{HalfNormal}(0, 5) \\
\sigma \sim \text{HalfNormal}(0, 5)
$

```python
gene_idx = dphelp.get_indices(data, "hugo_symbol")
num_genes = data["hugo_symbol"].nunique()
with pm.Model() as m1:
    σ_α = pm.HalfNormal("σ_α", 5.0)
    μ_α = pm.Normal("μ_α", 0, 5)

    α_g = pm.Normal("α_g", μ_α, σ_α, shape=num_genes)
    μ = pm.Deterministic("μ", α_g[gene_idx])
    σ = pm.HalfNormal("σ", 5.0)

    y = pm.Normal("y", mu=μ, sigma=σ, observed=data.lfc)
```

```python
pm.model_to_graphviz(m1)
```

![svg](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_20_0.svg)

```python
m1_cache_dir = pymc3_cache_dir / "subset_speclet_m1"

m1_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m1,
    num_mcmc=1000,
    tune=1000,
    chains=2,
    cores=2,
    random_seed=RANDOM_SEED,
    cache_dir=pymc3_cache_dir / m1_cache_dir,
    force=False,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 40000},
)
```

    Loading cached trace and posterior sample...

```python
m1_az = pmhelp.samples_to_arviz(model=m1, res=m1_sampling_results)
```

    posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

It looks like the effects of the genes are confidently estimated with most of the posterior distributions lying far from 0.

```python
az.plot_trace(m1_az, var_names="α_g", compact=True)
plt.show()
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_24_0.png)

```python
az.summary(m1_az, var_names=["α_g"])
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>α_g[0]</th>
      <td>-0.055</td>
      <td>0.010</td>
      <td>-0.073</td>
      <td>-0.036</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4313.0</td>
      <td>4313.0</td>
      <td>4292.0</td>
      <td>1314.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[1]</th>
      <td>-0.111</td>
      <td>0.010</td>
      <td>-0.132</td>
      <td>-0.094</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4567.0</td>
      <td>4401.0</td>
      <td>4552.0</td>
      <td>1031.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[2]</th>
      <td>-0.107</td>
      <td>0.010</td>
      <td>-0.126</td>
      <td>-0.088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3800.0</td>
      <td>3756.0</td>
      <td>3805.0</td>
      <td>1332.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[3]</th>
      <td>0.094</td>
      <td>0.010</td>
      <td>0.075</td>
      <td>0.113</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3779.0</td>
      <td>3676.0</td>
      <td>3804.0</td>
      <td>1266.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[4]</th>
      <td>-0.177</td>
      <td>0.011</td>
      <td>-0.197</td>
      <td>-0.157</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3689.0</td>
      <td>3689.0</td>
      <td>3726.0</td>
      <td>1335.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[5]</th>
      <td>0.060</td>
      <td>0.011</td>
      <td>0.039</td>
      <td>0.079</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4802.0</td>
      <td>4237.0</td>
      <td>4833.0</td>
      <td>1550.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[6]</th>
      <td>0.009</td>
      <td>0.010</td>
      <td>-0.011</td>
      <td>0.028</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3388.0</td>
      <td>1518.0</td>
      <td>3394.0</td>
      <td>1288.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[7]</th>
      <td>0.065</td>
      <td>0.012</td>
      <td>0.043</td>
      <td>0.088</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4077.0</td>
      <td>3509.0</td>
      <td>4168.0</td>
      <td>1133.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[8]</th>
      <td>0.069</td>
      <td>0.010</td>
      <td>0.051</td>
      <td>0.089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3038.0</td>
      <td>3036.0</td>
      <td>3028.0</td>
      <td>1474.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[9]</th>
      <td>0.090</td>
      <td>0.010</td>
      <td>0.070</td>
      <td>0.109</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3592.0</td>
      <td>3431.0</td>
      <td>3576.0</td>
      <td>1309.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[10]</th>
      <td>-0.775</td>
      <td>0.011</td>
      <td>-0.795</td>
      <td>-0.756</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4350.0</td>
      <td>4350.0</td>
      <td>4288.0</td>
      <td>1441.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[11]</th>
      <td>-0.005</td>
      <td>0.011</td>
      <td>-0.024</td>
      <td>0.016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3689.0</td>
      <td>873.0</td>
      <td>3665.0</td>
      <td>1358.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[12]</th>
      <td>0.006</td>
      <td>0.011</td>
      <td>-0.012</td>
      <td>0.027</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4356.0</td>
      <td>1200.0</td>
      <td>4380.0</td>
      <td>1508.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[13]</th>
      <td>-0.211</td>
      <td>0.011</td>
      <td>-0.232</td>
      <td>-0.190</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4614.0</td>
      <td>4613.0</td>
      <td>4575.0</td>
      <td>1326.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[14]</th>
      <td>-0.394</td>
      <td>0.011</td>
      <td>-0.415</td>
      <td>-0.375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4111.0</td>
      <td>4078.0</td>
      <td>4062.0</td>
      <td>1334.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[15]</th>
      <td>0.117</td>
      <td>0.010</td>
      <td>0.098</td>
      <td>0.136</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3779.0</td>
      <td>3779.0</td>
      <td>3763.0</td>
      <td>1303.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[16]</th>
      <td>-0.076</td>
      <td>0.011</td>
      <td>-0.097</td>
      <td>-0.056</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6602.0</td>
      <td>6278.0</td>
      <td>6602.0</td>
      <td>1352.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>α_g[17]</th>
      <td>-0.326</td>
      <td>0.010</td>
      <td>-0.346</td>
      <td>-0.307</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3796.0</td>
      <td>3748.0</td>
      <td>3829.0</td>
      <td>1355.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[18]</th>
      <td>-0.654</td>
      <td>0.011</td>
      <td>-0.672</td>
      <td>-0.632</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4896.0</td>
      <td>4852.0</td>
      <td>4427.0</td>
      <td>1307.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[19]</th>
      <td>0.132</td>
      <td>0.010</td>
      <td>0.113</td>
      <td>0.153</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4328.0</td>
      <td>4192.0</td>
      <td>4332.0</td>
      <td>1105.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[20]</th>
      <td>-0.085</td>
      <td>0.011</td>
      <td>-0.104</td>
      <td>-0.064</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4110.0</td>
      <td>3942.0</td>
      <td>4153.0</td>
      <td>1423.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[21]</th>
      <td>0.096</td>
      <td>0.010</td>
      <td>0.076</td>
      <td>0.114</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3997.0</td>
      <td>3900.0</td>
      <td>4039.0</td>
      <td>1488.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[22]</th>
      <td>0.073</td>
      <td>0.010</td>
      <td>0.055</td>
      <td>0.092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5721.0</td>
      <td>5355.0</td>
      <td>5741.0</td>
      <td>1494.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[23]</th>
      <td>0.026</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>0.046</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4642.0</td>
      <td>2591.0</td>
      <td>4658.0</td>
      <td>1199.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>α_g[24]</th>
      <td>-0.450</td>
      <td>0.010</td>
      <td>-0.468</td>
      <td>-0.430</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4651.0</td>
      <td>4651.0</td>
      <td>4646.0</td>
      <td>1417.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>α_g[25]</th>
      <td>-0.231</td>
      <td>0.011</td>
      <td>-0.251</td>
      <td>-0.211</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3723.0</td>
      <td>3723.0</td>
      <td>3750.0</td>
      <td>1609.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

```python
def calculate_hdi_labels(hdi_prob: float, prefix: str = "hdi_") -> Tuple[str, str]:
    low = 100 * (1 - hdi_prob) / 2
    high = 100 - low
    return f"{prefix}{low:.1f}%", f"{prefix}{high:.1f}%"


def logfc_model_ppc_dataframe(
    az_obj: az.InferenceData,
    real_values: pd.Series,
    to_merge_with: Optional[pd.DataFrame] = None,
    hdi_prob: float = 0.89,
    var_name: str = "y",
) -> pd.DataFrame:
    ppc_arry = np.asarray(az_obj.posterior_predictive[var_name]).squeeze()
    ppc_summary = pd.DataFrame(
        az.hdi(ppc_arry, hdi_prob=hdi_prob), columns=calculate_hdi_labels(hdi_prob)
    )
    ppc_summary["mean"] = np.mean(ppc_arry, axis=0)
    ppc_summary = ppc_summary.reset_index(drop=True)
    ppc_summary["real_value"] = real_values
    ppc_summary["error"] = ppc_summary["mean"] - ppc_summary["real_value"]

    if not to_merge_with is None:
        ppc_summary = ppc_summary.merge(
            to_merge_with.reset_index(drop=True), left_index=True, right_index=True
        )

    return ppc_summary
```

```python
ppc_m1_summary = logfc_model_ppc_dataframe(
    m1_az, data.lfc, data[["hugo_symbol", "sgrna", "depmap_id"]]
)
ppc_m1_summary.head()
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

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
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mean</th>
      <th>real_value</th>
      <th>error</th>
      <th>hugo_symbol</th>
      <th>sgrna</th>
      <th>depmap_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.857051</td>
      <td>0.772856</td>
      <td>-0.041082</td>
      <td>0.029491</td>
      <td>-0.070573</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ACH-000007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.887173</td>
      <td>0.771502</td>
      <td>-0.062950</td>
      <td>0.426017</td>
      <td>-0.488967</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ACH-000007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.901727</td>
      <td>0.750155</td>
      <td>-0.073091</td>
      <td>0.008626</td>
      <td>-0.081717</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ACH-000009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.847794</td>
      <td>0.798100</td>
      <td>-0.032507</td>
      <td>0.280821</td>
      <td>-0.313328</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ACH-000009</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.903893</td>
      <td>0.742157</td>
      <td>-0.034015</td>
      <td>0.239815</td>
      <td>-0.273830</td>
      <td>ADAMTS13</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ACH-000009</td>
    </tr>
  </tbody>
</table>
</div>

The predictions are also tightly bunched, further indicating the model's confidence.

```python
p = (
    gg.ggplot(ppc_m1_summary, gg.aes(y="hugo_symbol"))
    + gg.geom_jitter(
        gg.aes(x="mean", color="sgrna"), width=0, height=0.35, alpha=0.3, size=0.3
    )
    + gg.scale_color_discrete(guide=None)
    + gg.theme(figure_size=(5, 6))
    + gg.labs(x="mean", y=None, title="Posterior predictions of gene effects")
)

p
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_29_0.png)

    <ggplot: (8771905084046)>

But we can see there is actually a lot of error, and some genes have far greater prediction error than others.
This is not unexpected as there are still many factors to add to the model.

```python
(p + gg.geom_jitter(gg.aes(x="real_value"), alpha=0.2, size=0.1, height=0.1, width=0))
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_31_0.png)

    <ggplot: (8771912916294)>

Importantly, we can see that there are differing levels of error per sgRNA for many genes.

```python
(
    gg.ggplot(ppc_m1_summary, gg.aes(y="error", x="hugo_symbol"))
    + gg.geom_point(
        gg.aes(color="sgrna"),
        position=gg.position_jitterdodge(
            jitter_width=0.2, jitter_height=0, dodge_width=0.8, random_state=RANDOM_SEED
        ),
        alpha=0.5,
        size=0.3,
    )
    + gg.geom_boxplot(
        gg.aes(color="sgrna"),
        position=gg.position_dodge(width=0.8),
        outlier_alpha=0,
        alpha=0.5,
        fill="white",
    )
    + gg.scale_color_discrete(guide=None)
    + gg.theme(figure_size=(10, 7), axis_text_x=gg.element_text(angle=90))
    + gg.labs(x=None, y="density", title="Distribution of prediction error")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_33_0.png)

    <ggplot: (8771926513324)>

```python
(
    gg.ggplot(ppc_m1_summary, gg.aes(x="error"))
    + gg.facet_wrap("hugo_symbol", scales="free", ncol=3)
    + gg.geom_density(fill="black", color=None, alpha=0.4)
    + gg.geom_density(gg.aes(color="sgrna"), fill=None)
    + gg.scale_color_discrete(guide=None)
    + gg.theme(figure_size=(8, 20), subplots_adjust={"hspace": 0.5, "wspace": 0.4})
    + gg.labs(x=None, y="density", title="Distribution of prediction error")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_34_0.png)

    <ggplot: (8771912888980)>

---

### Model 2. Add a layer for gene above sgRNA

Model 1 has a single varying intercept for genes.
This model will shift the gene-level predictor to a heriarchical level over sgRNA.
Therefore, sgRNA will be in the first level of the model and each sgRNA's distribution will come from a heirarchical distribution for the gene.

$
y \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_s \\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha_s}, \sigma_{\alpha_s}) \\
\qquad \mu_{\alpha_s} = g_s \\
\qquad \quad g_s \sim \mathcal{N}(\mu_g, \sigma_g) \\
\qquad \qquad \mu_g \sim \mathcal{N}(0, 5) \quad \sigma_g \sim \text{Exp}(1) \\
\qquad \sigma_\alpha \sim \text{Exp}(1) \\
\sigma \sim \text{HalfNormal}(5)
$

```python
num_sgrnas = data["sgrna"].nunique()
num_genes = data["hugo_symbol"].nunique()
print(f"{num_sgrnas} sgRNAs from {num_genes} genes")

sgrna_idx = dphelp.get_indices(data, "sgrna")

sgrna_to_gene_map = data[["sgrna", "hugo_symbol"]].drop_duplicates()
gene_idx = dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol")
```

    103 sgRNAs from 26 genes

```python
with pm.Model() as m2:
    μ_g = pm.Normal("μ_g", 0, 5)
    σ_g = pm.Exponential("σ_g", 1)

    g_s = pm.Normal("g_s", μ_g, σ_g, shape=num_genes)

    μ_α_s = pm.Deterministic("μ_α_s", g_s[gene_idx])
    σ_α_s = pm.Exponential("σ_α_s", 1)

    α_s = pm.Normal("α_s", μ_α_s, σ_α_s, shape=num_sgrnas)

    μ = pm.Deterministic("μ", α_s[sgrna_idx])
    σ = pm.HalfNormal("σ", 5)

    y = pm.Normal("y", μ, σ, observed=data.lfc)
```

```python
pm.model_to_graphviz(m2)
```

![svg](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_38_0.svg)

```python
m2_cache_dir = pymc3_cache_dir / "subset_speclet_m2"

m2_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m2,
    num_mcmc=2000,
    tune=4000,
    chains=2,
    cores=2,
    random_seed=RANDOM_SEED,
    cache_dir=pymc3_cache_dir / m2_cache_dir,
    force=False,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 40000, "target_accept": 0.9},
)
```

    Loading cached trace and posterior sample...

```python
m2_az = pmhelp.samples_to_arviz(model=m2, res=m2_sampling_results)
```

    posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

The distributions for the genes are much wider, now, but this is probably a more accurate representation of the truth by taking into account per-sgRNA variability.

```python
az.plot_trace(m2_az, var_names=["μ_g", "σ_g", "g_s", "α_s"], compact=True)
plt.show()
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_42_0.png)

```python
az.summary(m2_az, var_names=["μ_g", "σ_g", "σ_α_s"])
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ_g</th>
      <td>-0.109</td>
      <td>0.049</td>
      <td>-0.198</td>
      <td>-0.013</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>7060.0</td>
      <td>5293.0</td>
      <td>7127.0</td>
      <td>3154.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_g</th>
      <td>0.221</td>
      <td>0.043</td>
      <td>0.145</td>
      <td>0.302</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4759.0</td>
      <td>4728.0</td>
      <td>4693.0</td>
      <td>3332.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_α_s</th>
      <td>0.228</td>
      <td>0.019</td>
      <td>0.194</td>
      <td>0.264</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>4422.0</td>
      <td>4389.0</td>
      <td>4436.0</td>
      <td>3350.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
g_alpha_post = pd.DataFrame(np.asarray(m2_az.posterior["g_s"]).reshape(-1, num_genes))
ordered_genes = (
    sgrna_to_gene_map.sort_values("hugo_symbol").hugo_symbol.drop_duplicates().to_list()
)
g_alpha_post.columns = ordered_genes
g_alpha_post = g_alpha_post.melt(var_name="hugo_symbol", value_name="value")

g_alpha_summary = az.summary(m2_az, var_names="g_s", kind="stats", hdi_prob=0.89)
g_alpha_summary["hugo_symbol"] = ordered_genes

point_color = "#FA6A48"
(
    gg.ggplot(g_alpha_post.sample(frac=0.25), gg.aes(x="hugo_symbol", y="value"))
    + gg.geom_violin(color=None, fill="grey", alpha=0.5)
    + gg.geom_linerange(
        gg.aes(x="hugo_symbol", y="mean", ymin="hdi_5.5%", ymax="hdi_94.5%"),
        data=g_alpha_summary,
        color=point_color,
    )
    + gg.geom_point(
        gg.aes(x="hugo_symbol", y="mean"), data=g_alpha_summary, color="black"
    )
    + gg.scale_y_continuous(expand=(0.02, 0, 0.02, 0))
    + gg.theme(axis_text_x=gg.element_text(angle=90))
    + gg.labs(x=None, y=r"$g_\alpha$", title="Posterior estimates for gene value")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_44_0.png)

    <ggplot: (8771694867472)>

We see that the gene averages ("x") lie close to the middle of the estimated averages for their sgRNAs (dots), though there is regularization towards the mean of the genes $\mu_g$ (blue line with 89% HDI in the dashed line).

```python
alpha_s_summary = az.summary(m2_az, var_names="α_s", kind="stats", hdi_prob=0.89)
alpha_s_summary["sgrna"] = sgrna_to_gene_map.sgrna.to_list()
alpha_s_summary["hugo_symbol"] = sgrna_to_gene_map.hugo_symbol.to_list()

mu_g_avg = az.summary(m2_az, var_names=["μ_g"], hdi_prob=0.89)

(
    gg.ggplot(alpha_s_summary, gg.aes(x="hugo_symbol"))
    + gg.geom_hline(yintercept=0, color="black", alpha=0.3)
    + gg.geom_rect(
        xmin=-np.inf,
        xmax=np.inf,
        ymin=mu_g_avg["hdi_5.5%"].values[0],
        ymax=mu_g_avg["hdi_94.5%"].values[0],
        color="#0D72B4",
        fill=None,
        linetype="--",
        size=0.3,
    )
    + gg.geom_hline(
        yintercept=mu_g_avg["mean"].values[0], color="#0D72B4", linetype="-", size=0.8
    )
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%", color="sgrna"))
    + gg.geom_point(gg.aes(y="mean", color="sgrna"), size=2.5, alpha=0.8)
    + gg.geom_point(
        gg.aes(x="hugo_symbol", y="mean"),
        data=g_alpha_summary,
        color="black",
        shape="x",
        size=3,
    )
    + gg.scale_color_discrete(guide=None)
    + gg.theme(figure_size=(10, 5), axis_text_x=gg.element_text(angle=90))
    + gg.labs(x=None, y="mean", title="Posterior estimates for sgRNA and gene values")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_46_0.png)

    <ggplot: (8771926572632)>

```python
ppc_m2_summary = logfc_model_ppc_dataframe(
    m2_az, data.lfc, data[["hugo_symbol", "sgrna", "depmap_id"]]
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

```python
(
    gg.ggplot(ppc_m2_summary, gg.aes(x="sgrna"))
    + gg.facet_wrap("hugo_symbol", scales="free", ncol=4)
    + gg.geom_hline(
        gg.aes(yintercept="mean"),
        data=g_alpha_summary,
        color="black",
        linetype="--",
        size=0.5,
    )
    + gg.geom_boxplot(
        gg.aes(y="mean"), outlier_alpha=0, color="#0D72B4", fill="#0D72B4", alpha=0.1
    )
    + gg.theme(
        axis_text_x=gg.element_blank(),
        figure_size=(8, 10),
        subplots_adjust={"hspace": 0.4, "wspace": 0.6},
        strip_text=gg.element_text(weight="bold"),
    )
    + gg.labs(x="sgRNA", y="mean", title="Posterior predictions")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_48_0.png)

    <ggplot: (8771690408506)>

The prediction error is still quite substantial, particularly around genes that are frequently mutated suchs as *KRAS* and *TP53*.

```python
(
    gg.ggplot(ppc_m2_summary, gg.aes(x="hugo_symbol", y="error"))
    + gg.geom_hline(yintercept=0)
    + gg.geom_jitter(
        gg.aes(color="abs(error)"), width=0.35, height=0, size=0.5, alpha=0.3
    )
    + gg.scale_color_distiller(type="seq", palette="RdPu", direction=1)
    + gg.theme(axis_text_x=gg.element_text(rotation=90))
    + gg.labs(
        x=None,
        y="error (predicted - true)",
        title="Posterior prediction error",
        color="error",
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_50_0.png)

    <ggplot: (8771914135197)>

Since we modeled the sgRNA variability, the errors for each sgRNA are now normally distributed.
The plot below specifically focuses on 4 genes that had a high degree of difference in error distributions among its sgRNAs.

```python
genes_with_sgrna_error = ["ADAMTS13", "LGALS7B", "PHACTR3", "PIK3CA", "UQCRC1"]
plot_data = ppc_m2_summary[ppc_m2_summary.hugo_symbol.isin(genes_with_sgrna_error)]
(
    gg.ggplot(plot_data, gg.aes(x="hugo_symbol", y="error"))
    + gg.geom_boxplot(
        gg.aes(color="sgrna"), position=gg.position_dodge(width=0.8), outlier_alpha=0
    )
    + gg.scale_color_discrete(guide=None)
    + gg.scale_y_continuous(limits=(-1.5, 1.5))
    + gg.labs(
        x=None,
        y="error",
        title="Prediction error of genes\nwith high variability of sgRNA effects",
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_52_0.png)

    <ggplot: (8769922610466)>

And, as noted above, some genes that are frequently mutated seem to show a correlation between the error and which data points had a mutation in those genes.

```python
def plot_ppc_error_against_mutation(
    ppc_summary: pd.DataFrame, num_muts: pd.Series, genes: Optional[List[str]] = None
) -> gg.ggplot:
    ppc_summary_mutations = ppc_summary.copy()
    ppc_summary_mutations["is_mutated"] = num_muts > 0

    if not genes is None:
        ppc_summary_mutations = ppc_summary_mutations[
            ppc_summary_mutations.hugo_symbol.isin(genes)
        ].reset_index()
        ppc_summary_mutations["sgrna_idx"] = dphelp.get_indices(
            ppc_summary_mutations, "sgrna"
        )

    return (
        gg.ggplot(ppc_summary_mutations, gg.aes(x="sgrna_idx", y="error"))
        + gg.facet_wrap("hugo_symbol", scales="free")
        + gg.geom_jitter(gg.aes(color="is_mutated", alpha="is_mutated"), size=0.5)
        + gg.scale_color_manual(
            values=list(mut_pal.values()),
            guide=gg.guide_legend(override_aes={"size": 2, "alpha": 1}),
        )
        + gg.scale_alpha_manual(
            values=[0.3, 0.7],
            guide=None,
        )
        + gg.theme(
            subplots_adjust={"hspace": 0.4, "wspace": 0.6},
            strip_text=gg.element_text(weight="bold"),
        )
        + gg.labs(
            x="sgRNA ID",
            y="prediction error",
            title="Error associated with mutation status",
            color="gene is\nmutated",
            alpha="gene is\nmutated",
        )
    )


genes_with_mutation_error = ["KRAS", "MDM2", "PTK2", "TP53"]
plot_ppc_error_against_mutation(
    ppc_m2_summary, num_muts=data.n_muts, genes=genes_with_mutation_error
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_54_0.png)

    <ggplot: (8771690716612)>

---

### Model 3. Add cell line varying intercept to Model 2

There is sure to be some variability per cell line, so this should be added to the model.
This model builds on model two by introducing another varying intercept for cell line with a tight prior distribution.
To help with sampling, I also tightened the priors for the sgRNA/gene varying intercepts.

$
y \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_s + \beta_c\\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha_s}, \sigma_{\alpha_s}) \\
\qquad \mu_{\alpha_s} = g_s \\
\qquad \quad g_s \sim \mathcal{N}(\mu_g, \sigma_g) \\
\qquad \qquad \mu_g \sim \mathcal{N}(0, 2) \quad \sigma_g \sim \text{Exp}(1) \\
\qquad \sigma_{\alpha_s} \sim \text{Exp}(1) \\
\quad \beta_c \sim \mathcal{N}(\mu_{\beta_c}, \sigma_{\beta_c}) \\
\qquad \mu_{\beta_c} \sim \mathcal{N}(0, 0.2) \quad \sigma_{\beta_c} \sim \text{Exp}(0.4) \\
\sigma \sim \text{HalfNormal}(5)
$

```python
cell_line_idx = dphelp.get_indices(data, "depmap_id")
num_cell_lines = data.depmap_id.nunique()
print(f"{num_cell_lines} cell lines")
```

    258 cell lines

```python
with pm.Model() as m3:
    μ_g = pm.Normal("μ_g", 0, 2)
    σ_g = pm.Exponential("σ_g", 1)

    g_s = pm.Normal("g_s", μ_g, σ_g, shape=num_genes)

    μ_β_c = pm.Normal("μ_β_c", 0, 0.2)
    σ_β_c = pm.Exponential("σ_β_c", 0.4)
    μ_α_s = pm.Deterministic("μ_α_s", g_s[gene_idx])
    σ_α_s = pm.Exponential("σ_α_s", 1)

    β_c = pm.Normal("β_c", μ_β_c, σ_β_c, shape=num_cell_lines)
    α_s = pm.Normal("α_s", μ_α_s, σ_α_s, shape=num_sgrnas)

    μ = pm.Deterministic("μ", α_s[sgrna_idx] + β_c[cell_line_idx])
    σ = pm.HalfNormal("σ", 5)

    y = pm.Normal("y", μ, σ, observed=data.lfc)
```

```python
pm.model_to_graphviz(m3)
```

![svg](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_58_0.svg)

```python
m3_cache_dir = pymc3_cache_dir / "subset_speclet_m3"

m3_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m3,
    num_mcmc=2000,
    tune=4000,
    chains=2,
    cores=2,
    random_seed=RANDOM_SEED,
    cache_dir=pymc3_cache_dir / m3_cache_dir,
    force=False,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 40000, "target_accept": 0.9},
)
```

    Loading cached trace and posterior sample...

```python
m3_az = pmhelp.samples_to_arviz(model=m3, res=m3_sampling_results)
```

    posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

```python
az.plot_trace(m3_az, var_names=["g_s", "α_s", "β_c"], compact=True)
plt.show()
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_61_0.png)

> This model is currently not sampling properly.
> This is most likely due to non-identifiability with the two varying intercepts.
> I will revist trying to add in cell line in later models.

---

### Model 4. Add mutation covariate to Model 2

As demonstrated above, the mutation status of a gene can have a large impact on the log fold change measurement.
This model adds to model 2 by including a variable to indicate if the gene is mutated.
For now, the variable will not vary by gene, but in the subsequent model, it will.

$
y \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_s + \gamma M \\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha_s}, \sigma_{\alpha_s}) \\
\qquad \mu_{\alpha_s} = g_s \\
\qquad \quad g_s \sim \mathcal{N}(\mu_g, \sigma_g) \\
\qquad \qquad \mu_g \sim \mathcal{N}(0, 5) \quad \sigma_g \sim \text{Exp}(1) \\
\qquad \sigma_\alpha \sim \text{Exp}(1) \\
\quad \gamma \sim \mathcal{N}(0, 5) \\
\sigma \sim \text{HalfNormal}(5)
$

```python
is_mutated = data[["n_muts"]].to_numpy().flatten().astype(bool).astype(int)

with pm.Model() as m4:
    μ_g = pm.Normal("μ_g", 0, 5)
    σ_g = pm.Exponential("σ_g", 1)

    g_s = pm.Normal("g_s", μ_g, σ_g, shape=num_genes)

    μ_α_s = pm.Deterministic("μ_α_s", g_s[gene_idx])
    σ_α_s = pm.Exponential("σ_α_s", 1)

    α_s = pm.Normal("α_s", μ_α_s, σ_α_s, shape=num_sgrnas)
    γ = pm.Normal("γ", 0, 5)

    μ = pm.Deterministic("μ", α_s[sgrna_idx] + γ * is_mutated)
    σ = pm.HalfNormal("σ", 5)

    y = pm.Normal("y", μ, σ, observed=data.lfc)
```

```python
pm.model_to_graphviz(m4)
```

![svg](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_65_0.svg)

```python
m4_cache_dir = pymc3_cache_dir / "subset_speclet_m4"

m4_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m4,
    num_mcmc=2000,
    tune=4000,
    chains=2,
    cores=2,
    random_seed=RANDOM_SEED,
    cache_dir=pymc3_cache_dir / m4_cache_dir,
    force=False,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 40000},
)

m4_az = pmhelp.samples_to_arviz(model=m4, res=m4_sampling_results)
```

    Loading cached trace and posterior sample...


    posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

```python
az.plot_trace(m4_az, var_names=["μ_g", "σ_g", "g_s", "α_s", "γ"], compact=True)
plt.show()
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_67_0.png)

```python
az.summary(m4_az, var_names=["μ_g", "σ_g", "σ_α_s", "γ"])
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ_g</th>
      <td>-0.077</td>
      <td>0.050</td>
      <td>-0.169</td>
      <td>0.020</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4855.0</td>
      <td>3640.0</td>
      <td>4848.0</td>
      <td>2385.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_g</th>
      <td>0.222</td>
      <td>0.042</td>
      <td>0.148</td>
      <td>0.302</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3467.0</td>
      <td>3467.0</td>
      <td>3322.0</td>
      <td>3104.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_α_s</th>
      <td>0.228</td>
      <td>0.019</td>
      <td>0.191</td>
      <td>0.261</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4194.0</td>
      <td>4192.0</td>
      <td>4032.0</td>
      <td>2694.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>γ</th>
      <td>-0.366</td>
      <td>0.008</td>
      <td>-0.382</td>
      <td>-0.351</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3322.0</td>
      <td>3313.0</td>
      <td>3338.0</td>
      <td>3159.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
ppc_m4_summary = logfc_model_ppc_dataframe(
    m4_az, data.lfc, data[["hugo_symbol", "sgrna", "depmap_id"]]
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

```python
plot_ppc_error_against_mutation(
    ppc_m4_summary, num_muts=data.n_muts, genes=genes_with_mutation_error
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_70_0.png)

    <ggplot: (8771925678999)>

```python
def get_concatenated_summaries(
    az_objs: List[az.InferenceData],
    names: List[str],
    var_names: List[str],
    hdi_prob: float = 0.89,
) -> pd.DataFrame:

    if len(az_objs) != len(names):
        raise Exception("Unequal number of Arviz objects and names.")

    def _get_summary(az_obj, name) -> pd.DataFrame:
        d = az.summary(az_obj, var_names=var_names, hdi_prob=hdi_prob)
        d["model_name"] = name
        return d

    return pd.concat([_get_summary(a, n) for a, n in zip(az_objs, names)])
```

```python
gene_summaries = get_concatenated_summaries(
    [m2_az, m4_az], names=["M2", "M4"], var_names=["g_s"]
)
gene_summaries["hugo_symbol"] = ordered_genes * 2

pos = gg.position_dodge(width=0.75)

(
    gg.ggplot(gene_summaries, gg.aes(x="hugo_symbol", color="model_name"))
    + gg.geom_linerange(
        gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), alpha=0.5, position=pos, size=0.8
    )
    + gg.geom_point(gg.aes(y="mean"), position=pos, size=2)
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.theme(axis_text_x=gg.element_text(angle=90))
    + gg.labs(
        x=None,
        y="posterior",
        color="model",
        title="Comparison of gene level effects\nwith and without a covariate for mutation",
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_72_0.png)

    <ggplot: (8771905088799)>

```python
sgrna_summaries = get_concatenated_summaries(
    [m2_az, m4_az], names=["M2", "M4"], var_names=["α_s"]
)
sgrna_gene_df = data[["sgrna", "hugo_symbol"]].drop_duplicates().reset_index(drop=True)
sgrna_summaries = pd.merge(
    sgrna_summaries.reset_index(drop=True),
    pd.concat([sgrna_gene_df for i in range(2)]).reset_index(drop=True),
    left_index=True,
    right_index=True,
)

sgrna_summaries = sgrna_summaries[["mean", "model_name", "sgrna", "hugo_symbol"]].pivot(
    index=["sgrna", "hugo_symbol"], columns="model_name", values="mean"
)
sgrna_summaries = sgrna_summaries.reset_index(drop=False)
sgrna_summaries["difference"] = sgrna_summaries["M4"] - sgrna_summaries["M2"]
sgrna_summaries
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
      <th>model_name</th>
      <th>sgrna</th>
      <th>hugo_symbol</th>
      <th>M2</th>
      <th>M4</th>
      <th>difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ADAMTS13</td>
      <td>0.064</td>
      <td>0.094</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCTACTTCCAGCCTAAGCCA</td>
      <td>ADAMTS13</td>
      <td>0.017</td>
      <td>0.046</td>
      <td>0.029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GTACAGAGTGGCCCTCACCG</td>
      <td>ADAMTS13</td>
      <td>-0.410</td>
      <td>-0.380</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TTTGACCTGGAGTTGCCTGA</td>
      <td>ADAMTS13</td>
      <td>0.108</td>
      <td>0.138</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ATACCCAATAGAGTCCGAGG</td>
      <td>BRAF</td>
      <td>-0.327</td>
      <td>-0.279</td>
      <td>0.048</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>TGAGACTGAGAAGAATAATG</td>
      <td>UQCRC1</td>
      <td>-0.700</td>
      <td>-0.694</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>99</th>
      <td>ACTCTGTTCCTTCATCTCCG</td>
      <td>ZSWIM8</td>
      <td>-0.327</td>
      <td>-0.310</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>100</th>
      <td>AGTGCGGATGAGTTTCAGCG</td>
      <td>ZSWIM8</td>
      <td>-0.271</td>
      <td>-0.254</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>101</th>
      <td>CGATTTACCTGAAGACCACA</td>
      <td>ZSWIM8</td>
      <td>-0.073</td>
      <td>-0.056</td>
      <td>0.017</td>
    </tr>
    <tr>
      <th>102</th>
      <td>GATTTACCTGAAGACCACAG</td>
      <td>ZSWIM8</td>
      <td>-0.255</td>
      <td>-0.238</td>
      <td>0.017</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 5 columns</p>
</div>

```python
(
    gg.ggplot(sgrna_summaries, gg.aes(x="hugo_symbol", y="difference"))
    + gg.geom_point(
        gg.aes(color="sgrna"),
        position=gg.position_dodge(width=0.9),
        alpha=0.7,
        size=0.8,
    )
    + gg.scale_color_manual(values=["black"] * sgrna_summaries.shape[0], guide=None)
    + gg.theme(axis_text_x=gg.element_text(angle=90))
    + gg.labs(
        x=None, y=r"$α_{s,M4}-α_{s,M2}$", title="Differences in sgRNA posterior means"
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_74_0.png)

    <ggplot: (8771914953286)>

---

### Model 5. Modify model 4 to make the mutation covariate very with gene

$
y \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_s + \gamma_g M \\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha_s}, \sigma_{\alpha_s}) \\
\qquad \mu_{\alpha_s} = g_s \\
\qquad \quad g_s \sim \mathcal{N}(\mu_g, \sigma_g) \\
\qquad \qquad \mu_g \sim \mathcal{N}(0, 5) \quad \sigma_g \sim \text{Exp}(1) \\
\qquad \sigma_\alpha \sim \text{Exp}(1) \\
\
\quad \gamma_g \sim \mathcal{N}(\mu_{\gamma_g}\, \sigma_{\gamma_g}) \\
\qquad \mu_{\gamma_g} \sim \mathcal{N}(0, 2) \\
\qquad \sigma_{\gamma_g} \sim \text{HalfNormal}(2) \\
\
\sigma \sim \text{HalfNormal}(5)
$

```python
num_sgrnas = data["sgrna"].nunique()
num_genes = data["hugo_symbol"].nunique()

sgrna_idx = dphelp.get_indices(data, "sgrna")
gene_idx = dphelp.get_indices(data, "hugo_symbol")

sgrna_to_gene_map = data[["sgrna", "hugo_symbol"]].drop_duplicates()
sgrna_to_gene_idx = dphelp.get_indices(sgrna_to_gene_map, "hugo_symbol")

with pm.Model() as m5:
    μ_g = pm.Normal("μ_g", 0, 5)
    σ_g = pm.Exponential("σ_g", 1)

    g_s = pm.Normal("g_s", μ_g, σ_g, shape=num_genes)

    μ_α_s = pm.Deterministic("μ_α_s", g_s[sgrna_to_gene_idx])
    σ_α_s = pm.Exponential("σ_α_s", 1)
    μ_γ_g = pm.Normal("μ_γ_g", 0, 2)
    σ_γ_g = pm.HalfNormal("σ_γ_g", 2)

    α_s = pm.Normal("α_s", μ_α_s, σ_α_s, shape=num_sgrnas)
    γ_g = pm.Normal("γ_g", μ_γ_g, σ_γ_g, shape=num_genes)

    μ = pm.Deterministic("μ", α_s[sgrna_idx] + γ_g[gene_idx] * is_mutated)
    σ = pm.HalfNormal("σ", 5)

    y = pm.Normal("y", μ, σ, observed=data.lfc)
```

```python
pm.model_to_graphviz(m5)
```

![svg](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_77_0.svg)

```python
m5_cache_dir = pymc3_cache_dir / "subset_speclet_m5"

m5_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m5,
    num_mcmc=2000,
    tune=4000,
    chains=2,
    cores=2,
    random_seed=RANDOM_SEED,
    cache_dir=pymc3_cache_dir / m5_cache_dir,
    force=False,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 40000},
)

m5_az = pmhelp.samples_to_arviz(model=m5, res=m5_sampling_results)
```

    Loading cached trace and posterior sample...


    posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

```python
az.plot_trace(m5_az, var_names=["μ_g", "σ_g", "g_s", "α_s", "γ_g"], compact=True)
plt.show()
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_79_0.png)

```python
az.summary(m5_az, var_names=["μ_g", "σ_g", "σ_α_s", "μ_γ_g", "σ_γ_g", "σ"])
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
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ_g</th>
      <td>-0.059</td>
      <td>0.058</td>
      <td>-0.165</td>
      <td>0.059</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3785.0</td>
      <td>2822.0</td>
      <td>3813.0</td>
      <td>2913.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_g</th>
      <td>0.264</td>
      <td>0.047</td>
      <td>0.184</td>
      <td>0.356</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3279.0</td>
      <td>3228.0</td>
      <td>3289.0</td>
      <td>3047.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_α_s</th>
      <td>0.228</td>
      <td>0.019</td>
      <td>0.195</td>
      <td>0.263</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3254.0</td>
      <td>3205.0</td>
      <td>3336.0</td>
      <td>3289.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>μ_γ_g</th>
      <td>-0.111</td>
      <td>0.067</td>
      <td>-0.236</td>
      <td>0.019</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>6062.0</td>
      <td>4280.0</td>
      <td>6108.0</td>
      <td>2390.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_γ_g</th>
      <td>0.328</td>
      <td>0.049</td>
      <td>0.246</td>
      <td>0.421</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3186.0</td>
      <td>2984.0</td>
      <td>3427.0</td>
      <td>2857.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>0.453</td>
      <td>0.001</td>
      <td>0.450</td>
      <td>0.455</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5793.0</td>
      <td>5793.0</td>
      <td>5796.0</td>
      <td>2611.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
ppc_m5_summary = logfc_model_ppc_dataframe(
    m5_az, data.lfc, data[["hugo_symbol", "sgrna", "depmap_id", "n_muts"]]
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

```python
plot_ppc_error_against_mutation(
    ppc_m5_summary, num_muts=data.n_muts, genes=genes_with_mutation_error
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_82_0.png)

    <ggplot: (8771914532855)>

```python
m5_gamma_post = pd.DataFrame(
    m5_sampling_results["trace"]["γ_g"][:250, :], columns=ordered_genes
).melt(var_name="hugo_symbol")

(
    gg.ggplot(m5_gamma_post, gg.aes(x="value"))
    + gg.geom_vline(xintercept=0, linetype="--")
    + gg.geom_density(
        gg.aes(color="hugo_symbol", fill="hugo_symbol"), size=0.7, alpha=0.1
    )
    + gg.labs(
        x="posterior",
        y="density",
        title=r"Posterior distributions on mutation covariate $\gamma_s$",
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_83_0.png)

    <ggplot: (8771914557910)>

```python
m5_gamma_summary = az.summary(
    m5_az, var_names=["γ_g"], hdi_prob=0.89, kind="stats"
).assign(hugo_symbol=ordered_genes)

p = (
    gg.ggplot(
        m5_gamma_summary,
        gg.aes(
            y="mean", x="hugo_symbol", color="hugo_symbol", alpha="np.log(abs(mean))"
        ),
    )
    + gg.geom_hline(yintercept=0, linetype="--")
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), size=1)
    + gg.geom_point()
    + gg.scale_color_discrete(guide=None)
    + gg.scale_alpha_continuous(guide=None)
    + gg.coord_flip()
    + gg.labs(
        x=None,
        y="posterior",
        title=r"Posterior distributions on mutation covariate $\gamma_s$",
    )
)
p
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_84_0.png)

    <ggplot: (8771914919054)>

```python
m4_gamma_summary = az.summary(
    m4_az, var_names=["γ"], hdi_prob=0.89, kind="stats"
).reset_index()
m4_gamma_mean = m4_gamma_summary.loc[0, "mean"]
m4_gamma_hdi = m4_gamma_summary.iloc[0, 3:].to_numpy().astype(float)

(
    p
    + gg.geom_hline(yintercept=m4_gamma_mean, color="blue")
    + gg.geom_hline(yintercept=m4_gamma_hdi, color="blue", linetype="--", alpha=0.5)
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_85_0.png)

    <ggplot: (8771914757178)>

---

### Model 6. Introduce to model 5 a covariate multiplied against CN that varies per gene

In the original CERES model, the covariate that multiplies against copy number varies per cell line.
I think it is worth comparing the results of varying per cell line, varying per gene, and varying on both.
To focus on the magnitude of the copy number, I will $\log_2$ scale it before z-scaling.

$
y \sim \mathcal{N}(\mu, \sigma) \\
\mu = \alpha_s + \gamma_g M + \delta_g \log_2 C \\
\quad \alpha_s \sim \mathcal{N}(\mu_{\alpha_s}, \sigma_{\alpha_s}) \\
\qquad \mu_{\alpha_s} = g_s \\
\qquad \quad g_s \sim \mathcal{N}(\mu_g, \sigma_g) \\
\qquad \qquad \mu_g \sim \mathcal{N}(0, 5) \quad \sigma_g \sim \text{Exp}(1) \\
\qquad \sigma_\alpha \sim \text{Exp}(1) \\
\
\quad \gamma_g \sim \mathcal{N}(\mu_{\gamma_g}\, \sigma_{\gamma_g}) \\
\qquad \mu_{\gamma_g} \sim \mathcal{N}(0, 2) \\
\qquad \sigma_{\gamma_g} \sim \text{HalfNormal}(2) \\
\
\quad \delta_g \sim \mathcal{N}(\mu_{\delta_g}\, \sigma_{\delta_g}) \\
\qquad \mu_{\delta_g} \sim \mathcal{N}(-0.2, 2) \\
\qquad \sigma_{\delta_g} \sim \text{HalfNormal}(2) \\
\
\sigma \sim \text{HalfNormal}(5)
$

```python
data["log2_cn"] = np.log2(data["gene_cn"].to_numpy() + 1)
scaled_log2_cn = dphelp.zscale_cna_by_group(
    data, gene_cn_col="log2_cn", new_col="z_log2_cn", cn_max=np.log2(10)
).z_log2_cn.to_numpy()
data["scaled_log2_cn"] = scaled_log2_cn
```

```python
with pm.Model() as m6:
    # Indices
    sgrna_to_gene_idx_shared = pm.Data("sgrna_to_gene_idx_shared", sgrna_to_gene_idx)
    sgrna_idx_shared = pm.Data("sgrna_idx_shared", sgrna_idx)
    gene_idx_shared = pm.Data("gene_idx_shared", gene_idx)

    # Data
    is_mutated_shared = pm.Data("is_mutated_shared", is_mutated)
    scaled_log2_cn_shared = pm.Data("scaled_log2_cn_shared", scaled_log2_cn)
    lfc_shared = pm.Data("lfc_shared", data.lfc.to_numpy().flatten())

    μ_g = pm.Normal("μ_g", 0, 5)
    σ_g = pm.Exponential("σ_g", 1)

    g_s = pm.Normal("g_s", μ_g, σ_g, shape=num_genes)

    μ_α_s = pm.Deterministic("μ_α_s", g_s[sgrna_to_gene_idx_shared])
    σ_α_s = pm.Exponential("σ_α_s", 1)
    μ_γ_g = pm.Normal("μ_γ_g", 0, 2)
    σ_γ_g = pm.HalfNormal("σ_γ_g", 2)
    μ_δ_g = pm.Normal("μ_δ_g", 0, 2)
    σ_δ_g = pm.HalfNormal("σ_δ_g", 2)

    α_s = pm.Normal("α_s", μ_α_s, σ_α_s, shape=num_sgrnas)
    γ_g = pm.Normal("γ_g", μ_γ_g, σ_γ_g, shape=num_genes)
    δ_g = pm.Normal("δ_g", μ_δ_g, σ_δ_g, shape=num_genes)

    μ = pm.Deterministic(
        "μ",
        α_s[sgrna_idx_shared]
        + γ_g[gene_idx_shared] * is_mutated_shared
        + δ_g[gene_idx_shared] * scaled_log2_cn_shared,
    )
    σ = pm.HalfNormal("σ", 5)

    y = pm.Normal("y", μ, σ, observed=lfc_shared)
```

```python
pm.model_to_graphviz(m6)
```

![svg](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_89_0.svg)

```python
m6_cache_dir = pymc3_cache_dir / "subset_speclet_m6"

m6_sampling_results = pmhelp.pymc3_sampling_procedure(
    model=m6,
    num_mcmc=2000,
    tune=4000,
    chains=2,
    cores=2,
    random_seed=RANDOM_SEED,
    cache_dir=pymc3_cache_dir / m6_cache_dir,
    force=False,
    sample_kwargs={"init": "advi+adapt_diag", "n_init": 40000},
)

m6_az = pmhelp.samples_to_arviz(model=m6, res=m6_sampling_results)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using advi+adapt_diag...

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='30668' class='' max='40000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  76.67% [30668/40000 01:23<00:25 Average Loss = 37,207]
</div>

    Convergence achieved at 30700
    Interrupted at 30,699 [76%]: Average Loss = 53,122
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [σ, δ_g, γ_g, α_s, σ_δ_g, μ_δ_g, σ_γ_g, μ_γ_g, σ_α_s, g_s, σ_g, μ_g]

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 02:54<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 4_000 tune and 2_000 draw iterations (8_000 + 4_000 draws total) took 177 seconds.

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:11<00:00]
</div>

    Caching trace and posterior sample...


    posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

```python
az.plot_trace(m6_az, var_names=["μ_g", "σ_g", "g_s", "α_s", "γ_g", "δ_g"], compact=True)
plt.show()
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_91_0.png)

```python
az.summary(
    m6_az,
    var_names=["μ_g", "σ_g", "σ_α_s", "μ_γ_g", "σ_γ_g", "μ_δ_g", "σ_δ_g", "σ"],
    hdi_prob=0.89,
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
      <th>mean</th>
      <th>sd</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ_g</th>
      <td>-0.060</td>
      <td>0.059</td>
      <td>-0.154</td>
      <td>0.034</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4127.0</td>
      <td>3176.0</td>
      <td>4084.0</td>
      <td>2743.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_g</th>
      <td>0.267</td>
      <td>0.045</td>
      <td>0.196</td>
      <td>0.333</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3078.0</td>
      <td>3078.0</td>
      <td>3022.0</td>
      <td>2955.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_α_s</th>
      <td>0.228</td>
      <td>0.019</td>
      <td>0.199</td>
      <td>0.258</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>2980.0</td>
      <td>2955.0</td>
      <td>3060.0</td>
      <td>2820.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>μ_γ_g</th>
      <td>-0.104</td>
      <td>0.063</td>
      <td>-0.205</td>
      <td>-0.006</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3604.0</td>
      <td>3222.0</td>
      <td>3670.0</td>
      <td>2337.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_γ_g</th>
      <td>0.313</td>
      <td>0.047</td>
      <td>0.236</td>
      <td>0.381</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3485.0</td>
      <td>3216.0</td>
      <td>3770.0</td>
      <td>2790.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>μ_δ_g</th>
      <td>-0.050</td>
      <td>0.012</td>
      <td>-0.070</td>
      <td>-0.032</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>5759.0</td>
      <td>5349.0</td>
      <td>5849.0</td>
      <td>2368.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ_δ_g</th>
      <td>0.059</td>
      <td>0.009</td>
      <td>0.046</td>
      <td>0.073</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4627.0</td>
      <td>4212.0</td>
      <td>5036.0</td>
      <td>2552.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>0.447</td>
      <td>0.001</td>
      <td>0.445</td>
      <td>0.449</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>6531.0</td>
      <td>6530.0</td>
      <td>6533.0</td>
      <td>2549.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
m6_delta_g = az.summary(m6_az, var_names=["δ_g"], hdi_prob=0.89, kind="stats")
m6_delta_g["hugo_symbol"] = ordered_genes

(
    gg.ggplot(m6_delta_g, gg.aes(x="hugo_symbol", y="mean"))
    + gg.geom_hline(yintercept=0, alpha=0.5)
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"))
    + gg.geom_point()
    + gg.theme(axis_text_x=gg.element_text(angle=90))
    + gg.labs(x=None, y="posterior mean", title="Posteriors of CN covariate")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_93_0.png)

    <ggplot: (8771694068387)>

```python
ppc_m6_summary = logfc_model_ppc_dataframe(
    m6_az,
    data.lfc,
    data[["hugo_symbol", "sgrna", "depmap_id", "n_muts", "scaled_log2_cn"]],
)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

```python
new_pred_data = (
    data.copy()
    .groupby("hugo_symbol")
    .agg({"scaled_log2_cn": ["min", "max"]})
    .reset_index()
)

new_pred_data = (
    pd.DataFrame(
        np.linspace(
            new_pred_data["scaled_log2_cn"]["min"],
            new_pred_data["scaled_log2_cn"]["max"],
            10,
        ),
        columns=new_pred_data.hugo_symbol,
    )
    .melt(var_name="hugo_symbol", value_name="scaled_log2_cn")
    .merge(sgrna_gene_df, left_index=False, right_index=False, on="hugo_symbol")
)

new_pred_data = dphelp.make_cat(new_pred_data, "hugo_symbol")
new_pred_data = pd.concat(
    [new_pred_data.assign(is_mutated=0), new_pred_data.assign(is_mutated=1)]
)

new_pred_data
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
      <th>hugo_symbol</th>
      <th>scaled_log2_cn</th>
      <th>sgrna</th>
      <th>is_mutated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>CCTACTTCCAGCCTAAGCCA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>GTACAGAGTGGCCCTCACCG</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>TTTGACCTGGAGTTGCCTGA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ADAMTS13</td>
      <td>-1.815216</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>ZSWIM8</td>
      <td>4.293112</td>
      <td>GATTTACCTGAAGACCACAG</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1026</th>
      <td>ZSWIM8</td>
      <td>5.167118</td>
      <td>ACTCTGTTCCTTCATCTCCG</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>ZSWIM8</td>
      <td>5.167118</td>
      <td>AGTGCGGATGAGTTTCAGCG</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1028</th>
      <td>ZSWIM8</td>
      <td>5.167118</td>
      <td>CGATTTACCTGAAGACCACA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>ZSWIM8</td>
      <td>5.167118</td>
      <td>GATTTACCTGAAGACCACAG</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2060 rows × 4 columns</p>
</div>

```python
with m6:
    # Set data to predict on.
    pm.set_data(
        {
            "sgrna_idx_shared": dphelp.get_indices(new_pred_data, "sgrna"),
            "gene_idx_shared": dphelp.get_indices(new_pred_data, "hugo_symbol"),
            "is_mutated_shared": new_pred_data.is_mutated.to_numpy().flatten(),
            "scaled_log2_cn_shared": new_pred_data.scaled_log2_cn.to_numpy().flatten(),
        }
    )
    m6_new_data_ppc = pm.sample_posterior_predictive(m6_sampling_results["trace"], 2000)
```

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:19<00:00]
</div>

```python
new_pred_data["ppc_mean"] = np.mean(m6_new_data_ppc["y"], axis=0)
new_pred_data["group"] = [
    f"{a}_{b}" for a, b in zip(new_pred_data.is_mutated, new_pred_data.sgrna)
]
new_pred_data = pd.merge(
    new_pred_data,
    pd.DataFrame(
        az.hdi(m6_new_data_ppc["y"], hdi_prob=0.89),
        columns=calculate_hdi_labels(hdi_prob=0.89),
    ),
    left_index=True,
    right_index=True,
)
new_pred_data.head()
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:493: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

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
      <th>hugo_symbol</th>
      <th>scaled_log2_cn</th>
      <th>sgrna</th>
      <th>is_mutated</th>
      <th>ppc_mean</th>
      <th>group</th>
      <th>hdi_5.5%</th>
      <th>hdi_94.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>0</td>
      <td>0.159735</td>
      <td>0_CCACCCACAGACGCTCAGCA</td>
      <td>-0.577399</td>
      <td>0.823857</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>1</td>
      <td>0.164041</td>
      <td>1_CCACCCACAGACGCTCAGCA</td>
      <td>-0.577399</td>
      <td>0.823857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>CCTACTTCCAGCCTAAGCCA</td>
      <td>0</td>
      <td>0.131140</td>
      <td>0_CCTACTTCCAGCCTAAGCCA</td>
      <td>-0.535059</td>
      <td>0.863401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>CCTACTTCCAGCCTAAGCCA</td>
      <td>1</td>
      <td>0.120027</td>
      <td>1_CCTACTTCCAGCCTAAGCCA</td>
      <td>-0.535059</td>
      <td>0.863401</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADAMTS13</td>
      <td>-2.820652</td>
      <td>GTACAGAGTGGCCCTCACCG</td>
      <td>0</td>
      <td>-0.325785</td>
      <td>0_GTACAGAGTGGCCCTCACCG</td>
      <td>-0.971105</td>
      <td>0.442466</td>
    </tr>
  </tbody>
</table>
</div>

```python
data.head()
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
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>gene_cn_z</th>
      <th>log2_cn</th>
      <th>z_log2_cn</th>
      <th>scaled_log2_cn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.029491</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.632215</td>
      <td>1.861144</td>
      <td>1.692178</td>
      <td>1.692178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.426017</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>G12D</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.632215</td>
      <td>1.861144</td>
      <td>1.692178</td>
      <td>1.692178</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>0.008626</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
      <td>1.375470</td>
      <td>-0.338293</td>
      <td>-0.338293</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.280821</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
      <td>1.375470</td>
      <td>-0.338293</td>
      <td>-0.338293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.239815</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>-0.365193</td>
      <td>1.375470</td>
      <td>-0.338293</td>
      <td>-0.338293</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>

```python
(
    gg.ggplot(new_pred_data, gg.aes(x="scaled_log2_cn", y="ppc_mean", group="group"))
    + gg.facet_wrap("hugo_symbol", scales="free", ncol=3)
    + gg.geom_point(
        gg.aes(x="scaled_log2_cn", y="lfc", color="n_muts>0"),
        data=data,
        inherit_aes=False,
        size=0.1,
        alpha=0.1,
    )
    + gg.geom_hline(yintercept=0, color="black", alpha=0.3)
    + gg.geom_vline(xintercept=0, color="black", alpha=0.3)
    + gg.geom_line(gg.aes(color="factor(is_mutated)"), size=1, alpha=0.8)
    + gg.scale_color_manual(values=list(mut_pal.values()), labels=["WT", "mut."])
    + gg.theme(
        figure_size=(8, 20),
        subplots_adjust={"hspace": 0.4, "wspace": 0.6},
        legend_title=gg.element_blank(),
    )
    + gg.labs(
        x="copy number (log2, z-scaled)",
        y="posterior prediction",
        title="M6 posterior predictions over CN",
    )
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_99_0.png)

    <ggplot: (8769919732902)>

TODO:

- compare gene effects between M1, M2, M4, M5, M6

```python
def get_gene_summary_df(
    az_obj: az.InferenceData,
    var_names: List[str],
    model_name,
    ordered_genes: List[str] = ordered_genes,
) -> pd.DataFrame:
    d = az.summary(az_obj, var_names=var_names, kind="stats", hdi_prob=0.89)
    d["hugo_symbol"] = ordered_genes
    d["model"] = model_name
    return d


zipped_data = zip(
    [m1_az, m2_az, m4_az, m5_az, m6_az],
    [["α_g"], ["g_s"], ["g_s"], ["g_s"], ["g_s"]],
    [f"M{i}" for i in [1, 2, 4, 5, 6]],
)

models_gene_posts = pd.concat([get_gene_summary_df(a, v, m) for a, v, m in zipped_data])

(
    gg.ggplot(models_gene_posts, gg.aes(x="model", y="mean"))
    + gg.facet_wrap("hugo_symbol", ncol=4, scales="fixed")
    + gg.geom_hline(yintercept=0, alpha=0.3)
    + gg.geom_line(gg.aes(group="hugo_symbol"), alpha=0.7, size=1, color="#429DD6")
    + gg.geom_linerange(gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"))
    + gg.geom_point()
    + gg.theme(figure_size=(8, 10))
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_101_0.png)

    <ggplot: (8771665915257)>

---

### Model 7. Introduce a covariate for the *KRAS* allele to model 6

Finally, we can add a categorical variable for the *KRAS* allele of a tumor sample.

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
model_comparisons = az.compare(
    {"M1": m1_az, "M2": m2_az, "M4": m4_az, "M5": m5_az, "M6": m6_az}, seed=RANDOM_SEED
)
model_comparisons
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
      <th>rank</th>
      <th>loo</th>
      <th>p_loo</th>
      <th>d_loo</th>
      <th>weight</th>
      <th>se</th>
      <th>dse</th>
      <th>warning</th>
      <th>loo_scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M6</th>
      <td>0</td>
      <td>-36879.364337</td>
      <td>166.504663</td>
      <td>0.000000</td>
      <td>8.436560e-01</td>
      <td>341.739937</td>
      <td>0.000000</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>M5</th>
      <td>1</td>
      <td>-37682.337418</td>
      <td>135.296161</td>
      <td>802.973081</td>
      <td>4.687433e-09</td>
      <td>353.702792</td>
      <td>82.083983</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>M4</th>
      <td>2</td>
      <td>-39321.566570</td>
      <td>109.669236</td>
      <td>2442.202233</td>
      <td>0.000000e+00</td>
      <td>368.749556</td>
      <td>120.478517</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>M2</th>
      <td>3</td>
      <td>-40278.638362</td>
      <td>107.558885</td>
      <td>3399.274025</td>
      <td>1.265856e-01</td>
      <td>387.887391</td>
      <td>153.616443</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>M1</th>
      <td>4</td>
      <td>-44901.556807</td>
      <td>29.735296</td>
      <td>8022.192470</td>
      <td>2.975843e-02</td>
      <td>356.456544</td>
      <td>166.261557</td>
      <td>False</td>
      <td>log</td>
    </tr>
  </tbody>
</table>
</div>

```python
plot_data = (
    model_comparisons.reset_index(drop=False)
    .assign(loo_low=lambda d: d.loo - d.se)
    .assign(loo_high=lambda d: d.loo + d.se)
)
(
    gg.ggplot(plot_data, gg.aes(x="index"))
    + gg.geom_linerange(gg.aes(ymin="loo_low", ymax="loo_high"))
    + gg.geom_point(gg.aes(y="loo", size="weight"))
    + gg.labs(x="model", y="LOO", size="weight", title="Model comparison")
)
```

![png](010_013_hierarchical-model-subsample_files/010_013_hierarchical-model-subsample_115_0.png)

    <ggplot: (8769918478626)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```
