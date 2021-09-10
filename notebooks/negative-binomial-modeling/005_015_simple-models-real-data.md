# Exploration of negative binomial models on real data

```python
%load_ext autoreload
%autoreload 2
```

```python
import re
import string
import warnings
from pathlib import Path
from time import time
from typing import Sequence, Union

import arviz as az
import janitor
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from theano import tensor as tt
```

```python
from src.analysis import pymc3_analysis as pmanal
from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.data_processing import vectors as vhelp
from src.globals import PYMC3
from src.io import cache_io, data_io
from src.loggers import logger
from src.modeling import pymc3_sampling_api as pmapi
from src.plot.color_pal import FitMethodColors, ModelColors, SeabornColor
```

```python
notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(
    gg.theme_classic()
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

For this analysis, I used the subsample of CRC data.

```python
crc_subsample_file = data_io.DataFile.crc_subsample
crc_modeling_data_subsample = pd.read_csv(data_io.data_path(crc_subsample_file))
crc_modeling_data_subsample.head()
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
      <td>AACAACTAACTTTGTACAT</td>
      <td>LS513_c903R1</td>
      <td>0.226828</td>
      <td>ERS717283.plasmid</td>
      <td>chr8_81679130_-</td>
      <td>IMPA1</td>
      <td>sanger</td>
      <td>True</td>
      <td>8</td>
      <td>81679130</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.957015</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AACAAGATGTTTTGCCAAC</td>
      <td>LS513_c903R1</td>
      <td>0.245798</td>
      <td>ERS717283.plasmid</td>
      <td>chr17_7675205_-</td>
      <td>TP53</td>
      <td>sanger</td>
      <td>True</td>
      <td>17</td>
      <td>7675205</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.970098</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AACAGCTCGTTGTACCGCT</td>
      <td>LS513_c903R1</td>
      <td>-2.819159</td>
      <td>ERS717283.plasmid</td>
      <td>chr3_41225481_+</td>
      <td>CTNNB1</td>
      <td>sanger</td>
      <td>True</td>
      <td>3</td>
      <td>41225481</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.964463</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AACGCATACCTTGAGCAAG</td>
      <td>LS513_c903R1</td>
      <td>-1.188274</td>
      <td>ERS717283.plasmid</td>
      <td>chr3_71046949_+</td>
      <td>FOXP1</td>
      <td>sanger</td>
      <td>True</td>
      <td>3</td>
      <td>71046949</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.949038</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAGTTCCTCTGAAGTTCGCA</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.182191</td>
      <td>2</td>
      <td>chr2_241704672_-</td>
      <td>ING5</td>
      <td>broad</td>
      <td>True</td>
      <td>2</td>
      <td>241704672</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.973700</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>

```python
f"number of rows: {crc_modeling_data_subsample.shape[0]}"
```

    'number of rows: 1443'

```python
crc_modeling_data_subsample.hugo_symbol.unique()
```

    array(['IMPA1', 'TP53', 'CTNNB1', 'FOXP1', 'ING5', 'NRAS', 'KRAS',
           'SEC23B', 'SOSTDC1', 'HSBP1L1', 'TMEM192', 'VCL', 'PLCD4',
           'SEC14L5', 'NOSTRIN', 'OTOF', 'APC', 'MDM2', 'CSDC2', 'MDM4',
           'INPP5A', 'FBXW7', 'KDELC1', 'TMPRSS3', 'FCN1', 'ADPRHL1',
           'CDK5RAP1', 'BRAF', 'LAPTM4B', 'RFWD3', 'KLF5', 'PTK2', 'DPY19L1',
           'RPL18A', 'SOWAHC', 'FAM92A', 'S100A7A', 'FUT7', 'PAFAH1B3',
           'DARS2', 'PLIN2', 'EEF1AKMT4', 'STK11', 'GATA6', 'SNX33',
           'EIF2AK1', 'TBX19', 'POU4F3', 'YY1', 'RPS26', 'CYTL1', 'ACVR1C',
           'SQLE', 'CCR3', 'NCDN', 'PLK5', 'POFUT2', 'SLC27A2', 'TXNDC17',
           'CCR9', 'GRIN2D', 'RTL3', 'PIK3CA', 'TMEM241'], dtype=object)

Unfortunately, most of the data points are missing read count data.
For now I will just drop these value.

```python
# Percent of data missing read counts.
crc_modeling_data_subsample.read_counts.isna().mean()
```

    0.604989604989605

```python
data = (
    crc_modeling_data_subsample[~crc_modeling_data_subsample.read_counts.isna()]
    .copy()
    .reset_index(drop=True)
)

# Reset categorical data categories.
achelp.set_achilles_categorical_columns(data, sort_cats=True)

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
      <td>AAGTTCCTCTGAAGTTCGCA</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.182191</td>
      <td>2</td>
      <td>chr2_241704672_-</td>
      <td>ING5</td>
      <td>broad</td>
      <td>True</td>
      <td>2</td>
      <td>241704672</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.973700</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AATGACTGAGTACAAACTGG</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.026600</td>
      <td>2</td>
      <td>chr1_114716144_-</td>
      <td>NRAS</td>
      <td>broad</td>
      <td>True</td>
      <td>1</td>
      <td>114716144</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.958285</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AATTACTACTTGCTTCCTGT</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-1.668700</td>
      <td>2</td>
      <td>chr12_25227402_+</td>
      <td>KRAS</td>
      <td>broad</td>
      <td>True</td>
      <td>12</td>
      <td>25227402</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>0.963609</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AGAAGTTTGGAGAGAGAACG</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.337427</td>
      <td>2</td>
      <td>chr5_112838158_+</td>
      <td>APC</td>
      <td>broad</td>
      <td>True</td>
      <td>5</td>
      <td>112838158</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.262371</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AGACACTTATACTATGAAAG</td>
      <td>LS513-311Cas9_RepA_p6_batch2</td>
      <td>-0.819049</td>
      <td>2</td>
      <td>chr12_68813623_+</td>
      <td>MDM2</td>
      <td>broad</td>
      <td>True</td>
      <td>12</td>
      <td>68813623</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.963609</td>
      <td>colorectal</td>
      <td>primary</td>
      <td>True</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>

The following table shows the number of sgRNA for each gene.

```python
data[["hugo_symbol", "sgrna"]].drop_duplicates().groupby(
    "hugo_symbol"
).count().sort_values("sgrna")
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
    <tr>
      <th>hugo_symbol</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACVR1C</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ING5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>INPP5A</th>
      <td>1</td>
    </tr>
    <tr>
      <th>LAPTM4B</th>
      <td>1</td>
    </tr>
    <tr>
      <th>PAFAH1B3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>PLCD4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>PLIN2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>PLK5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>POU4F3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>PTK2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>RTL3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SEC14L5</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SNX33</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SOSTDC1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SOWAHC</th>
      <td>1</td>
    </tr>
    <tr>
      <th>TMPRSS3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>TP53</th>
      <td>1</td>
    </tr>
    <tr>
      <th>TXNDC17</th>
      <td>1</td>
    </tr>
    <tr>
      <th>IMPA1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>HSBP1L1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>NCDN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>GRIN2D</th>
      <td>1</td>
    </tr>
    <tr>
      <th>FCN1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>APC</th>
      <td>1</td>
    </tr>
    <tr>
      <th>CYTL1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>FBXW7</th>
      <td>1</td>
    </tr>
    <tr>
      <th>DPY19L1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>EEF1AKMT4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>CCR9</th>
      <td>1</td>
    </tr>
    <tr>
      <th>FAM92A</th>
      <td>1</td>
    </tr>
    <tr>
      <th>GATA6</th>
      <td>2</td>
    </tr>
    <tr>
      <th>SEC23B</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CDK5RAP1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>SQLE</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CSDC2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>STK11</th>
      <td>2</td>
    </tr>
    <tr>
      <th>TBX19</th>
      <td>2</td>
    </tr>
    <tr>
      <th>TMEM192</th>
      <td>2</td>
    </tr>
    <tr>
      <th>TMEM241</th>
      <td>2</td>
    </tr>
    <tr>
      <th>BRAF</th>
      <td>2</td>
    </tr>
    <tr>
      <th>ADPRHL1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CCR3</th>
      <td>2</td>
    </tr>
    <tr>
      <th>S100A7A</th>
      <td>2</td>
    </tr>
    <tr>
      <th>RFWD3</th>
      <td>2</td>
    </tr>
    <tr>
      <th>DARS2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>EIF2AK1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>PIK3CA</th>
      <td>2</td>
    </tr>
    <tr>
      <th>NRAS</th>
      <td>2</td>
    </tr>
    <tr>
      <th>NOSTRIN</th>
      <td>2</td>
    </tr>
    <tr>
      <th>VCL</th>
      <td>2</td>
    </tr>
    <tr>
      <th>KRAS</th>
      <td>2</td>
    </tr>
    <tr>
      <th>KLF5</th>
      <td>2</td>
    </tr>
    <tr>
      <th>KDELC1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>FUT7</th>
      <td>2</td>
    </tr>
    <tr>
      <th>CTNNB1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>RPS26</th>
      <td>3</td>
    </tr>
    <tr>
      <th>RPL18A</th>
      <td>3</td>
    </tr>
    <tr>
      <th>POFUT2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>MDM2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>YY1</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

The next few plots show the distributino of LFC and read count values.

```python
sns.displot(data=data, x="lfc");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_15_0.png)

```python
sns.displot(data=data, x="read_counts", kind="hist");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_16_0.png)

```python
data.read_counts.agg(["mean", "var"])
```

    mean       554.850877
    var     313659.209709
    Name: read_counts, dtype: float64

```python
data["initial_reads"] = data.read_counts / (2 ** data.lfc)
```

```python
sns.displot(data, x="initial_reads", kind="hist");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_19_0.png)

```python
sns.jointplot(data=data, x="initial_reads", y="read_counts");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_20_0.png)

The following plot shows the distribution of reads with the *KRAS* data points highlighted for reference.

```python
plot_data = (
    data.copy().assign(is_kras=lambda d: d.hugo_symbol == "KRAS").sort_values("is_kras")
)

(
    gg.ggplot(plot_data, gg.aes(x="initial_reads", y="read_counts"))
    + gg.geom_point(gg.aes(color="is_kras", alpha="is_kras"), size=0.7)
    + gg.geom_abline(slope=1, intercept=0)
    + gg.scale_x_sqrt(expand=(0.02, 0, 0.02, 0))
    + gg.scale_y_sqrt(expand=(0.02, 0, 0.02, 0))
    + gg.scale_color_manual(values={True: "red", False: "k"})
    + gg.scale_alpha_manual(values={True: 0.8, False: 0.3})
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_22_0.png)

    <ggplot: (359307508)>

```python
(
    gg.ggplot(data, gg.aes(x="hugo_symbol", y="lfc"))
    + gg.geom_boxplot(gg.aes(color="hugo_symbol"))
    + gg.theme(
        axis_text_x=gg.element_text(angle=90, size=6),
        legend_position="none",
        figure_size=(10, 4),
    )
    + gg.labs(x="gene", y="log-fold change")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_23_0.png)

    <ggplot: (359307984)>

## Model 0. Single gene

For simplicity and to help me learn how to interpret coefficient values, the first model was just for the read counts of *KRAS*.
There are only 12 data points.

```python
data_0 = data[data.hugo_symbol == "KRAS"].reset_index()
f"Number of data points: {data_0.shape[0]}."
```

    'Number of data points: 12.'

Model `nb_m0`:

$$
\begin{gather}
\beta \sim N(0, 2.5) \\
\eta = \beta \\
\mu = \exp(\eta) \\
y \sim \text{NB}(\mu X_\text{initial}, \eta)
\end{gather}
$$

```python
with pm.Model() as nb_m0:
    initial_reads = pm.Data("initial_reads", data_0.initial_reads.values)
    final_reads = pm.Data("final_reads", data_0.read_counts.values)

    β = pm.Normal("β", 0, 2.5)
    η = β
    μ = pm.math.exp(η)
    α = pm.HalfNormal("α", 5)
    y = pm.NegativeBinomial("y", μ * initial_reads, α, observed=final_reads)
```

```python
pm.model_to_graphviz(nb_m0)
```

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_28_0.svg)

```python
with nb_m0:
    m0_trace = pm.sample(1000, chains=4, random_seed=820, return_inferencedata=True)
    m0_trace.extend(
        az.from_pymc3(posterior_predictive=pm.sample_posterior_predictive(m0_trace))
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, β]

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
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:11<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 28 seconds.

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
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 00:57<00:00]
</div>

```python
az.plot_trace(m0_trace, var_names=["β", "α"], compact=False);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_30_0.png)

```python
az.summary(m0_trace, var_names=["β", "α"])
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
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>β</th>
      <td>-0.539</td>
      <td>0.162</td>
      <td>-0.828</td>
      <td>-0.208</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>2696.0</td>
      <td>2024.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>α</th>
      <td>3.882</td>
      <td>1.413</td>
      <td>1.551</td>
      <td>6.580</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>2269.0</td>
      <td>2151.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
m0_ppc = m0_trace["posterior_predictive"]["y"].squeeze().values
(
    gg.ggplot(pd.DataFrame(m0_ppc[:1000, :]).pivot_longer(), gg.aes(x="value"))
    + gg.geom_density(gg.aes(group="variable"), color="k", size=0.3, linetype="-")
    + gg.geom_rug(
        gg.aes(x="read_counts"), data=data_0, color="blue", size=1.2, alpha=0.75
    )
    + gg.scale_x_sqrt(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.labs(x="read counts", y="density", title="Posterior predictive check")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_32_0.png)

    <ggplot: (365338609)>

```python
m0_ppc_df = pmanal.summarize_posterior_predictions(
    m0_ppc, merge_with=data_0, observed_y="read_counts"
)
(
    gg.ggplot(m0_ppc_df, gg.aes(x="read_counts", y="pred_mean"))
    + gg.geom_linerange(gg.aes(ymin="pred_hdi_low", ymax="pred_hdi_high"), alpha=0.2)
    + gg.geom_point()
    + gg.geom_abline(slope=1, intercept=0, linetype="--")
    + gg.labs(x="observed counts", y="posterior predicted counts (89% CI)")
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_33_1.png)

    <ggplot: (363828899)>

Below is the posterior point estimate for $\beta$ and the exponentiated value.

```python
beta_post = m0_trace["posterior"]["β"].values.mean()
beta_post, np.exp(beta_post)
```

    (-0.5389936736704781, 0.5833349824552863)

It is the same as the ratio of final to initial reads.

```python
(data_0.read_counts / data_0.initial_reads).mean()
```

    0.5752345646683334

### With full dataset

The model structure is the same as Model 0, except now the entire dataset is used instead of restricting to one gene.

```python
with pm.Model() as nb_m0_full:
    initial_reads = pm.Data("initial_reads", data.initial_reads.values)
    final_reads = pm.Data("final_reads", data.read_counts.values)

    β = pm.Normal("β", 0, 2.5)
    η = β
    μ = pm.math.exp(η)
    α = pm.HalfNormal("α", 5)
    y = pm.NegativeBinomial("y", μ * initial_reads, α, observed=final_reads)

    m0_full_trace = pm.sample(
        1000, tune=2000, chains=4, random_seed=922, return_inferencedata=True
    )
    m0_full_trace.extend(
        az.from_pymc3(
            posterior_predictive=pm.sample_posterior_predictive(
                m0_full_trace, random_seed=923
            )
        )
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, β]

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
  100.00% [12000/12000 00:15<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 2_000 tune and 1_000 draw iterations (8_000 + 4_000 draws total) took 30 seconds.

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
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 00:58<00:00]
</div>

```python
trace_collection: dict[str, az.InferenceData] = {"(m0) single β": m0_full_trace}
```

```python
az.plot_trace(m0_full_trace, compact=False);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_41_0.png)

```python
az.summary(m0_full_trace, var_names=["β", "α"])
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
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>β</th>
      <td>-0.078</td>
      <td>0.018</td>
      <td>-0.112</td>
      <td>-0.046</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3451.0</td>
      <td>2733.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>α</th>
      <td>5.567</td>
      <td>0.324</td>
      <td>4.981</td>
      <td>6.183</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>3188.0</td>
      <td>2712.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
m0_full_ppc_df = pmanal.summarize_posterior_predictions(
    m0_full_trace["posterior_predictive"]["y"].squeeze().values,
    merge_with=data,
    observed_y="read_counts",
)

(
    gg.ggplot(m0_full_ppc_df, gg.aes(x="read_counts", y="pred_mean"))
    + gg.geom_linerange(gg.aes(ymin="pred_hdi_low", ymax="pred_hdi_high"), alpha=0.2)
    + gg.geom_point(size=0.5, alpha=0.5)
    + gg.geom_abline(slope=1, intercept=0, linetype="--")
    + gg.scale_x_log10(expand=(0, 0, 0.02, 0))
    + gg.scale_y_log10(expand=(0, 0, 0.02, 0))
    + gg.labs(x="observed counts", y="posterior predicted counts (89% CI)")
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_43_1.png)

    <ggplot: (363824086)>

```python
beta = m0_full_trace["posterior"]["β"].values.mean()
beta, np.exp(beta)
```

    (-0.07774960581655685, 0.9251960612546195)

```python
np.mean(data.read_counts / data.initial_reads)
```

    0.9247845993336123

## Model 1. Varying per gene

The first model is simple with a varying intercept per gene.
The starting read count is treated as an exposure value and multiplied directly against $\mu$.

$$
\begin{gather}
\mu_\beta \sim N(0, 5) \\
\sigma_\beta \sim HN(5) \\
\beta_g \sim_g N(\mu_\beta, \sigma_\beta) \\
\eta = \beta_g[\text{gene}] \\
\mu = \exp(\eta) \\
\alpha = HN(5) \\
y \sim \text{NB}(\mu X_\text{initial}, \alpha)
\end{gather}
$$

```python
n_genes = len(data.hugo_symbol.unique())
gene_idx = dphelp.get_indices(data, "hugo_symbol")

with pm.Model() as nb_m1:
    g = pm.Data("gene_idx", gene_idx)
    initial_reads = pm.Data("initial_reads", data.initial_reads)
    final_reads = pm.Data("final_reads", data.read_counts)

    μ_β = pm.Normal("μ_β", 0, 2.5)
    σ_β = pm.HalfNormal("σ_β", 2.5)
    β_g = pm.Normal("β_g", μ_β, σ_β, shape=n_genes)
    η = pm.Deterministic("η", β_g[g])
    μ = pm.Deterministic("μ", pm.math.exp(η))
    α = pm.HalfNormal("α", 5)
    y = pm.NegativeBinomial("y", μ * initial_reads, α, observed=final_reads)
```

```python
pm.model_to_graphviz(nb_m1)
```

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_48_0.svg)

```python
with nb_m1:
    m1_trace = pm.sample(1000, tune=2000, random_seed=1022, return_inferencedata=True)
    m1_trace.extend(
        az.from_pymc3(
            posterior_predictive=pm.sample_posterior_predictive(
                m1_trace, random_seed=1022
            )
        )
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [α, β_g, σ_β, μ_β]

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
  <progress value='6000' class='' max='6000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [6000/6000 00:24<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 2_000 tune and 1_000 draw iterations (4_000 + 2_000 draws total) took 32 seconds.

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
  100.00% [2000/2000 00:30<00:00]
</div>

```python
trace_collection["(m1): varying β"] = m1_trace
```

```python
az.plot_trace(m1_trace, var_names=["α", "β"], filter_vars="like");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_51_0.png)

```python
az.plot_posterior(m1_trace, var_names=["μ_β", "σ_β", "α"]);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_52_0.png)

The interpretation of the values for $\beta_g$ are as follows: For gene $j$, the expected fold change to the number of final reads is $\exp(\beta_j)$.
In other words, the final number of reads is the initial number of reads times $\exp(\beta_j)$.
Alternatively, $\beta_j$ is the expected log fold change for gene $j$.
These are demonstrated to be equivalent below.

$$
\begin{gather}
\exp(\beta_j) = \frac{f}{i} \\
\beta_j = \log \frac{f}{i} \quad \text{ or } \quad ie^{\beta_j} = f
\end{gather}
$$

Therefore, if $\beta_j = 0$, there is expected to be no change to the number of reads before and after: $i e^0 = i = f$.
If $\beta_j$ is positive, then there is expected to be an increase in the number of reads and if $\beta_j$ is negative, a decrease in the number of reads.

```python
ax = az.plot_forest(m1_trace, var_names="β_g", hdi_prob=0.89)
ax[0].set_yticklabels(data.hugo_symbol.cat.categories)
plt.axvline(x=0, ls="--")
plt.show()
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_54_0.png)

```python
az.summary(m1_trace, var_names="β_g", hdi_prob=0.89).assign(
    hugo_symbol=data.hugo_symbol.cat.categories
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
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
      <th>hugo_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>β_g[0]</th>
      <td>0.083</td>
      <td>0.118</td>
      <td>-0.120</td>
      <td>0.258</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4051.0</td>
      <td>1558.0</td>
      <td>1.00</td>
      <td>ACVR1C</td>
    </tr>
    <tr>
      <th>β_g[1]</th>
      <td>-0.091</td>
      <td>0.093</td>
      <td>-0.233</td>
      <td>0.058</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>3867.0</td>
      <td>1389.0</td>
      <td>1.00</td>
      <td>ADPRHL1</td>
    </tr>
    <tr>
      <th>β_g[2]</th>
      <td>0.094</td>
      <td>0.126</td>
      <td>-0.116</td>
      <td>0.289</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3561.0</td>
      <td>1389.0</td>
      <td>1.00</td>
      <td>APC</td>
    </tr>
    <tr>
      <th>β_g[3]</th>
      <td>-0.033</td>
      <td>0.090</td>
      <td>-0.181</td>
      <td>0.107</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4685.0</td>
      <td>1691.0</td>
      <td>1.00</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>β_g[4]</th>
      <td>-0.005</td>
      <td>0.092</td>
      <td>-0.149</td>
      <td>0.147</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3684.0</td>
      <td>1377.0</td>
      <td>1.00</td>
      <td>CCR3</td>
    </tr>
    <tr>
      <th>β_g[5]</th>
      <td>-0.148</td>
      <td>0.121</td>
      <td>-0.350</td>
      <td>0.027</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4702.0</td>
      <td>1614.0</td>
      <td>1.00</td>
      <td>CCR9</td>
    </tr>
    <tr>
      <th>β_g[6]</th>
      <td>0.041</td>
      <td>0.094</td>
      <td>-0.112</td>
      <td>0.181</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3739.0</td>
      <td>1445.0</td>
      <td>1.00</td>
      <td>CDK5RAP1</td>
    </tr>
    <tr>
      <th>β_g[7]</th>
      <td>-0.076</td>
      <td>0.092</td>
      <td>-0.217</td>
      <td>0.076</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3574.0</td>
      <td>1243.0</td>
      <td>1.01</td>
      <td>CSDC2</td>
    </tr>
    <tr>
      <th>β_g[8]</th>
      <td>-0.427</td>
      <td>0.096</td>
      <td>-0.581</td>
      <td>-0.274</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2776.0</td>
      <td>1308.0</td>
      <td>1.00</td>
      <td>CTNNB1</td>
    </tr>
    <tr>
      <th>β_g[9]</th>
      <td>-0.053</td>
      <td>0.126</td>
      <td>-0.248</td>
      <td>0.148</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>3735.0</td>
      <td>1260.0</td>
      <td>1.00</td>
      <td>CYTL1</td>
    </tr>
    <tr>
      <th>β_g[10]</th>
      <td>-0.115</td>
      <td>0.096</td>
      <td>-0.267</td>
      <td>0.037</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3292.0</td>
      <td>1376.0</td>
      <td>1.00</td>
      <td>DARS2</td>
    </tr>
    <tr>
      <th>β_g[11]</th>
      <td>-0.123</td>
      <td>0.127</td>
      <td>-0.320</td>
      <td>0.069</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4186.0</td>
      <td>1383.0</td>
      <td>1.00</td>
      <td>DPY19L1</td>
    </tr>
    <tr>
      <th>β_g[12]</th>
      <td>-0.054</td>
      <td>0.123</td>
      <td>-0.255</td>
      <td>0.140</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4390.0</td>
      <td>1134.0</td>
      <td>1.00</td>
      <td>EEF1AKMT4</td>
    </tr>
    <tr>
      <th>β_g[13]</th>
      <td>0.156</td>
      <td>0.099</td>
      <td>0.000</td>
      <td>0.314</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4600.0</td>
      <td>1084.0</td>
      <td>1.00</td>
      <td>EIF2AK1</td>
    </tr>
    <tr>
      <th>β_g[14]</th>
      <td>0.083</td>
      <td>0.124</td>
      <td>-0.111</td>
      <td>0.284</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>5558.0</td>
      <td>1252.0</td>
      <td>1.00</td>
      <td>FAM92A</td>
    </tr>
    <tr>
      <th>β_g[15]</th>
      <td>-0.134</td>
      <td>0.123</td>
      <td>-0.339</td>
      <td>0.057</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3722.0</td>
      <td>1558.0</td>
      <td>1.00</td>
      <td>FBXW7</td>
    </tr>
    <tr>
      <th>β_g[16]</th>
      <td>0.172</td>
      <td>0.121</td>
      <td>-0.020</td>
      <td>0.358</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3725.0</td>
      <td>1325.0</td>
      <td>1.00</td>
      <td>FCN1</td>
    </tr>
    <tr>
      <th>β_g[17]</th>
      <td>0.012</td>
      <td>0.096</td>
      <td>-0.137</td>
      <td>0.167</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3813.0</td>
      <td>1461.0</td>
      <td>1.00</td>
      <td>FUT7</td>
    </tr>
    <tr>
      <th>β_g[18]</th>
      <td>-0.097</td>
      <td>0.098</td>
      <td>-0.248</td>
      <td>0.062</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4306.0</td>
      <td>1338.0</td>
      <td>1.00</td>
      <td>GATA6</td>
    </tr>
    <tr>
      <th>β_g[19]</th>
      <td>-0.215</td>
      <td>0.126</td>
      <td>-0.413</td>
      <td>-0.016</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4025.0</td>
      <td>1601.0</td>
      <td>1.00</td>
      <td>GRIN2D</td>
    </tr>
    <tr>
      <th>β_g[20]</th>
      <td>-0.085</td>
      <td>0.127</td>
      <td>-0.286</td>
      <td>0.124</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3849.0</td>
      <td>1406.0</td>
      <td>1.00</td>
      <td>HSBP1L1</td>
    </tr>
    <tr>
      <th>β_g[21]</th>
      <td>-0.073</td>
      <td>0.121</td>
      <td>-0.269</td>
      <td>0.115</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4557.0</td>
      <td>1349.0</td>
      <td>1.00</td>
      <td>IMPA1</td>
    </tr>
    <tr>
      <th>β_g[22]</th>
      <td>-0.000</td>
      <td>0.124</td>
      <td>-0.205</td>
      <td>0.195</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4733.0</td>
      <td>1371.0</td>
      <td>1.00</td>
      <td>ING5</td>
    </tr>
    <tr>
      <th>β_g[23]</th>
      <td>-0.293</td>
      <td>0.128</td>
      <td>-0.500</td>
      <td>-0.100</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>6021.0</td>
      <td>1591.0</td>
      <td>1.00</td>
      <td>INPP5A</td>
    </tr>
    <tr>
      <th>β_g[24]</th>
      <td>-0.052</td>
      <td>0.099</td>
      <td>-0.213</td>
      <td>0.097</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3387.0</td>
      <td>1171.0</td>
      <td>1.00</td>
      <td>KDELC1</td>
    </tr>
    <tr>
      <th>β_g[25]</th>
      <td>-0.632</td>
      <td>0.103</td>
      <td>-0.791</td>
      <td>-0.466</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2829.0</td>
      <td>1535.0</td>
      <td>1.00</td>
      <td>KLF5</td>
    </tr>
    <tr>
      <th>β_g[26]</th>
      <td>-0.456</td>
      <td>0.106</td>
      <td>-0.634</td>
      <td>-0.300</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>4366.0</td>
      <td>1325.0</td>
      <td>1.00</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>β_g[27]</th>
      <td>-0.227</td>
      <td>0.129</td>
      <td>-0.421</td>
      <td>-0.013</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4555.0</td>
      <td>1165.0</td>
      <td>1.00</td>
      <td>LAPTM4B</td>
    </tr>
    <tr>
      <th>β_g[28]</th>
      <td>-0.012</td>
      <td>0.077</td>
      <td>-0.136</td>
      <td>0.107</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>5569.0</td>
      <td>1350.0</td>
      <td>1.01</td>
      <td>MDM2</td>
    </tr>
    <tr>
      <th>β_g[29]</th>
      <td>-0.028</td>
      <td>0.124</td>
      <td>-0.208</td>
      <td>0.193</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4343.0</td>
      <td>1525.0</td>
      <td>1.00</td>
      <td>NCDN</td>
    </tr>
    <tr>
      <th>β_g[30]</th>
      <td>-0.007</td>
      <td>0.093</td>
      <td>-0.150</td>
      <td>0.146</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3518.0</td>
      <td>1366.0</td>
      <td>1.00</td>
      <td>NOSTRIN</td>
    </tr>
    <tr>
      <th>β_g[31]</th>
      <td>-0.031</td>
      <td>0.096</td>
      <td>-0.181</td>
      <td>0.126</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3890.0</td>
      <td>1428.0</td>
      <td>1.00</td>
      <td>NRAS</td>
    </tr>
    <tr>
      <th>β_g[32]</th>
      <td>-0.094</td>
      <td>0.129</td>
      <td>-0.285</td>
      <td>0.116</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4534.0</td>
      <td>1440.0</td>
      <td>1.00</td>
      <td>PAFAH1B3</td>
    </tr>
    <tr>
      <th>β_g[33]</th>
      <td>-0.087</td>
      <td>0.093</td>
      <td>-0.231</td>
      <td>0.069</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>5266.0</td>
      <td>1204.0</td>
      <td>1.00</td>
      <td>PIK3CA</td>
    </tr>
    <tr>
      <th>β_g[34]</th>
      <td>0.232</td>
      <td>0.122</td>
      <td>0.053</td>
      <td>0.436</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3919.0</td>
      <td>1487.0</td>
      <td>1.00</td>
      <td>PLCD4</td>
    </tr>
    <tr>
      <th>β_g[35]</th>
      <td>-0.203</td>
      <td>0.130</td>
      <td>-0.408</td>
      <td>-0.002</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4151.0</td>
      <td>1152.0</td>
      <td>1.00</td>
      <td>PLIN2</td>
    </tr>
    <tr>
      <th>β_g[36]</th>
      <td>-0.014</td>
      <td>0.125</td>
      <td>-0.222</td>
      <td>0.178</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4918.0</td>
      <td>1516.0</td>
      <td>1.00</td>
      <td>PLK5</td>
    </tr>
    <tr>
      <th>β_g[37]</th>
      <td>-0.106</td>
      <td>0.079</td>
      <td>-0.235</td>
      <td>0.015</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>5016.0</td>
      <td>1218.0</td>
      <td>1.00</td>
      <td>POFUT2</td>
    </tr>
    <tr>
      <th>β_g[38]</th>
      <td>0.165</td>
      <td>0.119</td>
      <td>-0.037</td>
      <td>0.343</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3522.0</td>
      <td>1326.0</td>
      <td>1.00</td>
      <td>POU4F3</td>
    </tr>
    <tr>
      <th>β_g[39]</th>
      <td>-0.243</td>
      <td>0.126</td>
      <td>-0.434</td>
      <td>-0.033</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3697.0</td>
      <td>1402.0</td>
      <td>1.00</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>β_g[40]</th>
      <td>-0.353</td>
      <td>0.098</td>
      <td>-0.507</td>
      <td>-0.192</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4498.0</td>
      <td>1339.0</td>
      <td>1.00</td>
      <td>RFWD3</td>
    </tr>
    <tr>
      <th>β_g[41]</th>
      <td>-0.508</td>
      <td>0.085</td>
      <td>-0.633</td>
      <td>-0.360</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3527.0</td>
      <td>1529.0</td>
      <td>1.00</td>
      <td>RPL18A</td>
    </tr>
    <tr>
      <th>β_g[42]</th>
      <td>-0.658</td>
      <td>0.089</td>
      <td>-0.792</td>
      <td>-0.505</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3762.0</td>
      <td>1128.0</td>
      <td>1.00</td>
      <td>RPS26</td>
    </tr>
    <tr>
      <th>β_g[43]</th>
      <td>-0.006</td>
      <td>0.118</td>
      <td>-0.193</td>
      <td>0.181</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>3767.0</td>
      <td>1545.0</td>
      <td>1.00</td>
      <td>RTL3</td>
    </tr>
    <tr>
      <th>β_g[44]</th>
      <td>0.181</td>
      <td>0.091</td>
      <td>0.025</td>
      <td>0.316</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3487.0</td>
      <td>1347.0</td>
      <td>1.01</td>
      <td>S100A7A</td>
    </tr>
    <tr>
      <th>β_g[45]</th>
      <td>0.104</td>
      <td>0.118</td>
      <td>-0.098</td>
      <td>0.270</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3546.0</td>
      <td>1261.0</td>
      <td>1.00</td>
      <td>SEC14L5</td>
    </tr>
    <tr>
      <th>β_g[46]</th>
      <td>-0.158</td>
      <td>0.093</td>
      <td>-0.307</td>
      <td>-0.013</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3514.0</td>
      <td>1274.0</td>
      <td>1.00</td>
      <td>SEC23B</td>
    </tr>
    <tr>
      <th>β_g[47]</th>
      <td>0.004</td>
      <td>0.121</td>
      <td>-0.183</td>
      <td>0.192</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4929.0</td>
      <td>1492.0</td>
      <td>1.00</td>
      <td>SNX33</td>
    </tr>
    <tr>
      <th>β_g[48]</th>
      <td>0.112</td>
      <td>0.118</td>
      <td>-0.083</td>
      <td>0.289</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4063.0</td>
      <td>1360.0</td>
      <td>1.00</td>
      <td>SOSTDC1</td>
    </tr>
    <tr>
      <th>β_g[49]</th>
      <td>-0.055</td>
      <td>0.121</td>
      <td>-0.249</td>
      <td>0.136</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>3371.0</td>
      <td>1513.0</td>
      <td>1.00</td>
      <td>SOWAHC</td>
    </tr>
    <tr>
      <th>β_g[50]</th>
      <td>-0.134</td>
      <td>0.095</td>
      <td>-0.291</td>
      <td>0.010</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3684.0</td>
      <td>1637.0</td>
      <td>1.00</td>
      <td>SQLE</td>
    </tr>
    <tr>
      <th>β_g[51]</th>
      <td>0.065</td>
      <td>0.094</td>
      <td>-0.074</td>
      <td>0.225</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3632.0</td>
      <td>1094.0</td>
      <td>1.00</td>
      <td>STK11</td>
    </tr>
    <tr>
      <th>β_g[52]</th>
      <td>0.094</td>
      <td>0.095</td>
      <td>-0.064</td>
      <td>0.235</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3248.0</td>
      <td>1452.0</td>
      <td>1.00</td>
      <td>TBX19</td>
    </tr>
    <tr>
      <th>β_g[53]</th>
      <td>0.118</td>
      <td>0.092</td>
      <td>-0.034</td>
      <td>0.256</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3795.0</td>
      <td>1531.0</td>
      <td>1.00</td>
      <td>TMEM192</td>
    </tr>
    <tr>
      <th>β_g[54]</th>
      <td>0.292</td>
      <td>0.094</td>
      <td>0.135</td>
      <td>0.434</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>3989.0</td>
      <td>1225.0</td>
      <td>1.00</td>
      <td>TMEM241</td>
    </tr>
    <tr>
      <th>β_g[55]</th>
      <td>-0.089</td>
      <td>0.120</td>
      <td>-0.289</td>
      <td>0.088</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4948.0</td>
      <td>1105.0</td>
      <td>1.00</td>
      <td>TMPRSS3</td>
    </tr>
    <tr>
      <th>β_g[56]</th>
      <td>0.089</td>
      <td>0.123</td>
      <td>-0.112</td>
      <td>0.280</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3282.0</td>
      <td>1400.0</td>
      <td>1.00</td>
      <td>TP53</td>
    </tr>
    <tr>
      <th>β_g[57]</th>
      <td>-0.007</td>
      <td>0.126</td>
      <td>-0.211</td>
      <td>0.189</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>3973.0</td>
      <td>1601.0</td>
      <td>1.00</td>
      <td>TXNDC17</td>
    </tr>
    <tr>
      <th>β_g[58]</th>
      <td>-0.140</td>
      <td>0.094</td>
      <td>-0.288</td>
      <td>0.013</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4494.0</td>
      <td>1551.0</td>
      <td>1.00</td>
      <td>VCL</td>
    </tr>
    <tr>
      <th>β_g[59]</th>
      <td>-0.398</td>
      <td>0.084</td>
      <td>-0.539</td>
      <td>-0.272</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4162.0</td>
      <td>1464.0</td>
      <td>1.00</td>
      <td>YY1</td>
    </tr>
  </tbody>
</table>
</div>

The following plot shows the raw data for a few genes with large posterior estimates for $\beta$ in model 1.

```python
g = ["FUT7", "STK11", "GATA6", "KRAS", "KLF5", "CCR9", "FCN1", "NRAS", "PLCD4"]

(
    gg.ggplot(
        data[data.hugo_symbol.isin(g)].astype({"hugo_symbol": str}),
        gg.aes(x="initial_reads", y="read_counts", color="hugo_symbol"),
    )
    + gg.facet_wrap("~ hugo_symbol", ncol=3, scales="free")
    + gg.geom_point()
    + gg.geom_smooth(method="lm", formula="y~x", se=False, size=0.7, linetype="--")
    + gg.geom_abline(slope=1, intercept=0, linetype="--", color="k")
    + gg.scale_x_log10()
    + gg.scale_y_log10()
    + gg.theme(figure_size=(8, 8), panel_spacing=0.5, legend_position="none")
    + gg.labs(x="initial reads", y="final reads")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_57_0.png)

    <ggplot: (364641687)>

```python
(
    gg.ggplot(
        data[data.hugo_symbol.isin(g)].astype({"hugo_symbol": str}),
        gg.aes(x="hugo_symbol", y="lfc"),
    )
    + gg.geom_boxplot(outlier_alpha=0)
    + gg.geom_jitter(gg.aes(color="is_mutated"), width=0.3, height=0, alpha=0.4)
    + gg.geom_hline(yintercept=0, linetype="--")
    + gg.scale_y_continuous(expand=(0, 0.02, 0, 0.02))
    + gg.scale_color_manual(values={True: "red", False: "black"})
    + gg.theme(figure_size=(5, 4))
    + gg.labs(x="gene", y="LFC")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_58_0.png)

    <ggplot: (365086091)>

In this model, $\mu_\beta$ represents the average change in read counts.
Therefore, $\exp(\mu_\beta)$ is the same as the average fold change ($\frac{\text{final}}{\text{initial}}$).

```python
mu_beta = m1_trace["posterior"]["μ_β"].values.mean()
sigma_beta = m1_trace["posterior"]["σ_β"].values.mean()
mu_beta, np.exp(mu_beta), np.mean(data.read_counts / data.initial_reads)
```

    (-0.07720980439205351, 0.9256956182250815, 0.9247845993336123)

As the coefficient $\mu_\beta$ represents the average fold change over all genes, the coefficients $\beta_g$ represent individual gene fold changes.
The observed values and model posterior distribtuions are plotted below.
For each gene $j$, the circle and vertical line represent the mean and 89% CI of the posterior distribution for $\exp(\beta_j)$.
The dashed blue horizontal line represents the mean of the posterior for $\mu_\beta$.
The blue `x` represents the average observed fold change for the gene and the size of the `x` represents the number of data points for that gene.

This plot shows clearly the shrinkage of the estimates for $\beta_g$ towards the global mean.
Also, there is noticably less shrinkage for the genes with more data points — for example, compare the level of shrinkage for *PLCD4* and *TMEM241* (the genes with the two highest observed fold change value), both have similar observed averages, but there are more data points supporting *TMEM241*.

```python
gene_fold_change = (
    data.groupby("hugo_symbol")
    .apply(lambda d: np.mean(d.read_counts / d.initial_reads))
    .reset_index(drop=False)
    .rename(columns={0: "fold_change"})
)

num_data_points_per_gene = (
    data.groupby("hugo_symbol")[["read_counts"]].count().reset_index().read_counts
)

beta_g_posterior = (
    az.summary(m1_trace, var_names=["β_g"], hdi_prob=0.89)
    .reset_index(drop=False)
    .rename(
        columns={"index": "parameter", "hdi_5.5%": "hdi_low", "hdi_94.5%": "hdi_high"}
    )
    .assign(
        fold_change=gene_fold_change.fold_change,
        hugo_symbol=gene_fold_change.hugo_symbol,
        n_dp=num_data_points_per_gene,
    )
)

(
    gg.ggplot(beta_g_posterior, gg.aes(x="hugo_symbol"))
    + gg.geom_linerange(
        gg.aes(ymin="np.exp(hdi_low)", ymax="np.exp(hdi_high)"), alpha=0.25
    )
    + gg.geom_point(gg.aes(y="np.exp(mean)"))
    + gg.geom_point(
        gg.aes(y="fold_change", size="n_dp"), shape="x", color="blue", alpha=0.5
    )
    + gg.geom_hline(yintercept=np.exp(mu_beta), color="#011F4B", linetype="--")
    + gg.scale_size_continuous(range=(2, 4))
    + gg.theme(axis_text_x=gg.element_text(angle=90, size=7), figure_size=(8, 4))
    + gg.labs(
        x="gene",
        y="fold change (f/i)",
        title="Expected and observed fold change",
        size="# data\npoints",
    )
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_62_0.png)

    <ggplot: (365399827)>

Below are some explorations of the posterior predictions of the model.
Generally, they make sense and seem reasonable.

```python
def down_sample_ppc(ppc_ary: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    r_idx = np.arange(ppc_ary.shape[1])
    np.random.shuffle(r_idx)
    return ppc_ary[:, r_idx[:n]], r_idx
```

```python
m1_ppc_sample, sample_idx = down_sample_ppc(
    m1_trace["posterior_predictive"]["y"].squeeze().values, n=10
)
```

```python
m1_ppc_df = (
    pd.concat(
        [
            pd.DataFrame(m1_ppc_sample.T),
            data.loc[sample_idx[:10], :][["hugo_symbol", "sgrna"]].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    .assign(ppc_idx=lambda d: np.arange(d.shape[0]))
    .pivot_longer(
        index=["hugo_symbol", "sgrna", "ppc_idx"], names_to="_to_drop", values_to="draw"
    )
    .drop(columns="_to_drop")
)
```

```python
(
    gg.ggplot(m1_ppc_df)
    + gg.facet_wrap("~hugo_symbol", ncol=4, scales="free")
    + gg.geom_histogram(gg.aes(x="draw"), color="grey", fill="grey", alpha=0.2)
    + gg.geom_rug(
        gg.aes(x="read_counts"),
        data=data[data.hugo_symbol.isin(m1_ppc_df.hugo_symbol)],
        color="b",
        size=1.2,
        alpha=0.5,
    )
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.theme(figure_size=(8, 10), panel_spacing_x=0.25, panel_spacing_y=0.25)
    + gg.labs(x="predicted final read count", y="count")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_67_0.png)

    <ggplot: (365995731)>

```python
ppc_lfc = (
    m1_trace["posterior_predictive"]["y"].squeeze()
    / data.initial_reads.values.reshape(1, -1)
).values
ppc_lfc = np.log2(ppc_lfc)

ppc_lfc_df = (
    pd.concat([pd.DataFrame(ppc_lfc[:100, :].T), data[["hugo_symbol", "lfc"]]], axis=1)
    .pivot_longer(index=["hugo_symbol", "lfc"], values_to="ppc_lfc")
    .drop(columns=["variable"])
)

(
    gg.ggplot(ppc_lfc_df)
    + gg.facet_wrap("~hugo_symbol", ncol=4, scales="free")
    + gg.geom_density(gg.aes(x="ppc_lfc"), color="k", fill="k", alpha=0.1)
    + gg.geom_density(gg.aes(x="lfc"), color="b")
    + gg.geom_vline(xintercept=0, linetype="--", color="k", size=0.9)
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.theme(figure_size=(8, 20), panel_spacing_x=0.25, panel_spacing_y=0.25)
    + gg.labs(x="LFC", y="density", title="Posterior predictive check of LFC")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_68_0.png)

    <ggplot: (366401651)>

## Model 2

The second model includes the initial number of reads as a parameter in the linear portion of the model instead of as an exposure.
This perhaps will make the value of $\beta$ make more sense.

$$
\begin{gather}
\mu_\beta \sim N(0, 5) \\
\sigma_\beta \sim HN(5) \\
\beta_g \sim_g N(\mu_\beta, \sigma_\beta) \\
\eta = \beta_g[\text{gene}] X_\text{initial} \\
\mu = \exp(\eta) \\
\alpha = HN(5) \\
y \sim \text{NB}(\mu, \alpha)
\end{gather}
$$

```python
def noncentered_normal(name, shape, μ=None):
    if μ is None:
        μ = pm.Normal(f"μ_{name}", 0.0, 2.5)

    Δ = pm.Normal(f"Δ_{name}", 0.0, 1.0, shape=shape)
    σ = pm.HalfNormal(f"σ_{name}", 2.5)

    return pm.Deterministic(name, μ + Δ * σ)


with pm.Model() as nb_m2:
    g = pm.Data("gene_idx", gene_idx)
    log_initial_reads = pm.Data("log_initial_reads", np.log(data.initial_reads.values))
    final_reads = pm.Data("final_reads", data.read_counts.values)

    μ_β = pm.Normal("μ_β", 0, 2.5)
    σ_β = pm.HalfNormal("σ_β", 2.5)
    β = pm.Normal("β", μ_β, σ_β, shape=n_genes)

    η = pm.Deterministic("η", β[g] * log_initial_reads)

    μ = pm.Deterministic("μ", pm.math.exp(η))
    α = pm.HalfNormal("α", 5.0)

    y = pm.NegativeBinomial("y", μ, α, observed=final_reads)
```

```python
pm.model_to_graphviz(nb_m2)
```

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_71_0.svg)

```python
with nb_m2:
    m2_trace = pm.sample(1000, tune=2000, return_inferencedata=True)
    m2_trace.extend(
        az.from_pymc3(posterior_predictive=pm.sample_posterior_predictive(m2_trace))
    )
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [α, β, σ_β, μ_β]

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
  <progress value='6000' class='' max='6000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [6000/6000 00:29<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 2_000 tune and 1_000 draw iterations (4_000 + 2_000 draws total) took 37 seconds.

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
  100.00% [2000/2000 00:28<00:00]
</div>

```python
az.plot_trace(m2_trace, var_names=["α", "β"], filter_vars="like");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_73_0.png)

```python
az.plot_posterior(m2_trace, var_names=["μ_β", "σ_β", "α"]);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_74_0.png)

The relationship between initial and final read counts and the values for $\beta_g$ are slightly different than in model 1.
The relevant part of the formula for model 2 is shown below:

$$
\begin{gather}
\eta = \beta_g \times \log(x_i) \\
\mu = e^\eta \\
y \sim \text{NB}(\mu, \alpha)
\end{gather}
$$

If we collapse $\eta$, we get

$$
\mu = e^{\beta_g \times \log(x_i)}
$$

which can be further simplified

$$
\begin{aligned}
\mu &= e^{\beta_g \times \log(x_i)} \\
 &= (e^{\log(x_i)})^{\beta_g} \\
 &= x_i^{\beta_g}
\end{aligned}
$$

Thus, instead of $f = ie^\beta$ as in model 1, the relationship is $f = i^\beta$ (where $f$ is the final read count and $i$ is the initial read count).
I demonstrate this is relationship is true after the forest plot of the posteriors for $\beta_g$ below.

```python
ax = az.plot_forest(m2_trace, var_names="β", hdi_prob=0.89)
ax[0].set_yticklabels(data.hugo_symbol.cat.categories)
plt.axvline(x=1, ls="--")
plt.show()
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_76_0.png)

```python
beta_post = pd.DataFrame(
    {
        "hugo_symbol": data.hugo_symbol.cat.categories,
        "beta": m2_trace["posterior"]["β"].values.reshape(-1, n_genes).mean(axis=0),
    }
)

m2_post_pred_calc = (
    data[["hugo_symbol", "read_counts", "initial_reads"]]
    .merge(beta_post, on="hugo_symbol", how="left")
    .assign(pred_f_counts=lambda d: d.initial_reads ** d.beta)
)


(
    gg.ggplot(
        m2_post_pred_calc.sort_values(["hugo_symbol", "read_counts"]).assign(
            idx=lambda d: np.arange(d.shape[0]).astype(str)
        ),
        gg.aes(x="idx"),
    )
    + gg.facet_wrap("~ hugo_symbol", ncol=4, scales="free")
    + gg.geom_point(gg.aes(y="read_counts"), color="black", shape="x", size=1, alpha=1)
    + gg.geom_point(gg.aes(y="pred_f_counts"), color="blue", size=0.7, alpha=0.75)
    + gg.scale_y_log10()
    + gg.theme(
        axis_text_x=gg.element_blank(),
        axis_text_y=gg.element_text(size=7),
        figure_size=(8, 27),
        panel_spacing=0.3,
        panel_grid_major_x=gg.element_line(),
    )
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_77_0.png)

    <ggplot: (363142331)>

```python
g = ["APC", "FCN1", "NRAS", "PLIN2", "FUT7", "GATA6", "PLCD4"]

(
    gg.ggplot(
        data[data.hugo_symbol.isin(g)]
        .astype({"hugo_symbol": str})
        .assign(count_diff=lambda d: d.read_counts - d.initial_reads)
        .sort_values("is_mutated"),
        gg.aes(x="hugo_symbol", y="count_diff"),
    )
    + gg.geom_boxplot(outlier_alpha=0, color="#011F4B")
    + gg.geom_jitter(
        gg.aes(color="is_mutated"), width=0.25, height=0, alpha=0.6, size=1.8
    )
    + gg.geom_hline(yintercept=0, linetype="--")
    + gg.scale_y_continuous(expand=(0.02, 0, 0.02, 0))
    + gg.scale_color_manual(values={True: "red", False: "#011F4B"})
    + gg.labs(
        x="gene", y="read count difference\n(final - initial)", color="gene\nmutation"
    )
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_78_0.png)

    <ggplot: (367013479)>

```python
m2_ppc = pmanal.summarize_posterior_predictions(
    m2_trace["posterior_predictive"]["y"].squeeze().values,
    merge_with=data,
    calc_error=True,
    observed_y="read_counts",
)

(
    gg.ggplot(m2_ppc, gg.aes(x="read_counts", y="pred_mean"))
    + gg.geom_linerange(
        gg.aes(ymin="pred_hdi_low", ymax="pred_hdi_high"),
        alpha=0.1,
        size=0.4,
    )
    + gg.geom_point(alpha=0.4, size=1)
    + gg.geom_abline(slope=1, intercept=0, linetype="--", color="#011F4B")
    + gg.scale_x_log10()
    + gg.scale_y_log10()
    + gg.labs(
        x="observed final read count",
        y="predicted final read count",
        title="Posterior predictions",
    )
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_79_1.png)

    <ggplot: (368741370)>

## Model 3. Experimenting with copy number effect

```python
(
    gg.ggplot(data, gg.aes(x="copy_number", y="lfc"))
    + gg.geom_point(alpha=0.5)
    + gg.geom_smooth(method="lm", formula="y~x")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_81_0.png)

    <ggplot: (367214252)>

```python
(
    gg.ggplot(data, gg.aes(x="factor(np.round(copy_number*2)/2)", y="lfc"))
    + gg.geom_boxplot()
    + gg.labs(x="copy number", y="log fold change")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_82_0.png)

    <ggplot: (366316301)>

model formula:

$$
\begin{aligned}
\mu_\beta &\sim \text{N}(0, 2.5) \\
\sigma_\beta &\sim \text{HN}(2.5) \\
\beta_0 &\sim_g \text{N}(\mu_\beta, \sigma_\beta) \\
\beta_\text{CNA} &\sim \text{N}(0, 2.5) \\
\eta &= \beta_0 + \beta_\text{CNA} x_\text{cna} \\
\mu &= \exp(\eta) \\
\alpha &\sim \text{N}(0, 5) \\
y &\sim \text{NB}(\mu x_\text{initial}, \alpha)
\end{aligned}
$$

```python
data["copy_number_scaled"] = vhelp.careful_zscore(data.copy_number.values)

with pm.Model() as nb_m3:
    g = pm.Data("g", dphelp.get_indices(data, "hugo_symbol"))
    x_cna = pm.Data("x_cna", data.copy_number_scaled.values)
    initial_count = pm.Data("initial_count", data.initial_reads.values)
    final_count = pm.Data("final_count", data.read_counts.values)

    μ_β = pm.Normal("μ_β", 0, 2.5)
    σ_β = pm.HalfNormal("σ_β", 2.5)

    β_0 = pm.Normal("β_0", μ_β, σ_β, shape=n_genes)
    β_cna = pm.Normal("β_cna", 0, 2.5)
    η = pm.Deterministic("η", β_0[g] + β_cna * x_cna)

    μ = pm.Deterministic("μ", pm.math.exp(η))
    α = pm.HalfNormal("α", 5)

    y = pm.NegativeBinomial("y", μ * initial_count, α, observed=final_count)
```

```python
pm.model_to_graphviz(nb_m3)
```

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_85_0.svg)

```python
with nb_m3:
    m3_trace = pm.sample(
        1000, tune=2000, chains=4, random_seed=154, return_inferencedata=True
    )
    ppc = pm.sample_posterior_predictive(m3_trace, random_seed=154)
    m3_trace.extend(az.from_pymc3(posterior_predictive=ppc))
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, β_cna, β_0, σ_β, μ_β]

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
  100.00% [12000/12000 00:37<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 2_000 tune and 1_000 draw iterations (8_000 + 4_000 draws total) took 54 seconds.

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
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 00:56<00:00]
</div>

```python
trace_collection["(m3): single CNA"] = m3_trace
```

```python
az.plot_trace(m3_trace, var_names=["α", "β"], filter_vars="like");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_88_0.png)

```python
m3_ppc_df = pmanal.summarize_posterior_predictions(
    m3_trace["posterior_predictive"]["y"].values.squeeze(),
    merge_with=data,
    observed_y="y",
)

(
    gg.ggplot(m3_ppc_df, gg.aes(x="read_counts", y="pred_mean"))
    + gg.geom_linerange(gg.aes(ymin="pred_hdi_low", ymax="pred_hdi_high"), alpha=0.1)
    + gg.geom_point(alpha=0.25)
    + gg.geom_abline(slope=1, intercept=0, linetype="--", color="blue")
    + gg.scale_x_log10(expand=(0, 0, 0.02, 0))
    + gg.scale_y_log10(expand=(0, 0, 0.02, 0))
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_89_1.png)

    <ggplot: (367690042)>

```python
def prep_ppc(m: az.InferenceData, name: str) -> pd.DataFrame:
    df = pmanal.summarize_posterior_predictions(
        m["posterior_predictive"]["y"].values.squeeze()
    ).rename(
        columns={
            "pred_mean": f"{name}_pred",
            "pred_hdi_low": f"{name}_hdi_low",
            "pred_hdi_high": f"{name}_hdi_high",
        }
    )
    return df
```

```python
comparison_ppc_df = pd.concat(
    [prep_ppc(m, n) for m, n in [(m2_trace, "m2"), (m3_trace, "m3")]], axis=1
)
```

    /usr/local/Caskroom/miniconda/base/envs/speclet/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

```python
(
    gg.ggplot(comparison_ppc_df, gg.aes(x="m2_pred", y="m3_pred"))
    + gg.geom_point(alpha=0.2, size=0.5)
    + gg.scale_x_log10(expand=(0, 0, 0.02, 0))
    + gg.scale_y_log10(expand=(0, 0, 0.02, 0))
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_92_0.png)

    <ggplot: (367537785)>

```python
def prep_posterior_beta(
    m: az.InferenceData, m_name: str, var_name: str, genes: Sequence[str]
) -> pd.DataFrame:
    df = az.summary(m, var_names=var_name, kind="stats", hdi_prob=0.89).assign(
        hugo_symbol=genes, model=m_name
    )
    return df
```

```python
beta_posteriors = pd.concat(
    [
        prep_posterior_beta(m, n, b, data.hugo_symbol.cat.categories)
        for m, n, b in [(m1_trace, "m1", "β_g"), (m3_trace, "m3", "β_0")]
    ]
)

beta_posteriors.head()
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
      <th>hugo_symbol</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>β_g[0]</th>
      <td>0.083</td>
      <td>0.118</td>
      <td>-0.120</td>
      <td>0.258</td>
      <td>ACVR1C</td>
      <td>m1</td>
    </tr>
    <tr>
      <th>β_g[1]</th>
      <td>-0.091</td>
      <td>0.093</td>
      <td>-0.233</td>
      <td>0.058</td>
      <td>ADPRHL1</td>
      <td>m1</td>
    </tr>
    <tr>
      <th>β_g[2]</th>
      <td>0.094</td>
      <td>0.126</td>
      <td>-0.116</td>
      <td>0.289</td>
      <td>APC</td>
      <td>m1</td>
    </tr>
    <tr>
      <th>β_g[3]</th>
      <td>-0.033</td>
      <td>0.090</td>
      <td>-0.181</td>
      <td>0.107</td>
      <td>BRAF</td>
      <td>m1</td>
    </tr>
    <tr>
      <th>β_g[4]</th>
      <td>-0.005</td>
      <td>0.092</td>
      <td>-0.149</td>
      <td>0.147</td>
      <td>CCR3</td>
      <td>m1</td>
    </tr>
  </tbody>
</table>
</div>

```python
pos = gg.position_dodge(width=0.6)
(
    gg.ggplot(beta_posteriors, gg.aes(x="hugo_symbol", y="mean", color="model"))
    + gg.geom_linerange(
        gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), position=pos, size=0.5, alpha=0.5
    )
    + gg.geom_point(position=pos, size=0.75)
    + gg.scale_color_brewer(type="qual", palette="Dark2")
    + gg.theme(
        axis_text_x=gg.element_text(size=7, angle=90),
        axis_text_y=gg.element_text(size=7),
        figure_size=(8, 4),
        panel_grid_major_x=gg.element_line(),
    )
    + gg.labs(
        x="gene",
        y="posterior for varying intercept\n(mean and 95% CI)",
        title="Comparing posterior distributions for the varying intercept\nwith and without CNA covariate",
    )
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_95_0.png)

    <ggplot: (363987797)>

```python
beta_cna_post = m3_trace["posterior"]["β_cna"].values.mean()
beta_cna_post
```

    -0.061580945389013375

```python
gl = ["RTL3", "CSDC2", "KLF5", "EIF2AK1", "SQLE"]
#      down    down      up       up        up

(
    gg.ggplot(
        data[data.hugo_symbol.isin(gl)].assign(
            hugo_symbol=lambda d: d.hugo_symbol.astype(str)
        ),
        gg.aes(x="copy_number_scaled", y="lfc"),
    )
    + gg.geom_point(gg.aes(color="hugo_symbol"))
    + gg.geom_smooth(
        gg.aes(color="hugo_symbol"),
        formula="y~x",
        method="lm",
        linetype="--",
        se=False,
        size=0.6,
    )
    + gg.geom_abline(
        slope=beta_cna_post, intercept=0, color="black", linetype="--", size=0.8
    )
    + gg.scale_color_brewer(type="qual", palette="Set2")
    + gg.labs(x="copy number (z-scaled)", y="log fold change", color="gene")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_97_0.png)

    <ggplot: (366530036)>

## Model 4. Varying CNA covariate and intercept

model formula:

$$
\begin{aligned}
\mu_{\beta_0} &\sim \text{N}(0, 2.5) \quad \sigma_{\beta_0} \sim \text{HN}(2.5) \\
\mu_{\beta_\text{CNA}} &\sim \text{N}(0, 2.5) \quad \sigma_{\beta_\text{CNA}} \sim \text{HN}(2.5) \\
\beta_0 &\sim_g \text{N}(\mu_{\beta_0}, \sigma_{\beta_0}) \\
\beta_\text{CNA} &\sim_g \text{N}(\mu_{\beta_\text{CNA}}, \sigma_{\beta_\text{CNA}}) \\
\eta &= \beta_0[g] + x_\text{cna} \beta_\text{CNA}[g] \\
\mu &= \exp(\eta) \\
\alpha &\sim \text{N}(0, 5) \\
y &\sim \text{NB}(\mu x_\text{initial}, \alpha)
\end{aligned}
$$

```python
data["copy_number_scaled"] = vhelp.careful_zscore(data.copy_number.values)


def make_hierarchical_noncentered_coef(
    name: str,
    shape: Union[float, tuple[float, ...]],
    mu_sd: float = 2.5,
    sigma_sd: float = 2.5,
) -> pm.model.FreeRV:
    μ = pm.Normal(f"μ_{name}", 0.0, mu_sd)
    σ = pm.HalfNormal(f"σ_{name}", sigma_sd)
    Δ = pm.Normal(f"Δ_{name}", 0.0, 1.0, shape=shape)
    β = pm.Deterministic(name, μ + Δ * σ)
    return β


with pm.Model() as nb_m4:
    g = pm.Data("g", dphelp.get_indices(data, "hugo_symbol"))
    x_cna = pm.Data("x_cna", data.copy_number_scaled.values)
    initial_count = pm.Data("initial_count", data.initial_reads.values)
    final_count = pm.Data("final_count", data.read_counts.values)

    β_0 = make_hierarchical_noncentered_coef("β_0", shape=n_genes)
    β_cna = make_hierarchical_noncentered_coef(
        "β_cna", shape=n_genes, mu_sd=1.0, sigma_sd=1.0
    )
    η = pm.Deterministic("η", β_0[g] + β_cna[g] * x_cna)

    μ = pm.Deterministic("μ", pm.math.exp(η))
    α = pm.HalfNormal("α", 5)

    y = pm.NegativeBinomial("y", μ * initial_count, α, observed=final_count)
```

```python
pm.model_to_graphviz(nb_m4)
```

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_100_0.svg)

```python
with nb_m4:
    m4_trace = pm.sample(
        2000,
        chains=4,
        tune=3000,
        random_seed=400,
        target_accept=0.95,
        return_inferencedata=True,
    )
    ppc = pm.sample_posterior_predictive(m4_trace, random_seed=400)
    m4_trace.extend(az.from_pymc3(posterior_predictive=ppc))
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 2 jobs)
    NUTS: [α, Δ_β_cna, σ_β_cna, μ_β_cna, Δ_β_0, σ_β_0, μ_β_0]

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
  <progress value='20000' class='' max='20000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [20000/20000 03:12<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 3_000 tune and 2_000 draw iterations (12_000 + 8_000 draws total) took 211 seconds.
    The number of effective samples is smaller than 25% for some parameters.

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
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 02:05<00:00]
</div>

```python
trace_collection["(m4) varying CNA"] = m4_trace
```

```python
az.plot_trace(
    m4_trace, var_names=["α", "β_0", "β_cna", "μ_β_0", "σ_β_0", "μ_β_cna", "σ_β_cna"]
);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_103_0.png)

```python
m4_beta_cna_post_df = az.summary(m4_trace, var_names="β_cna", hdi_prob=0.89).assign(
    hugo_symbol=data.hugo_symbol.cat.categories
)

mu_beta_cna_post = az.summary(
    m4_trace, var_names="μ_β_cna", hdi_prob=0.89, kind="stats"
)

(
    gg.ggplot(
        m4_beta_cna_post_df.sort_values("mean").assign(
            hugo_symbol=lambda d: pd.Categorical(
                d.hugo_symbol.astype(str),
                ordered=True,
                categories=d.hugo_symbol.astype(str),
            )
        ),
        gg.aes(x="hugo_symbol", y="mean"),
    )
    + gg.geom_hline(yintercept=mu_beta_cna_post["mean"][0])
    + gg.geom_hline(
        yintercept=[mu_beta_cna_post["hdi_5.5%"][0], mu_beta_cna_post["hdi_94.5%"][0]],
        alpha=0.5,
        linetype="--",
    )
    + gg.geom_linerange(
        gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"), color="#011F4B", alpha=0.5
    )
    + gg.geom_point(color="#011F4B", size=1)
    + gg.scale_y_continuous(expand=(0.02, 0, 0.02, 0))
    + gg.theme(
        figure_size=(8, 5),
        axis_text_x=gg.element_text(angle=90, size=7),
        panel_grid_major_y=gg.element_line(),
    )
    + gg.labs(
        x="gene",
        y="β_CNA posterior (mean and 89% CI)",
        title="Posterior coefficients for copy number effect",
    )
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_104_0.png)

    <ggplot: (363741561)>

```python
gl = m4_beta_cna_post_df.sort_values("mean").head(n=6).hugo_symbol.values

(
    gg.ggplot(
        data[data.hugo_symbol.isin(gl)].assign(
            hugo_symbol=lambda d: d.hugo_symbol.astype(str)
        ),
        gg.aes(x="copy_number_scaled", y="lfc"),
    )
    + gg.geom_point(gg.aes(color="hugo_symbol"), size=1, alpha=0.75)
    + gg.geom_smooth(
        gg.aes(color="hugo_symbol"),
        formula="y~x",
        method="lm",
        linetype="--",
        se=False,
        size=0.6,
    )
    + gg.scale_color_brewer(type="qual", palette="Dark2")
    + gg.labs(x="copy number (z-scaled)", y="log fold change", color="gene")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_105_0.png)

    <ggplot: (368512199)>

## Final comparison of all models

Make sure to note that this analysis was conducted with a relatively small subset of the data.
Further, the amount of data per gene is quite small, limiting the hierarchical models.

```python
model_comparison = az.compare(trace_collection)
az.plot_compare(model_comparison);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_107_0.png)

```python
model_comparison
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
      <th>(m4) varying CNA</th>
      <td>0</td>
      <td>-3671.334442</td>
      <td>53.263715</td>
      <td>0.000000</td>
      <td>5.321194e-01</td>
      <td>25.612641</td>
      <td>0.000000</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>(m3): single CNA</th>
      <td>1</td>
      <td>-3671.443414</td>
      <td>41.760554</td>
      <td>0.108973</td>
      <td>2.501019e-01</td>
      <td>25.390234</td>
      <td>2.273203</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>(m1): varying β</th>
      <td>2</td>
      <td>-3675.537718</td>
      <td>39.705807</td>
      <td>4.203277</td>
      <td>2.177787e-01</td>
      <td>25.478605</td>
      <td>4.647333</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>(m0) single β</th>
      <td>3</td>
      <td>-3752.388299</td>
      <td>2.346482</td>
      <td>81.053857</td>
      <td>1.110223e-16</td>
      <td>25.017064</td>
      <td>12.145775</td>
      <td>False</td>
      <td>log</td>
    </tr>
  </tbody>
</table>
</div>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 16.00 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-09-10

    Python implementation: CPython
    Python version       : 3.9.6
    IPython version      : 7.26.0

    Compiler    : Clang 11.1.0
    OS          : Darwin
    Release     : 20.4.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: JHCookMac.local

    Git branch: nb-model

    seaborn   : 0.11.2
    pandas    : 1.3.2
    re        : 2.2.1
    plotnine  : 0.8.0
    janitor   : 0.21.0
    arviz     : 0.11.2
    matplotlib: 3.4.3
    theano    : 1.0.5
    pymc3     : 3.11.2
    numpy     : 1.21.2
