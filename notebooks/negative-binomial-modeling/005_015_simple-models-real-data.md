# Exploration of negative binomial models on real data

```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload

```python
import re
import string
import warnings
from pathlib import Path
from time import time

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

    <ggplot: (357398010)>

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

    <ggplot: (357297824)>

## Model 0. Single gene

For simplicity and to help me lear nho to interpret coefficient values, the first model was just for the read counts of *KRAS*.
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
  100.00% [8000/8000 00:09<00:00 Sampling 4 chains, 0 divergences]
</div>

    Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 30 seconds.

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
  100.00% [4000/4000 00:55<00:00]
</div>

```python
az.plot_trace(m0_trace, compact=False);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_30_0.png)

```python
az.summary(m0_trace)
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

    <ggplot: (370101547)>

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

    <ggplot: (369756456)>

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

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_40_0.svg)

```python
with nb_m1:
    m1_trace = pm.sample(1000, tune=2000, random_seed=True, return_inferencedata=True)
    m1_trace.extend(
        az.from_pymc3(posterior_predictive=pm.sample_posterior_predictive(m1_trace))
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
  100.00% [6000/6000 00:17<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 2_000 tune and 1_000 draw iterations (4_000 + 2_000 draws total) took 28 seconds.

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
  100.00% [2000/2000 00:23<00:00]
</div>

```python
az.plot_trace(m1_trace, var_names=["α", "β"], filter_vars="like");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_42_0.png)

```python
az.plot_posterior(m1_trace, var_names=["μ_β", "σ_β", "α"]);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_43_0.png)

The interpretation of the values for $\beta_g$ are as follows: For gene $j$, the expected change to the number of log final read counts is $\beta_j$.
Therefore, if $\beta_j = 0$, there is expected to be a change in the count of final reads $\exp(0) = 1$.
If $\beta_j$ is positive, then there is expected to be

```python
ax = az.plot_forest(m1_trace, var_names="β_g", hdi_prob=0.89)
ax[0].set_yticklabels(data.hugo_symbol.cat.categories)
plt.axvline(x=0, ls="--")
plt.show()
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_45_0.png)

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
      <td>0.081</td>
      <td>0.122</td>
      <td>-0.119</td>
      <td>0.270</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4871.0</td>
      <td>1498.0</td>
      <td>1.00</td>
      <td>ACVR1C</td>
    </tr>
    <tr>
      <th>β_g[1]</th>
      <td>-0.089</td>
      <td>0.091</td>
      <td>-0.241</td>
      <td>0.044</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3490.0</td>
      <td>1213.0</td>
      <td>1.00</td>
      <td>ADPRHL1</td>
    </tr>
    <tr>
      <th>β_g[2]</th>
      <td>0.095</td>
      <td>0.125</td>
      <td>-0.090</td>
      <td>0.292</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4986.0</td>
      <td>1442.0</td>
      <td>1.00</td>
      <td>APC</td>
    </tr>
    <tr>
      <th>β_g[3]</th>
      <td>-0.035</td>
      <td>0.096</td>
      <td>-0.174</td>
      <td>0.136</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>5220.0</td>
      <td>1559.0</td>
      <td>1.00</td>
      <td>BRAF</td>
    </tr>
    <tr>
      <th>β_g[4]</th>
      <td>-0.000</td>
      <td>0.093</td>
      <td>-0.160</td>
      <td>0.138</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4741.0</td>
      <td>1456.0</td>
      <td>1.00</td>
      <td>CCR3</td>
    </tr>
    <tr>
      <th>β_g[5]</th>
      <td>-0.148</td>
      <td>0.124</td>
      <td>-0.327</td>
      <td>0.072</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4831.0</td>
      <td>1586.0</td>
      <td>1.00</td>
      <td>CCR9</td>
    </tr>
    <tr>
      <th>β_g[6]</th>
      <td>0.042</td>
      <td>0.096</td>
      <td>-0.113</td>
      <td>0.182</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4584.0</td>
      <td>1301.0</td>
      <td>1.00</td>
      <td>CDK5RAP1</td>
    </tr>
    <tr>
      <th>β_g[7]</th>
      <td>-0.075</td>
      <td>0.098</td>
      <td>-0.220</td>
      <td>0.090</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>5183.0</td>
      <td>1517.0</td>
      <td>1.00</td>
      <td>CSDC2</td>
    </tr>
    <tr>
      <th>β_g[8]</th>
      <td>-0.428</td>
      <td>0.102</td>
      <td>-0.588</td>
      <td>-0.267</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3889.0</td>
      <td>1519.0</td>
      <td>1.00</td>
      <td>CTNNB1</td>
    </tr>
    <tr>
      <th>β_g[9]</th>
      <td>-0.055</td>
      <td>0.124</td>
      <td>-0.257</td>
      <td>0.137</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4802.0</td>
      <td>1463.0</td>
      <td>1.00</td>
      <td>CYTL1</td>
    </tr>
    <tr>
      <th>β_g[10]</th>
      <td>-0.113</td>
      <td>0.097</td>
      <td>-0.278</td>
      <td>0.031</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3487.0</td>
      <td>1177.0</td>
      <td>1.00</td>
      <td>DARS2</td>
    </tr>
    <tr>
      <th>β_g[11]</th>
      <td>-0.125</td>
      <td>0.126</td>
      <td>-0.321</td>
      <td>0.078</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4085.0</td>
      <td>1369.0</td>
      <td>1.00</td>
      <td>DPY19L1</td>
    </tr>
    <tr>
      <th>β_g[12]</th>
      <td>-0.055</td>
      <td>0.125</td>
      <td>-0.249</td>
      <td>0.144</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4937.0</td>
      <td>1362.0</td>
      <td>1.00</td>
      <td>EEF1AKMT4</td>
    </tr>
    <tr>
      <th>β_g[13]</th>
      <td>0.157</td>
      <td>0.096</td>
      <td>0.008</td>
      <td>0.310</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3816.0</td>
      <td>1221.0</td>
      <td>1.00</td>
      <td>EIF2AK1</td>
    </tr>
    <tr>
      <th>β_g[14]</th>
      <td>0.081</td>
      <td>0.125</td>
      <td>-0.122</td>
      <td>0.268</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>5718.0</td>
      <td>1166.0</td>
      <td>1.00</td>
      <td>FAM92A</td>
    </tr>
    <tr>
      <th>β_g[15]</th>
      <td>-0.134</td>
      <td>0.131</td>
      <td>-0.347</td>
      <td>0.066</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5382.0</td>
      <td>1426.0</td>
      <td>1.00</td>
      <td>FBXW7</td>
    </tr>
    <tr>
      <th>β_g[16]</th>
      <td>0.172</td>
      <td>0.122</td>
      <td>-0.025</td>
      <td>0.357</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4117.0</td>
      <td>1434.0</td>
      <td>1.00</td>
      <td>FCN1</td>
    </tr>
    <tr>
      <th>β_g[17]</th>
      <td>0.012</td>
      <td>0.097</td>
      <td>-0.132</td>
      <td>0.172</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4011.0</td>
      <td>1579.0</td>
      <td>1.00</td>
      <td>FUT7</td>
    </tr>
    <tr>
      <th>β_g[18]</th>
      <td>-0.095</td>
      <td>0.094</td>
      <td>-0.233</td>
      <td>0.065</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3520.0</td>
      <td>1331.0</td>
      <td>1.00</td>
      <td>GATA6</td>
    </tr>
    <tr>
      <th>β_g[19]</th>
      <td>-0.212</td>
      <td>0.125</td>
      <td>-0.413</td>
      <td>-0.023</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5428.0</td>
      <td>1464.0</td>
      <td>1.00</td>
      <td>GRIN2D</td>
    </tr>
    <tr>
      <th>β_g[20]</th>
      <td>-0.088</td>
      <td>0.129</td>
      <td>-0.296</td>
      <td>0.111</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4386.0</td>
      <td>1417.0</td>
      <td>1.00</td>
      <td>HSBP1L1</td>
    </tr>
    <tr>
      <th>β_g[21]</th>
      <td>-0.075</td>
      <td>0.120</td>
      <td>-0.267</td>
      <td>0.112</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4715.0</td>
      <td>1221.0</td>
      <td>1.00</td>
      <td>IMPA1</td>
    </tr>
    <tr>
      <th>β_g[22]</th>
      <td>0.006</td>
      <td>0.121</td>
      <td>-0.198</td>
      <td>0.186</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4770.0</td>
      <td>1573.0</td>
      <td>1.00</td>
      <td>ING5</td>
    </tr>
    <tr>
      <th>β_g[23]</th>
      <td>-0.286</td>
      <td>0.132</td>
      <td>-0.497</td>
      <td>-0.071</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4213.0</td>
      <td>1222.0</td>
      <td>1.00</td>
      <td>INPP5A</td>
    </tr>
    <tr>
      <th>β_g[24]</th>
      <td>-0.052</td>
      <td>0.097</td>
      <td>-0.218</td>
      <td>0.091</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3979.0</td>
      <td>1305.0</td>
      <td>1.00</td>
      <td>KDELC1</td>
    </tr>
    <tr>
      <th>β_g[25]</th>
      <td>-0.631</td>
      <td>0.107</td>
      <td>-0.792</td>
      <td>-0.454</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>4153.0</td>
      <td>1366.0</td>
      <td>1.00</td>
      <td>KLF5</td>
    </tr>
    <tr>
      <th>β_g[26]</th>
      <td>-0.459</td>
      <td>0.102</td>
      <td>-0.622</td>
      <td>-0.303</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3797.0</td>
      <td>1398.0</td>
      <td>1.00</td>
      <td>KRAS</td>
    </tr>
    <tr>
      <th>β_g[27]</th>
      <td>-0.228</td>
      <td>0.124</td>
      <td>-0.429</td>
      <td>-0.037</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3199.0</td>
      <td>1605.0</td>
      <td>1.00</td>
      <td>LAPTM4B</td>
    </tr>
    <tr>
      <th>β_g[28]</th>
      <td>-0.013</td>
      <td>0.078</td>
      <td>-0.145</td>
      <td>0.107</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4700.0</td>
      <td>1397.0</td>
      <td>1.00</td>
      <td>MDM2</td>
    </tr>
    <tr>
      <th>β_g[29]</th>
      <td>-0.028</td>
      <td>0.122</td>
      <td>-0.220</td>
      <td>0.165</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4937.0</td>
      <td>1438.0</td>
      <td>1.00</td>
      <td>NCDN</td>
    </tr>
    <tr>
      <th>β_g[30]</th>
      <td>-0.003</td>
      <td>0.097</td>
      <td>-0.149</td>
      <td>0.156</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>5125.0</td>
      <td>1392.0</td>
      <td>1.00</td>
      <td>NOSTRIN</td>
    </tr>
    <tr>
      <th>β_g[31]</th>
      <td>-0.032</td>
      <td>0.096</td>
      <td>-0.176</td>
      <td>0.123</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4768.0</td>
      <td>1315.0</td>
      <td>1.00</td>
      <td>NRAS</td>
    </tr>
    <tr>
      <th>β_g[32]</th>
      <td>-0.092</td>
      <td>0.124</td>
      <td>-0.269</td>
      <td>0.125</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5261.0</td>
      <td>1492.0</td>
      <td>1.00</td>
      <td>PAFAH1B3</td>
    </tr>
    <tr>
      <th>β_g[33]</th>
      <td>-0.087</td>
      <td>0.091</td>
      <td>-0.225</td>
      <td>0.063</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4260.0</td>
      <td>1395.0</td>
      <td>1.00</td>
      <td>PIK3CA</td>
    </tr>
    <tr>
      <th>β_g[34]</th>
      <td>0.234</td>
      <td>0.120</td>
      <td>0.028</td>
      <td>0.417</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3983.0</td>
      <td>1581.0</td>
      <td>1.00</td>
      <td>PLCD4</td>
    </tr>
    <tr>
      <th>β_g[35]</th>
      <td>-0.205</td>
      <td>0.131</td>
      <td>-0.411</td>
      <td>0.005</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>5447.0</td>
      <td>1566.0</td>
      <td>1.00</td>
      <td>PLIN2</td>
    </tr>
    <tr>
      <th>β_g[36]</th>
      <td>-0.013</td>
      <td>0.125</td>
      <td>-0.211</td>
      <td>0.178</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>5285.0</td>
      <td>1645.0</td>
      <td>1.00</td>
      <td>PLK5</td>
    </tr>
    <tr>
      <th>β_g[37]</th>
      <td>-0.105</td>
      <td>0.082</td>
      <td>-0.248</td>
      <td>0.014</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4775.0</td>
      <td>1713.0</td>
      <td>1.00</td>
      <td>POFUT2</td>
    </tr>
    <tr>
      <th>β_g[38]</th>
      <td>0.162</td>
      <td>0.122</td>
      <td>-0.038</td>
      <td>0.353</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4245.0</td>
      <td>1585.0</td>
      <td>1.00</td>
      <td>POU4F3</td>
    </tr>
    <tr>
      <th>β_g[39]</th>
      <td>-0.244</td>
      <td>0.125</td>
      <td>-0.434</td>
      <td>-0.041</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>3742.0</td>
      <td>1531.0</td>
      <td>1.00</td>
      <td>PTK2</td>
    </tr>
    <tr>
      <th>β_g[40]</th>
      <td>-0.349</td>
      <td>0.096</td>
      <td>-0.498</td>
      <td>-0.192</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4340.0</td>
      <td>1660.0</td>
      <td>1.00</td>
      <td>RFWD3</td>
    </tr>
    <tr>
      <th>β_g[41]</th>
      <td>-0.508</td>
      <td>0.082</td>
      <td>-0.631</td>
      <td>-0.371</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4304.0</td>
      <td>1618.0</td>
      <td>1.00</td>
      <td>RPL18A</td>
    </tr>
    <tr>
      <th>β_g[42]</th>
      <td>-0.660</td>
      <td>0.089</td>
      <td>-0.800</td>
      <td>-0.514</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4492.0</td>
      <td>1598.0</td>
      <td>1.00</td>
      <td>RPS26</td>
    </tr>
    <tr>
      <th>β_g[43]</th>
      <td>-0.003</td>
      <td>0.127</td>
      <td>-0.227</td>
      <td>0.176</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4415.0</td>
      <td>1723.0</td>
      <td>1.00</td>
      <td>RTL3</td>
    </tr>
    <tr>
      <th>β_g[44]</th>
      <td>0.186</td>
      <td>0.098</td>
      <td>0.030</td>
      <td>0.342</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4676.0</td>
      <td>1414.0</td>
      <td>1.00</td>
      <td>S100A7A</td>
    </tr>
    <tr>
      <th>β_g[45]</th>
      <td>0.106</td>
      <td>0.124</td>
      <td>-0.096</td>
      <td>0.300</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4090.0</td>
      <td>1224.0</td>
      <td>1.00</td>
      <td>SEC14L5</td>
    </tr>
    <tr>
      <th>β_g[46]</th>
      <td>-0.161</td>
      <td>0.096</td>
      <td>-0.308</td>
      <td>-0.005</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4773.0</td>
      <td>1244.0</td>
      <td>1.00</td>
      <td>SEC23B</td>
    </tr>
    <tr>
      <th>β_g[47]</th>
      <td>0.005</td>
      <td>0.121</td>
      <td>-0.183</td>
      <td>0.195</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4406.0</td>
      <td>1452.0</td>
      <td>1.00</td>
      <td>SNX33</td>
    </tr>
    <tr>
      <th>β_g[48]</th>
      <td>0.112</td>
      <td>0.128</td>
      <td>-0.093</td>
      <td>0.319</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4438.0</td>
      <td>1334.0</td>
      <td>1.00</td>
      <td>SOSTDC1</td>
    </tr>
    <tr>
      <th>β_g[49]</th>
      <td>-0.052</td>
      <td>0.121</td>
      <td>-0.242</td>
      <td>0.142</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4520.0</td>
      <td>1306.0</td>
      <td>1.00</td>
      <td>SOWAHC</td>
    </tr>
    <tr>
      <th>β_g[50]</th>
      <td>-0.134</td>
      <td>0.091</td>
      <td>-0.285</td>
      <td>0.006</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>3662.0</td>
      <td>1275.0</td>
      <td>1.00</td>
      <td>SQLE</td>
    </tr>
    <tr>
      <th>β_g[51]</th>
      <td>0.067</td>
      <td>0.091</td>
      <td>-0.080</td>
      <td>0.207</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4317.0</td>
      <td>1126.0</td>
      <td>1.01</td>
      <td>STK11</td>
    </tr>
    <tr>
      <th>β_g[52]</th>
      <td>0.096</td>
      <td>0.096</td>
      <td>-0.055</td>
      <td>0.252</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4821.0</td>
      <td>1428.0</td>
      <td>1.01</td>
      <td>TBX19</td>
    </tr>
    <tr>
      <th>β_g[53]</th>
      <td>0.121</td>
      <td>0.094</td>
      <td>-0.026</td>
      <td>0.276</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4005.0</td>
      <td>1362.0</td>
      <td>1.00</td>
      <td>TMEM192</td>
    </tr>
    <tr>
      <th>β_g[54]</th>
      <td>0.292</td>
      <td>0.094</td>
      <td>0.140</td>
      <td>0.436</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4302.0</td>
      <td>1529.0</td>
      <td>1.00</td>
      <td>TMEM241</td>
    </tr>
    <tr>
      <th>β_g[55]</th>
      <td>-0.088</td>
      <td>0.119</td>
      <td>-0.282</td>
      <td>0.099</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>4627.0</td>
      <td>1633.0</td>
      <td>1.00</td>
      <td>TMPRSS3</td>
    </tr>
    <tr>
      <th>β_g[56]</th>
      <td>0.086</td>
      <td>0.122</td>
      <td>-0.102</td>
      <td>0.287</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>4477.0</td>
      <td>1376.0</td>
      <td>1.00</td>
      <td>TP53</td>
    </tr>
    <tr>
      <th>β_g[57]</th>
      <td>-0.011</td>
      <td>0.122</td>
      <td>-0.204</td>
      <td>0.178</td>
      <td>0.002</td>
      <td>0.003</td>
      <td>5038.0</td>
      <td>1518.0</td>
      <td>1.00</td>
      <td>TXNDC17</td>
    </tr>
    <tr>
      <th>β_g[58]</th>
      <td>-0.136</td>
      <td>0.101</td>
      <td>-0.287</td>
      <td>0.026</td>
      <td>0.001</td>
      <td>0.002</td>
      <td>4697.0</td>
      <td>1147.0</td>
      <td>1.00</td>
      <td>VCL</td>
    </tr>
    <tr>
      <th>β_g[59]</th>
      <td>-0.397</td>
      <td>0.082</td>
      <td>-0.532</td>
      <td>-0.274</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>4101.0</td>
      <td>1320.0</td>
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

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_48_0.png)

    <ggplot: (362459901)>

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

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_49_0.png)

    <ggplot: (357150847)>

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

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_54_0.png)

    <ggplot: (356057793)>

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

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_55_0.png)

    <ggplot: (355150165)>

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

![svg](005_015_simple-models-real-data_files/005_015_simple-models-real-data_58_0.svg)

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
  100.00% [6000/6000 00:20<00:00 Sampling 2 chains, 0 divergences]
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
  100.00% [2000/2000 00:24<00:00]
</div>

```python
az.plot_trace(m2_trace, var_names=["α", "β"], filter_vars="like");
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_60_0.png)

```python
az.plot_posterior(m2_trace, var_names=["μ_β", "σ_β", "α"]);
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_61_0.png)

The posterior estimates for $\beta$ are just shifted by 1 now.
Here is the interpretation for $\beta_g$: for gene $j$, a unit change in initial read counts reduces the expected number of log final read counts by $\beta_j$.
These values are not as clear as when modeling the inital read count as the exposure because there, a $\beta_j = 0$ means that the gene is

```python
ax = az.plot_forest(m2_trace, var_names="β", hdi_prob=0.89)
ax[0].set_yticklabels(data.hugo_symbol.cat.categories)
plt.axvline(x=1, ls="--")
plt.show()
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_63_0.png)

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

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_64_0.png)

    <ggplot: (363141286)>

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

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_65_1.png)

    <ggplot: (363295702)>

## Model 3. Experimenting with copy number effect

```python
(
    gg.ggplot(data, gg.aes(x="copy_number", y="lfc"))
    + gg.geom_point(alpha=0.5)
    + gg.geom_smooth(method="lm", formula="y~x")
)
```

![png](005_015_simple-models-real-data_files/005_015_simple-models-real-data_67_0.png)

    <ggplot: (286594813)>

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

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 3.45 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-09-09

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

    arviz     : 0.11.2
    matplotlib: 3.4.3
    re        : 2.2.1
    pandas    : 1.3.2
    numpy     : 1.21.2
    pymc3     : 3.11.2
    plotnine  : 0.8.0
    theano    : 1.0.5
    seaborn   : 0.11.2
    janitor   : 0.21.0
