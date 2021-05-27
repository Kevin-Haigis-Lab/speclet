# Model Report

```python
import warnings
from pathlib import Path
from time import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from src.command_line_interfaces import sampling_pymc3_models_cli as sampling
from src.data_processing import common as dphelp
from src.modeling import pymc3_analysis as pmanal
from src.modeling import pymc3_sampling_api as pmapi
from src.models.speclet_pipeline_test_model import SpecletTestModel
from src.plot.color_pal import SeabornColor

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)
```

Parameters for papermill:

- `MODEL`: which model was tested
- `MODEL_NAME`: name of the model
- `DEBUG`: if in debug mode or not
- `FIT_METHOD`: method used to fit the model; either "ADVI" or "MCMC"

## Setup

### Papermill parameters

```python
MODEL = ""
MODEL_NAME = ""
DEBUG = True
FIT_METHOD = ""
```

```python
# Parameters
MODEL = "speclet-three"
MODEL_NAME = "SpecletThree-kras-debug"
DEBUG = True
FIT_METHOD = "MCMC"
```

```python
assert FIT_METHOD in ["ADVI", "MCMC"]
```

```python
speclet_model = sampling.sample_speclet_model(
    MODEL,
    name=MODEL_NAME,
    fit_method=FIT_METHOD,
    debug=DEBUG,
    random_seed=RANDOM_SEED,
    touch=False,
)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[05/27/21 14:28:22] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Cache directory: <span style="color: #800080; text-decoration-color: #800080">/n/data1/hms/db</span> <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/sampling_pymc3_models_cli.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sampling_pymc3_models_cli.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:87</span>
                             <span style="color: #800080; text-decoration-color: #800080">mi/park/Cook/speclet/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">models</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Sampling in debug mode.          <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/sampling_pymc3_models_cli.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sampling_pymc3_models_cli.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:92</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Sampling <span style="color: #008000; text-decoration-color: #008000">'speclet-three'</span> with    <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/sampling_pymc3_models_cli.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sampling_pymc3_models_cli.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:94</span>
                             custom name
                             <span style="color: #008000; text-decoration-color: #008000">'SpecletThree-kras-debug'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Including KRAS allele covariate in the        <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/cli_helpers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">cli_helpers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:136</span>
                             model.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Changing `kras_cov` attribute to <span style="color: #008000; text-decoration-color: #008000">'True'</span>.     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_three.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_three.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:92</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Reseting all model and results.             <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:131</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Running model build method.     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/sampling_pymc3_models_cli.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sampling_pymc3_models_cli.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:102</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Calling `model_specification<span style="font-weight: bold">()</span>` method.     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:171</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Beginning PyMC3 model specification.        <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_three.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_three.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:261</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping data points of sgRNA that    <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:125</span>
                             map to multiple genes.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping data points with missing     <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:131</span>
                             copy number.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Getting Theano shared variables.            <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_three.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_three.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:268</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Creating PyMC3 model <span style="font-weight: bold">(</span>non-centered          <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_three.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_three.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:277</span>
                             parameterization<span style="font-weight: bold">)</span>.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[05/27/21 14:28:36] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Running MCMC fitting method.    <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/sampling_pymc3_models_cli.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sampling_pymc3_models_cli.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:114</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> ArvizCacheManager: MCMC cache exists.      <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/cache_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">cache_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:273</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Returning results from cache.               <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:268</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> ArvizCacheManager: MCMC cache exists.      <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/managers/cache_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">cache_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:273</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> finished; execution time: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.24</span>  <a href="file:///n/data1/hms/dbmi/park/Cook/speclet/src/command_line_interfaces/sampling_pymc3_models_cli.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sampling_pymc3_models_cli.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:134</span>
                             minutes
</pre>

```python
if FIT_METHOD == "ADVI":
    model_az, advi_approx = speclet_model.advi_results
else:
    model_az = speclet_model.mcmc_results
```

### Data

```python
data = speclet_model.data_manager.get_data()
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
      <th>any_deleterious</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>log2_cn</th>
      <th>z_log2_cn</th>
      <th>is_mutated</th>
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
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.861144</td>
      <td>1.584371</td>
      <td>0</td>
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
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.861144</td>
      <td>1.584371</td>
      <td>0</td>
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
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.053926</td>
      <td>0</td>
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
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.053926</td>
      <td>0</td>
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
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.053926</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>

### Cached model fit

```python
print(speclet_model.model)
```

          μ_h ~ Normal
    σ_h_log__ ~ TransformedDistribution
          μ_g ~ Normal
    σ_g_log__ ~ TransformedDistribution
          μ_b ~ Normal
    σ_b_log__ ~ TransformedDistribution
          μ_a ~ Normal
    σ_a_log__ ~ TransformedDistribution
    σ_σ_log__ ~ TransformedDistribution
      σ_log__ ~ TransformedDistribution
     h_offset ~ Normal
     g_offset ~ Normal
     b_offset ~ Normal
     a_offset ~ Normal
          σ_h ~ HalfNormal
          σ_g ~ HalfNormal
          σ_b ~ HalfNormal
          σ_a ~ HalfNormal
          σ_σ ~ HalfNormal
            σ ~ HalfNormal
            h ~ Deterministic
            g ~ Deterministic
            b ~ Deterministic
            a ~ Deterministic
            μ ~ Deterministic
          lfc ~ Normal

```python
pm.model_to_graphviz(speclet_model.model)
```

![svg](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_14_0.svg)

## Fit diagnostics

```python
if FIT_METHOD == "ADVI":
    pmanal.plot_vi_hist(advi_approx).draw()
    plt.show()
else:
    print("R-HAT")
    print(az.rhat(model_az))
    print("=" * 60)
    print("BFMI")
    print(az.bfmi(model_az))
```

    R-HAT
    <xarray.Dataset>
    Dimensions:         (a_dim_0: 49, a_dim_1: 4, a_offset_dim_0: 49, a_offset_dim_1: 4, b_dim_0: 2, b_offset_dim_0: 2, g_dim_0: 49, g_dim_1: 29, g_offset_dim_0: 49, g_offset_dim_1: 29, h_dim_0: 49, h_offset_dim_0: 49, μ_dim_0: 14162, σ_dim_0: 194)
    Coordinates: (12/14)
      * h_offset_dim_0  (h_offset_dim_0) int64 0 1 2 3 4 5 6 ... 43 44 45 46 47 48
      * g_offset_dim_0  (g_offset_dim_0) int64 0 1 2 3 4 5 6 ... 43 44 45 46 47 48
      * g_offset_dim_1  (g_offset_dim_1) int64 0 1 2 3 4 5 6 ... 23 24 25 26 27 28
      * b_offset_dim_0  (b_offset_dim_0) int64 0 1
      * a_offset_dim_0  (a_offset_dim_0) int64 0 1 2 3 4 5 6 ... 43 44 45 46 47 48
      * a_offset_dim_1  (a_offset_dim_1) int64 0 1 2 3
        ...              ...
      * g_dim_0         (g_dim_0) int64 0 1 2 3 4 5 6 7 ... 41 42 43 44 45 46 47 48
      * g_dim_1         (g_dim_1) int64 0 1 2 3 4 5 6 7 ... 21 22 23 24 25 26 27 28
      * b_dim_0         (b_dim_0) int64 0 1
      * a_dim_0         (a_dim_0) int64 0 1 2 3 4 5 6 7 ... 41 42 43 44 45 46 47 48
      * a_dim_1         (a_dim_1) int64 0 1 2 3
      * μ_dim_0         (μ_dim_0) int64 0 1 2 3 4 ... 14157 14158 14159 14160 14161
    Data variables: (12/19)
        μ_h             float64 1.0
        μ_g             float64 1.0
        μ_b             float64 0.9998
        μ_a             float64 1.0
        h_offset        (h_offset_dim_0) float64 1.001 1.002 1.004 ... 1.0 1.001 1.0
        g_offset        (g_offset_dim_0, g_offset_dim_1) float64 0.9998 ... 1.0
        ...              ...
        σ               (σ_dim_0) float64 0.9999 0.9999 0.9998 ... 1.0 1.0 0.9998
        h               (h_dim_0) float64 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
        g               (g_dim_0, g_dim_1) float64 1.0 0.9998 0.9999 ... 1.0 1.0
        b               (b_dim_0) float64 1.001 1.001
        a               (a_dim_0, a_dim_1) float64 1.0 1.0 0.9999 ... 1.0 1.0 0.9999
        μ               (μ_dim_0) float64 1.0 1.0 0.9999 0.9999 ... 1.0 1.0 1.0 1.0
    ============================================================
    BFMI
    [0.60715043 0.60715043 0.60715043 0.60715043]

## Model parameters

```python
def check_shape(trace: np.ndarray) -> np.ndarray:
    if len(trace.shape) == 1:
        return trace[:, None]
    return trace


def add_hdi(p: gg.ggplot, values: np.ndarray, color: str) -> gg.ggplot:
    m = np.mean(values)
    hdi = az.hdi(values, hdi_prob=0.89).flatten()
    p = (
        p
        + gg.geom_vline(xintercept=m, color=color)
        + gg.geom_vline(xintercept=hdi, color=color, linetype="--")
    )
    return p


def variable_distribution_plot(var, trace: np.ndarray, max_plot=20000) -> gg.ggplot:
    trace = check_shape(trace)

    # Sample 25% of the trace.
    d = pd.DataFrame(trace).melt().assign(variable=lambda d: d.variable.astype("str"))
    d_summaries = d.groupby(["variable"])["value"].mean().reset_index(drop=False)

    if d.shape[0] > max_plot:
        d = d.sample(n=max_plot)
    else:
        d = d.sample(frac=0.2)

    p = (
        gg.ggplot(d, gg.aes(x="value"))
        + gg.geom_density(alpha=0.1)
        + gg.geom_vline(xintercept=0, color="black", size=0.7, alpha=0.7, linetype="--")
        + gg.scale_x_continuous(expand=(0, 0))
        + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
        + gg.theme(legend_position="none", figure_size=(6.5, 3))
        + gg.labs(x="posterior", y="density", title=f"Posterior distirbution of {var}")
    )

    c = SeabornColor.blue

    if len(d_summaries) > 1:
        p = p + gg.geom_rug(
            data=d_summaries, sides="b", alpha=0.5, color=c, length=0.08
        )
    else:
        p = add_hdi(p, trace.flatten(), color=c)

    return p
```

```python
ignore_vars = "μ"
vars_to_inspect = model_az.posterior.keys()
vars_to_inspect = [v for v in vars_to_inspect if not "log" in v]
vars_to_inspect.sort()

for var in vars_to_inspect:
    trace = model_az.posterior[var]
    if trace.shape[1] == data.shape[0]:
        # Do not plot the final deterministic mean (usually "μ").
        continue
    try:
        print(variable_distribution_plot(var, model_az.posterior[var].values.flatten()))
    except Exception as err:
        print(f"Skipping variable '{var}'.")
        print(err)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_0.png)

    <ggplot: (2981242871637)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_2.png)

    <ggplot: (2981241050254)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_4.png)

    <ggplot: (2981231176319)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_6.png)

    <ggplot: (2981220624731)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_8.png)

    <ggplot: (2981225008517)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_10.png)

    <ggplot: (2981224806096)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_12.png)

    <ggplot: (2981225000349)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_14.png)

    <ggplot: (2981225000238)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_16.png)

    <ggplot: (2981224790703)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_18.png)

    <ggplot: (2981224796983)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_20.png)

    <ggplot: (2981224784138)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_22.png)

    <ggplot: (2981224660474)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_24.png)

    <ggplot: (2981224670492)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_26.png)

    <ggplot: (2981223868880)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_28.png)

    <ggplot: (2981243429931)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_30.png)

    <ggplot: (2981222392155)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_32.png)

    <ggplot: (2981257931831)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_34.png)

    <ggplot: (2981243429991)>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_19_36.png)

    <ggplot: (2981239013783)>

```python
if isinstance(speclet_model, SpecletTestModel):
    raise KeyboardInterrupt()
```

## Model predictions

```python
predictions = model_az.posterior_predictive
pred_summary = pmanal.summarize_posterior_predictions(
    predictions["lfc"].values,
    merge_with=data,
    calc_error=True,
    observed_y="lfc",
)
pred_summary.head()
```

    /n/data1/hms/dbmi/park/Cook/speclet/.snakemake/conda/7988df32/lib/python3.9/site-packages/arviz/stats/stats.py:456: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions

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
      <th>pred_mean</th>
      <th>pred_hdi_low</th>
      <th>pred_hdi_high</th>
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>pdna_batch</th>
      <th>passes_qc</th>
      <th>depmap_id</th>
      <th>primary_or_metastasis</th>
      <th>...</th>
      <th>variant_classification</th>
      <th>is_deleterious</th>
      <th>is_tcga_hotspot</th>
      <th>is_cosmic_hotspot</th>
      <th>mutated_at_guide_location</th>
      <th>rna_expr</th>
      <th>log2_cn</th>
      <th>z_log2_cn</th>
      <th>is_mutated</th>
      <th>error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.145060</td>
      <td>-0.603216</td>
      <td>0.849075</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repa_p6_batch2</td>
      <td>0.029491</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.861144</td>
      <td>1.584371</td>
      <td>0</td>
      <td>-0.115569</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.144838</td>
      <td>-0.551890</td>
      <td>0.846379</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>ls513-311cas9_repb_p6_batch2</td>
      <td>0.426017</td>
      <td>2</td>
      <td>True</td>
      <td>ACH-000007</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.480265</td>
      <td>1.861144</td>
      <td>1.584371</td>
      <td>0</td>
      <td>0.281179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.057998</td>
      <td>-0.788572</td>
      <td>0.648532</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep a p5_batch3</td>
      <td>0.008626</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.053926</td>
      <td>0</td>
      <td>0.066625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.060519</td>
      <td>-0.784735</td>
      <td>0.688524</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep b p5_batch3</td>
      <td>0.280821</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.053926</td>
      <td>0</td>
      <td>0.341340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.073475</td>
      <td>-0.804272</td>
      <td>0.609349</td>
      <td>CCACCCACAGACGCTCAGCA</td>
      <td>c2bbe1-311cas9 rep c p5_batch3</td>
      <td>0.239815</td>
      <td>3</td>
      <td>True</td>
      <td>ACH-000009</td>
      <td>Primary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0.695994</td>
      <td>1.375470</td>
      <td>-0.053926</td>
      <td>0</td>
      <td>0.313291</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>

```python
try:
    az.plot_loo_pit(model_az, y="lfc")
except Exception as e:
    print(e)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_23_0.png)

```python
model_loo = az.loo(model_az, pointwise=True)
print(model_loo)
```

    Computed from 16000 by 14162 log-likelihood matrix

             Estimate       SE
    elpd_loo -7749.93   111.61
    p_loo     1202.03        -

    There has been a warning during the calculation. Please check the results.
    ------

    Pareto k diagnostic values:
                             Count   Pct.
    (-Inf, 0.5]   (good)     13978   98.7%
     (0.5, 0.7]   (ok)         159    1.1%
       (0.7, 1]   (bad)         23    0.2%
       (1, Inf)   (very bad)     2    0.0%

```python
sns.distplot(model_loo.loo_i.values);
```

    /n/data1/hms/dbmi/park/Cook/speclet/.snakemake/conda/7988df32/lib/python3.9/site-packages/seaborn/distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).





    <AxesSubplot:ylabel='Density'>

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_25_2.png)

```python
pred_summary["loo"] = model_loo.loo_i.values
```

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="pred_mean"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.geom_abline(slope=1, intercept=0, size=1, alpha=0.7, color="grey")
    + gg.geom_smooth(method="glm", color=SeabornColor.red, size=1, alpha=0.7, se=False)
    + gg.labs(x="observed LFC", y="prediticed LFC (posterior avg.)")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_27_0.png)

    <ggplot: (2981543500442)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="loo"))
    + gg.geom_point(gg.aes(color="np.abs(error)"), alpha=0.5)
    + gg.scale_color_gradient(low="grey", high="red")
    + gg.theme()
    + gg.labs(x="observed LFC", y="LOO", color="abs(error)")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_28_0.png)

    <ggplot: (2981543454705)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="np.abs(error)", y="loo"))
    + gg.geom_point(gg.aes(color="lfc"), alpha=0.5)
    + gg.labs(x="abs(error)", y="loo", color="LFC")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_29_0.png)

    <ggplot: (2981210034318)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="lfc", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7)
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7)
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="observed LFC", y="prediction error")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_30_0.png)

    <ggplot: (2981266781712)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="hugo_symbol", y="loo"))
    + gg.geom_point(alpha=0.2, size=0.7)
    + gg.geom_boxplot(outlier_alpha=0, alpha=0.4)
    + gg.theme(axis_text_x=gg.element_blank(), axis_ticks_major_x=gg.element_blank())
    + gg.labs(x="genes", y="LOO")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_31_0.png)

    <ggplot: (2982981383693)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="depmap_id", y="loo"))
    + gg.geom_jitter(width=0.2, alpha=0.3, size=0.7)
    + gg.geom_boxplot(outlier_alpha=0, alpha=0.4)
    + gg.theme(
        axis_text_x=gg.element_text(angle=90, size=8),
    )
    + gg.labs(x="cell lines", y="LOO")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_32_0.png)

    <ggplot: (2981543271771)>

```python
# Remove samples without gene CN data.
ppc_df_no_missing = pred_summary.copy()[~pred_summary.gene_cn.isna()]
ppc_df_no_missing["binned_gene_cn"] = [
    np.min([round(x), 10]) for x in ppc_df_no_missing.gene_cn
]

(
    gg.ggplot(ppc_df_no_missing, gg.aes(x="factor(binned_gene_cn)", y="loo"))
    + gg.geom_jitter(size=0.6, alpha=0.5, width=0.3)
    + gg.geom_boxplot(outlier_alpha=0, alpha=0.8)
    + gg.labs(x="gene copy number (max. 10)", y="LOO")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_33_0.png)

    <ggplot: (2981543276120)>

```python
gene_error = (
    pred_summary.groupby(["hugo_symbol"])["error"]
    .agg([np.mean, np.std])
    .reset_index(drop=False)
    .sort_values(["mean"])
    .reset_index(drop=True)
    .assign(
        hugo_symbol=lambda d: pd.Categorical(
            d.hugo_symbol.astype(str),
            categories=d.hugo_symbol.astype(str),
            ordered=True,
        )
    )
)

n_genes = 15

(
    gg.ggplot(
        gene_error.iloc[list(range(n_genes)) + list(range(-n_genes, -1))],
        gg.aes(x="hugo_symbol", y="mean"),
    )
    + gg.geom_col()
    + gg.theme(axis_text_x=gg.element_text(angle=90))
    + gg.labs(x="gene", y="error", title="Genes with the highest average error")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_34_0.png)

    <ggplot: (2981231745470)>

```python
(
    gg.ggplot(pred_summary, gg.aes(x="log2_cn", y="error"))
    + gg.geom_hline(yintercept=0, size=0.5, alpha=0.7, linetype="--")
    + gg.geom_vline(xintercept=0, size=0.5, alpha=0.7, linetype="--")
    + gg.geom_point(size=0.1, alpha=0.2)
    + gg.labs(x="gene copy number (log2)", y="predition error")
)
```

![png](speclet-three_SpecletThree-kras-debug_MCMC_files/speclet-three_SpecletThree-kras-debug_MCMC_35_0.png)

    <ggplot: (2981543355293)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 15.83 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-05-27

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.21.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-178.o2.rc.hms.harvard.edu

    Git branch: noncentered-reparam

    arviz     : 0.11.2
    numpy     : 1.20.1
    pymc3     : 3.11.1
    pandas    : 1.2.3
    seaborn   : 0.11.1
    matplotlib: 3.3.4
    plotnine  : 0.7.1
