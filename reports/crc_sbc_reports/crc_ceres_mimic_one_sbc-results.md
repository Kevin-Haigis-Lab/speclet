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

from src.command_line_interfaces import simulation_based_calibration_cli as sbc_cli
from src.context_managers import set_directory
from src.data_processing import common as dphelp
from src.modeling import pymc3_analysis as pmanal
from src.modeling import pymc3_sampling_api as pmapi
from src.modeling import sampling_pymc3_models as sampling
from src.modeling.sampling_pymc3_models import SamplingArguments
from src.modeling.simulation_based_calibration_helpers import SBCFileManager
from src.plot.color_pal import SeabornColor

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("..", "models", "modeling_cache", "pymc3_model_cache")
```

Parameters for papermill:

- `MODEL`: which model was tested
- `SBC_RESULTS_DIR`: directory containing results of many rounds of SBC
- `NUM_SIMULATIONS`: the number of simiulations; will be used to check that all results are found

## Setup

### Papermill parameters

```python
MODEL = ""
SBC_RESULTS_DIR = ""
NUM_SIMULATIONS = -1
```

```python
# Parameters
MODEL = "crc_ceres_mimic_one"
SBC_RESULTS_DIR = "temp/crc_ceres_mimic_one"
NUM_SIMULATIONS = 5

```

### Prepare and validate papermill parameters

Build the model using the `MODEL` parameter.

```python
ModelClass = sbc_cli.get_model_class(sbc_cli.ModelOption[MODEL])
```

Check values passed as the directory with results of the rounds of SBC.

```python
sbc_results_dir = Path("../..", SBC_RESULTS_DIR)
assert sbc_results_dir.is_dir()
assert sbc_results_dir.exists()
```

Confirm that there is a positive number of simulations.

```python
assert NUM_SIMULATIONS > 0
```

## Read in all results

```python
for sbc_dir in sbc_results_dir.iterdir():
    sbc_fm = SBCFileManager(sbc_dir)
    if not sbc_fm.all_data_exists():
        raise Exception(f"Not all output from '{sbc_fm.dir.name}' exist.")
    res = sbc_fm.get_sbc_results()
```

```python
res.posterior_summary
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
    <tr>
      <th>parameter</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μ_h</th>
      <td>0.404</td>
      <td>0.338</td>
      <td>-0.198</td>
      <td>1.046</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>1102.0</td>
      <td>877.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ_d</th>
      <td>0.112</td>
      <td>0.118</td>
      <td>-0.112</td>
      <td>0.329</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>875.0</td>
      <td>981.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ_η</th>
      <td>0.133</td>
      <td>0.205</td>
      <td>-0.285</td>
      <td>0.482</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>990.0</td>
      <td>1013.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>h[0]</th>
      <td>0.750</td>
      <td>0.691</td>
      <td>-0.406</td>
      <td>2.187</td>
      <td>0.022</td>
      <td>0.016</td>
      <td>989.0</td>
      <td>934.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>h[1]</th>
      <td>0.250</td>
      <td>0.714</td>
      <td>-1.015</td>
      <td>1.602</td>
      <td>0.023</td>
      <td>0.017</td>
      <td>972.0</td>
      <td>983.0</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>μ[146]</th>
      <td>0.964</td>
      <td>0.881</td>
      <td>-0.618</td>
      <td>2.598</td>
      <td>0.028</td>
      <td>0.020</td>
      <td>966.0</td>
      <td>992.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ[147]</th>
      <td>1.583</td>
      <td>0.920</td>
      <td>-0.241</td>
      <td>3.234</td>
      <td>0.028</td>
      <td>0.020</td>
      <td>1096.0</td>
      <td>980.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ[148]</th>
      <td>1.578</td>
      <td>0.923</td>
      <td>-0.028</td>
      <td>3.520</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>1002.0</td>
      <td>941.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ[149]</th>
      <td>1.562</td>
      <td>0.887</td>
      <td>-0.040</td>
      <td>3.361</td>
      <td>0.027</td>
      <td>0.020</td>
      <td>1060.0</td>
      <td>977.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>2.567</td>
      <td>0.196</td>
      <td>2.260</td>
      <td>2.997</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>854.0</td>
      <td>856.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>271 rows × 9 columns</p>
</div>

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

    execution time: 0.03 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-04-01

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.21.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 32
    Architecture: 64bit

    Hostname: compute-a-16-163.o2.rc.hms.harvard.edu

    Git branch: simulation-based-calibration

    pymc3     : 3.11.1
    arviz     : 0.11.2
    plotnine  : 0.7.1
    matplotlib: 3.3.4
    numpy     : 1.20.1
    pandas    : 1.2.3
    seaborn   : 0.11.1
