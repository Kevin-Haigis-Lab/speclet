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
MODEL = "crc_model_one"
SBC_RESULTS_DIR = "temp/crc_model_one"
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
      <th>μ_g</th>
      <td>0.144</td>
      <td>0.338</td>
      <td>-0.519</td>
      <td>0.731</td>
      <td>0.011</td>
      <td>0.008</td>
      <td>966.0</td>
      <td>902.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ_α[0]</th>
      <td>0.571</td>
      <td>0.319</td>
      <td>0.040</td>
      <td>1.232</td>
      <td>0.010</td>
      <td>0.007</td>
      <td>1023.0</td>
      <td>906.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ_α[1]</th>
      <td>1.642</td>
      <td>0.460</td>
      <td>0.760</td>
      <td>2.475</td>
      <td>0.016</td>
      <td>0.011</td>
      <td>875.0</td>
      <td>947.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ_α[2]</th>
      <td>1.486</td>
      <td>0.397</td>
      <td>0.709</td>
      <td>2.200</td>
      <td>0.013</td>
      <td>0.009</td>
      <td>1009.0</td>
      <td>963.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ_α[3]</th>
      <td>-1.373</td>
      <td>0.534</td>
      <td>-2.309</td>
      <td>-0.304</td>
      <td>0.017</td>
      <td>0.012</td>
      <td>1037.0</td>
      <td>852.0</td>
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
      <td>1.477</td>
      <td>0.236</td>
      <td>1.039</td>
      <td>1.906</td>
      <td>0.008</td>
      <td>0.005</td>
      <td>941.0</td>
      <td>944.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ[147]</th>
      <td>1.365</td>
      <td>0.227</td>
      <td>0.957</td>
      <td>1.800</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>923.0</td>
      <td>941.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ[148]</th>
      <td>2.071</td>
      <td>0.233</td>
      <td>1.614</td>
      <td>2.505</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>1060.0</td>
      <td>895.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>μ[149]</th>
      <td>1.487</td>
      <td>0.231</td>
      <td>1.069</td>
      <td>1.924</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>971.0</td>
      <td>905.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>0.392</td>
      <td>0.027</td>
      <td>0.343</td>
      <td>0.441</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1090.0</td>
      <td>935.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>215 rows × 9 columns</p>
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

    matplotlib: 3.3.4
    pandas    : 1.2.3
    plotnine  : 0.7.1
    numpy     : 1.20.1
    pymc3     : 3.11.1
    seaborn   : 0.11.1
    arviz     : 0.11.2
