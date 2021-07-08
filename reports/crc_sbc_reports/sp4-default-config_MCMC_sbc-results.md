# Model Report

```python
import re
import warnings
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import arviz as az
import janitor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns

from src.command_line_interfaces import cli_helpers
from src.modeling.pymc3_analysis import get_hdi_colnames_from_az_summary
from src.modeling.simulation_based_calibration_helpers import SBCFileManager

notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

pymc3_cache_dir = Path("..", "models", "modeling_cache", "pymc3_model_cache")
```

Parameters for papermill:

- `MODEL_NAME`: unique, identifiable name of the model
- `SBC_RESULTS_DIR`: directory containing results of many rounds of SBC
- `SBC_COLLATED_RESULTS`: path to collated simulation posteriors
- `NUM_SIMULATIONS`: the number of simiulations; will be used to check that all results are found
- `CONFIG_PATH`: path to the model configuration file

## Setup

### Papermill parameters

```python
MODEL_NAME = ""
SBC_RESULTS_DIR = ""
SBC_COLLATED_RESULTS = ""
NUM_SIMULATIONS = -1
CONFIG_PATH = ""
```

```python
# Parameters
MODEL_NAME = "sp4-default-config"
SBC_RESULTS_DIR = "/n/scratch3/users/j/jc604/speclet-sbc/sp4-default-config_MCMC"
SBC_COLLATED_RESULTS = (
    "cache/sbc-cache/sp4-default-config_MCMC_collated-posterior-summaries.pkl"
)
NUM_SIMULATIONS = 2
CONFIG_PATH = "models/model-configs.yaml"

```

### Prepare and validate papermill parameters

Check values passed as the directory with results of the rounds of SBC.

```python
path_addition = "../.."

sbc_results_dir = Path(path_addition, SBC_RESULTS_DIR)
assert sbc_results_dir.is_dir()
assert sbc_results_dir.exists()

sbc_collated_results_path = Path(path_addition, SBC_COLLATED_RESULTS)
assert sbc_collated_results_path.is_file()
assert sbc_collated_results_path.exists()
```

Confirm that there is a positive number of simulations.

```python
assert NUM_SIMULATIONS > 0
```

## Read in all results

```python
simulation_posteriors_df = pd.read_pickle(sbc_collated_results_path)
simulation_posteriors_df.head()
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
      <th>true_value</th>
      <th>simulation_id</th>
      <th>within_hdi</th>
    </tr>
    <tr>
      <th>parameter</th>
      <th>parameter_name</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>μ_h</th>
      <td>2.099</td>
      <td>0.282</td>
      <td>1.635</td>
      <td>2.536</td>
      <td>0.082</td>
      <td>0.060</td>
      <td>12.0</td>
      <td>52.0</td>
      <td>1.25</td>
      <td>1.764052</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>μ_d</th>
      <th>μ_d</th>
      <td>0.128</td>
      <td>0.281</td>
      <td>-0.352</td>
      <td>0.534</td>
      <td>0.102</td>
      <td>0.075</td>
      <td>8.0</td>
      <td>24.0</td>
      <td>1.47</td>
      <td>0.978738</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>μ_η</th>
      <th>μ_η</th>
      <td>0.033</td>
      <td>0.180</td>
      <td>-0.259</td>
      <td>0.314</td>
      <td>0.034</td>
      <td>0.024</td>
      <td>30.0</td>
      <td>69.0</td>
      <td>1.10</td>
      <td>0.080031</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>h[0]</th>
      <th>h</th>
      <td>2.634</td>
      <td>0.460</td>
      <td>1.973</td>
      <td>3.332</td>
      <td>0.110</td>
      <td>0.079</td>
      <td>17.0</td>
      <td>38.0</td>
      <td>1.23</td>
      <td>1.653345</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>h[1]</th>
      <th>h</th>
      <td>2.168</td>
      <td>0.558</td>
      <td>1.291</td>
      <td>2.995</td>
      <td>0.116</td>
      <td>0.083</td>
      <td>25.0</td>
      <td>61.0</td>
      <td>1.12</td>
      <td>1.671294</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

## Analysis

```python
accuracy_per_parameter = (
    simulation_posteriors_df.copy()
    .groupby(["parameter_name"])["within_hdi"]
    .mean()
    .reset_index(drop=False)
    .sort_values("within_hdi", ascending=False)
    .reset_index(drop=True)
)

accuracy_per_parameter["parameter_name"] = pd.Categorical(
    accuracy_per_parameter["parameter_name"],
    categories=accuracy_per_parameter["parameter_name"].values,
)

(
    gg.ggplot(accuracy_per_parameter, gg.aes(x="parameter_name", y="within_hdi"))
    + gg.geom_col()
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.labs(
        x="parameter",
        y="freq. of true value in 89% HDI",
        title="Average accuracy of each parameter",
    )
    + gg.theme(axis_ticks_major_x=gg.element_blank(), figure_size=(6, 4))
)
```

![png](sp4-default-config_MCMC_sbc-results_files/sp4-default-config_MCMC_sbc-results_15_0.png)

    <ggplot: (2943410513824)>

```python
hdi_low, hdi_high = get_hdi_colnames_from_az_summary(simulation_posteriors_df)


def filter_uninsteresting_parameters(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.reset_index(drop=False)
        .query("parameter_name != 'μ'")
        .filter_string("parameter_name", search_string="offset", complement=True)
    )


(
    gg.ggplot(
        filter_uninsteresting_parameters(simulation_posteriors_df),
        gg.aes(x="true_value", y="mean", color="within_hdi"),
    )
    + gg.facet_wrap("~ parameter_name", ncol=3, scales="free")
    + gg.geom_linerange(gg.aes(ymin=hdi_low, ymax=hdi_high), alpha=0.2, size=0.2)
    + gg.geom_point(size=0.3, alpha=0.3)
    + gg.geom_abline(slope=1, intercept=0, linetype="--")
    + gg.scale_color_brewer(
        type="qual",
        palette="Set1",
        labels=("outside", "inside"),
        guide=gg.guide_legend(
            title="within HDI",
            override_aes={"alpha": 1, "size": 1},
        ),
    )
    + gg.theme(
        figure_size=(10, 20),
        strip_background=gg.element_blank(),
        strip_text=gg.element_text(face="bold"),
        panel_spacing=0.25,
    )
    + gg.labs(
        x="true value",
        y="mean of posterior",
    )
)
```

![png](sp4-default-config_MCMC_sbc-results_files/sp4-default-config_MCMC_sbc-results_16_0.png)

    <ggplot: (2943419717373)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 0.09 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-07-08

    Python implementation: CPython
    Python version       : 3.9.2
    IPython version      : 7.21.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 20
    Architecture: 64bit

    Hostname: compute-f-17-14.o2.rc.hms.harvard.edu

    Git branch: pipeline-confg

    pymc3     : 3.11.1
    janitor   : 0.20.14
    plotnine  : 0.7.1
    matplotlib: 3.3.4
    numpy     : 1.20.1
    seaborn   : 0.11.1
    arviz     : 0.11.2
    re        : 2.2.1
    pandas    : 1.2.3
