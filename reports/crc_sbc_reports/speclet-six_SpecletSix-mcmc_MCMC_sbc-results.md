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

- `MODEL`: which model was tested
- `MODEL_NAME`: unique, identifiable name of the model
- `SBC_RESULTS_DIR`: directory containing results of many rounds of SBC
- `NUM_SIMULATIONS`: the number of simiulations; will be used to check that all results are found

## Setup

### Papermill parameters

```python
MODEL = ""
MODEL_NAME = ""
SBC_RESULTS_DIR = ""
SBC_COLLATED_RESULTS = ""
NUM_SIMULATIONS = -1
```

```python
# Parameters
MODEL = "speclet-six"
MODEL_NAME = "SpecletSix-mcmc"
SBC_RESULTS_DIR = (
    "/n/scratch3/users/j/jc604/speclet-sbc/speclet-six_SpecletSix-mcmc_MCMC"
)
SBC_COLLATED_RESULTS = (
    "cache/sbc-cache/speclet-six_SpecletSix-mcmc_MCMC_collated-posterior-summaries.pkl"
)
NUM_SIMULATIONS = 5

```

### Prepare and validate papermill parameters

Build the model using the `MODEL` parameter.

```python
ModelClass = cli_helpers.get_model_class(cli_helpers.ModelOption(MODEL))
```

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
      <th>μ_j</th>
      <th>μ_j</th>
      <td>0.073</td>
      <td>0.835</td>
      <td>-1.301</td>
      <td>1.381</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>12165.0</td>
      <td>11762.0</td>
      <td>1.0</td>
      <td>-1.347759</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>j_offset[0]</th>
      <th>j_offset</th>
      <td>-0.492</td>
      <td>0.384</td>
      <td>-1.097</td>
      <td>0.131</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>9206.0</td>
      <td>9653.0</td>
      <td>1.0</td>
      <td>-0.635242</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>j_offset[1]</th>
      <th>j_offset</th>
      <td>0.508</td>
      <td>0.379</td>
      <td>-0.086</td>
      <td>1.110</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>10313.0</td>
      <td>10822.0</td>
      <td>1.0</td>
      <td>0.484698</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>μ_h</th>
      <th>μ_h</th>
      <td>0.025</td>
      <td>0.481</td>
      <td>-0.719</td>
      <td>0.809</td>
      <td>0.004</td>
      <td>0.004</td>
      <td>15481.0</td>
      <td>12116.0</td>
      <td>1.0</td>
      <td>-0.447733</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>h_offset[0,0]</th>
      <th>h_offset</th>
      <td>0.497</td>
      <td>0.383</td>
      <td>-0.107</td>
      <td>1.117</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>10870.0</td>
      <td>11039.0</td>
      <td>1.0</td>
      <td>0.193451</td>
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

![png](speclet-six_SpecletSix-mcmc_MCMC_sbc-results_files/speclet-six_SpecletSix-mcmc_MCMC_sbc-results_17_0.png)

    <ggplot: (2968649914290)>

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

![png](speclet-six_SpecletSix-mcmc_MCMC_sbc-results_files/speclet-six_SpecletSix-mcmc_MCMC_sbc-results_18_0.png)

    <ggplot: (2968650249978)>

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 0.16 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-06-30

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

    Hostname: compute-a-16-53.o2.rc.hms.harvard.edu

    Git branch: sbc-run-limited

    matplotlib: 3.3.4
    numpy     : 1.20.1
    pymc3     : 3.11.1
    plotnine  : 0.7.1
    janitor   : 0.20.14
    re        : 2.2.1
    seaborn   : 0.11.1
    arviz     : 0.11.2
    pandas    : 1.2.3
