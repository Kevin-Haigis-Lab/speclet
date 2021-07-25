# Model SBC Report

```python
import logging
import warnings
from pathlib import Path
from time import time
from typing import List

import arviz as az
import janitor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg

from src.loggers import set_console_handler_level
from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.modeling import pymc3_analysis as pmanal
from src.modeling import simulation_based_calibration_helpers as sbc
from src.project_enums import ModelFitMethod
```

```python
notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(gg.theme_classic())
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 847
np.random.seed(RANDOM_SEED)

set_console_handler_level(logging.WARNING)
pymc3_cache_dir = Path("..", "models", "modeling_cache", "pymc3_model_cache")
```

Parameters for papermill:

- `MODEL_NAME`: unique, identifiable name of the model
- `SBC_RESULTS_DIR`: directory containing results of many rounds of SBC
- `SBC_COLLATED_RESULTS`: path to collated simulation posteriors
- `NUM_SIMULATIONS`: the number of simiulations; will be used to check that all results are found
- `CONFIG_PATH`: path to the model configuration file
- `FIT_METHOD`: model fitting method used for this SBC

## Setup

### Papermill parameters

```python
MODEL_NAME = ""
SBC_RESULTS_DIR = ""
SBC_COLLATED_RESULTS = ""
NUM_SIMULATIONS = -1
CONFIG_PATH = ""
FIT_METHOD_STR = ""
```

```python
# Parameters
MODEL_NAME = "sp5-default"
SBC_RESULTS_DIR = "/n/scratch3/users/j/jc604/speclet-sbc/sp5-default_ADVI"
SBC_COLLATED_RESULTS = (
    "cache/sbc-cache/sp5-default_ADVI_collated-posterior-summaries.pkl"
)
NUM_SIMULATIONS = 25
CONFIG_PATH = "models/model-configs.yaml"
FIT_METHOD_STR = "ADVI"

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

```python
FIT_METHOD = ModelFitMethod(FIT_METHOD_STR)
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
      <td>0.108</td>
      <td>0.198</td>
      <td>-0.198</td>
      <td>0.424</td>
      <td>0.006</td>
      <td>0.005</td>
      <td>1039.0</td>
      <td>912.0</td>
      <td>NaN</td>
      <td>-0.088185</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>j[0]</th>
      <th>j</th>
      <td>0.210</td>
      <td>0.020</td>
      <td>0.179</td>
      <td>0.241</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>994.0</td>
      <td>983.0</td>
      <td>NaN</td>
      <td>-0.190428</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>j[1]</th>
      <th>j</th>
      <td>1.330</td>
      <td>0.030</td>
      <td>1.284</td>
      <td>1.379</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1049.0</td>
      <td>1016.0</td>
      <td>NaN</td>
      <td>-0.044252</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>j[2]</th>
      <th>j</th>
      <td>0.273</td>
      <td>0.029</td>
      <td>0.231</td>
      <td>0.320</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>903.0</td>
      <td>873.0</td>
      <td>NaN</td>
      <td>0.074007</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>μ_h</th>
      <th>μ_h</th>
      <td>0.130</td>
      <td>0.100</td>
      <td>-0.026</td>
      <td>0.295</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>793.0</td>
      <td>948.0</td>
      <td>NaN</td>
      <td>-0.341254</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>

## Analysis

### ADVI approximation histories

```python
if FIT_METHOD is ModelFitMethod.ADVI:
    advi_histories: List[np.ndarray] = []

    for dir in sbc_results_dir.iterdir():
        if not dir.is_dir():
            continue

        cache_manager = Pymc3ModelCacheManager(name=MODEL_NAME, root_cache_dir=dir)
        if cache_manager.advi_cache_exists():
            _, advi_approx = cache_manager.get_advi_cache()
            advi_histories.append(advi_approx.hist)
    n_sims_advi_hist = min(NUM_SIMULATIONS, 5)
    sample_hist_idxs = np.random.choice(
        list(range(len(advi_histories))), size=n_sims_advi_hist, replace=False
    )

    def make_hist_df(sim_idx: int, hist_list: List[np.ndarray]) -> pd.DataFrame:
        df = pd.DataFrame({"sim_idx": sim_idx, "loss": hist_list[sim_idx].flatten()})
        df["step"] = np.arange(df.shape[0])
        return df

    sampled_advi_histories = pd.concat(
        [make_hist_df(i, advi_histories) for i in sample_hist_idxs]
    ).reset_index(drop=True)

    (
        gg.ggplot(
            sampled_advi_histories,
            gg.aes(x="step", y="np.log(loss)", color="factor(sim_idx)"),
        )
        + gg.geom_line(alpha=0.5)
        + gg.scale_color_brewer(type="qual", palette="Set1")
        + gg.scale_x_continuous(expand=(0, 0))
        + gg.scale_y_continuous(expand=(0.01, 0, 0.02, 0))
        + gg.theme(legend_position=(0.8, 0.5))
        + gg.labs(y="log loss", color="sim. idx.")
    ).draw()
    plt.show()
```

    /n/data1/hms/dbmi/park/Cook/speclet/.snakemake/conda/daab5ac56849df6113f96e98614e98ee/lib/python3.9/site-packages/pandas/core/arraylike.py:358: RuntimeWarning: invalid value encountered in log

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_18_1.png)

### MCMC diagnostics

```python
class IncompleteCachedResultsWarning(UserWarning):
    pass


all_sbc_perm_dirs = list(sbc_results_dir.iterdir())

for perm_dir in np.random.choice(
    all_sbc_perm_dirs, size=min([5, len(all_sbc_perm_dirs)]), replace=False
):
    print(perm_dir.name)
    print("-" * 30)
    sbc_fm = sbc.SBCFileManager(perm_dir)
    if sbc_fm.all_data_exists():
        sbc_res = sbc_fm.get_sbc_results()
        _ = pmanal.describe_mcmc(sbc_res.inference_obj)
    else:
        warnings.warn(
            "Cannot find all components of the SBC results.",
            IncompleteCachedResultsWarning,
        )
```

    sbc-perm21
    ------------------------------
    Unable to get sampling stats.
    sbc-perm20
    ------------------------------
    Unable to get sampling stats.
    sbc-perm13
    ------------------------------
    Unable to get sampling stats.
    sbc-perm3
    ------------------------------
    Unable to get sampling stats.
    sbc-perm18
    ------------------------------
    Unable to get sampling stats.

### Estimate accuracy

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

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_22_0.png)

    <ggplot: (2983931521802)>

```python
hdi_low, hdi_high = pmanal.get_hdi_colnames_from_az_summary(simulation_posteriors_df)


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

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_23_0.png)

    <ggplot: (2983932368656)>

### SBC Uniformity Test

```python
sbc_analyzer = sbc.SBCAnalysis(
    root_dir=sbc_results_dir, pattern="sbc-perm", n_simulations=NUM_SIMULATIONS
)
```

```python
K_DRAWS = 100
sbc_uniformity_test = sbc_analyzer.uniformity_test(k_draws=K_DRAWS)
sbc_uniformity_test.head()
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
      <th>parameter</th>
      <th>rank_stat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>μ_j</td>
      <td>92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>j[0]</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>j[1]</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>j[2]</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>μ_h</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>

```python
var_names = sbc_uniformity_test.parameter.unique().tolist()
var_names = [v for v in var_names if "μ" not in v]
for v in np.random.choice(var_names, size=min((10, len(var_names))), replace=False):
    ax = sbc_analyzer.plot_uniformity(
        sbc_uniformity_test.query(f"parameter == '{v}'"), k_draws=K_DRAWS
    )
    ax.set_title(v)
    plt.show()
```

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_0.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_1.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_2.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_3.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_4.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_5.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_6.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_7.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_8.png)

![png](sp5-default_ADVI_sbc-results_files/sp5-default_ADVI_sbc-results_27_9.png)

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 1.53 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-07-25

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

    Hostname: compute-a-16-170.o2.rc.hms.harvard.edu

    Git branch: sbc-uniform-check

    janitor   : 0.20.14
    plotnine  : 0.7.1
    arviz     : 0.11.2
    matplotlib: 3.3.4
    numpy     : 1.20.1
    pandas    : 1.2.3
    logging   : 0.5.1.2
