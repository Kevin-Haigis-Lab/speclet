# Model SBC Report

```python
import logging
import warnings
from pathlib import Path
from pprint import pprint
from time import time

import janitor  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg

from src.analysis import pymc3_analysis as pmanal
from src.analysis import sbc_analysis as sbcanal
from src.exceptions import CacheDoesNotExistError
from src.loggers import set_console_handler_level
from src.managers.model_cache_managers import Pymc3ModelCacheManager
from src.modeling import simulation_based_calibration_helpers as sbc
from src.project_enums import ModelFitMethod
```

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.

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
- `SBC_UNIFORMITY_RESULTS`: path to results of the uniformity test
- `NUM_SIMULATIONS`: the number of simiulations; will be used to check that all results are found
- `CONFIG_PATH`: path to the model configuration file
- `FIT_METHOD`: model fitting method used for this SBC

## Setup

### Papermill parameters

```python
MODEL_NAME = ""
SBC_RESULTS_DIR = ""
SBC_COLLATED_RESULTS = ""
SBC_UNIFORMITY_RESULTS = ""
NUM_SIMULATIONS = -1
CONFIG_PATH = ""
FIT_METHOD_STR = ""
```

```python
# Parameters
MODEL_NAME = "sp7-default"
SBC_RESULTS_DIR = "/n/no_backup2/dbmi/park/jc604/speclet-sbc/sp7-default_ADVI"
SBC_COLLATED_RESULTS = (
    "cache/sbc-cache/sp7-default_ADVI_collated-posterior-summaries.pkl"
)
SBC_UNIFORMITY_RESULTS = "cache/sbc-cache/sp7-default_ADVI_uniformity-test-results.pkl"
NUM_SIMULATIONS = 1000
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

sbc_uniformity_results_path = Path(path_addition, SBC_UNIFORMITY_RESULTS)
assert sbc_uniformity_results_path.is_file()
assert sbc_uniformity_results_path.exists()
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
# Posterior summaries dataframe.
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
      <th>μ_μ_h</th>
      <th>μ_μ_h</th>
      <td>-0.782</td>
      <td>0.230</td>
      <td>-1.122</td>
      <td>-0.383</td>
      <td>0.007</td>
      <td>0.005</td>
      <td>1068.0</td>
      <td>932.0</td>
      <td>NaN</td>
      <td>-0.936418</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>μ_h[0,0]</th>
      <th>μ_h</th>
      <td>-1.081</td>
      <td>0.259</td>
      <td>-1.486</td>
      <td>-0.697</td>
      <td>0.008</td>
      <td>0.006</td>
      <td>944.0</td>
      <td>1023.0</td>
      <td>NaN</td>
      <td>-1.044010</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>μ_h[0,1]</th>
      <th>μ_h</th>
      <td>-1.871</td>
      <td>0.150</td>
      <td>-2.105</td>
      <td>-1.638</td>
      <td>0.005</td>
      <td>0.003</td>
      <td>1025.0</td>
      <td>908.0</td>
      <td>NaN</td>
      <td>-2.110361</td>
      <td>sim_id_0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>μ_h[1,0]</th>
      <th>μ_h</th>
      <td>0.443</td>
      <td>0.269</td>
      <td>0.035</td>
      <td>0.875</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>891.0</td>
      <td>834.0</td>
      <td>NaN</td>
      <td>0.555118</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>μ_h[1,1]</th>
      <th>μ_h</th>
      <td>0.427</td>
      <td>0.149</td>
      <td>0.211</td>
      <td>0.674</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>923.0</td>
      <td>948.0</td>
      <td>NaN</td>
      <td>0.324540</td>
      <td>sim_id_0000</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Uniformity test results dataframe.
sbc_uniformity_test = pd.read_pickle(sbc_uniformity_results_path)
var_names = sbc_uniformity_test.parameter.tolist()
parameter_names = [x.split("[")[0] for x in var_names]
sbc_uniformity_test["parameter_name"] = parameter_names
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
      <th>parameter_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>μ_μ_h</td>
      <td>50</td>
      <td>μ_μ_h</td>
    </tr>
    <tr>
      <th>1</th>
      <td>μ_h[0,0]</td>
      <td>40</td>
      <td>μ_h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>μ_h[0,1]</td>
      <td>26</td>
      <td>μ_h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>μ_h[1,0]</td>
      <td>36</td>
      <td>μ_h</td>
    </tr>
    <tr>
      <th>4</th>
      <td>μ_h[1,1]</td>
      <td>78</td>
      <td>μ_h</td>
    </tr>
  </tbody>
</table>
</div>

## Analysis

```python
sbc_analyzer = sbcanal.SBCAnalysis(
    root_dir=sbc_results_dir,
    pattern="perm",
    n_simulations=NUM_SIMULATIONS,
    simulation_posteriors=simulation_posteriors_df,
    uniformity_test_results=sbc_uniformity_test,
)
```

### ADVI approximation histories

```python
if FIT_METHOD is ModelFitMethod.ADVI:
    advi_histories: list[np.ndarray] = []
    example_result_dirs = sbc_analyzer.random_simulation_directories(5)
    for dir in example_result_dirs:
        cache_manager = Pymc3ModelCacheManager(name=MODEL_NAME, root_cache_dir=dir)
        if cache_manager.advi_cache_exists():
            _, advi_approx = cache_manager.get_advi_cache()
            advi_histories.append(advi_approx.hist)
        else:
            raise CacheDoesNotExistError("Missing SBC results.")

    def make_hist_df(advi_history: np.ndarray, sim_idx: int) -> pd.DataFrame:
        df = pd.DataFrame({"sim_idx": sim_idx, "loss": advi_history.flatten()})
        df["step"] = np.arange(df.shape[0])
        return df

    sample_advi_indices = [d.name.replace("sbc-perm", "") for d in example_result_dirs]
    sampled_advi_histories = pd.concat(
        [make_hist_df(ah, i) for ah, i in zip(advi_histories, sample_advi_indices)]
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

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_20_0.png)

### MCMC diagnostics

```python
if FIT_METHOD is ModelFitMethod.MCMC:
    pprint(sbc_analyzer.mcmc_diagnostics())
    print("=" * 60)

    for perm_dir in sbc_analyzer.random_simulation_directories(5):
        print(perm_dir.name)
        print("-" * 30)
        sbc_fm = sbc.SBCFileManager(perm_dir)
        if sbc_fm.all_data_exists():
            sbc_res = sbc_fm.get_sbc_results()
            _ = pmanal.describe_mcmc(sbc_res.inference_obj)
        else:
            raise CacheDoesNotExistError("Missing SBC results.")
```

### Estimate accuracy

```python
accuracy_per_parameter = sbc_analyzer.run_posterior_accuracy_test()

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

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_24_0.png)

    <ggplot: (2958210523121)>

```python
hdi_low, hdi_high = pmanal.get_hdi_colnames_from_az_summary(simulation_posteriors_df)


def filter_uninsteresting_parameters(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.reset_index(drop=False)
        .query("parameter_name != 'μ'")
        .filter_string("parameter_name", search_string="offset", complement=True)
    )


plot_df = filter_uninsteresting_parameters(simulation_posteriors_df)
n_params = plot_df.parameter_name.nunique()
plot_height = (1 + n_params // 4) * 4

(
    gg.ggplot(plot_df, gg.aes(x="true_value", y="mean", color="within_hdi"))
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
        figure_size=(10, plot_height),
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

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_25_0.png)

    <ggplot: (2958211034980)>

### SBC Uniformity Test

```python
var_names_to_plot = (
    sbc_uniformity_test[["parameter", "parameter_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
    .sort_values(["parameter_name", "parameter"])
    .groupby("parameter_name")
    .head(3)
    .reset_index(drop=True)
    .parameter.tolist()
)
```

```python
for v in var_names_to_plot:
    ax = sbc_analyzer.plot_uniformity(
        sbc_uniformity_test.query(f"parameter == '{v}'").reset_index(drop=True)
    )
    ax.set_title(v)
    plt.show()
```

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_0.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_1.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_2.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_3.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_4.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_5.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_6.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_7.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_8.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_9.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_10.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_11.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_12.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_13.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_14.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_15.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_16.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_17.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_18.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_19.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_20.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_21.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_22.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_23.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_24.png)

![png](sp7-default_ADVI_sbc-results_files/sp7-default_ADVI_sbc-results_28_25.png)

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

    execution time: 3.25 minutes

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-09-03

    Python implementation: CPython
    Python version       : 3.9.6
    IPython version      : 7.26.0

    Compiler    : GCC 9.3.0
    OS          : Linux
    Release     : 3.10.0-1062.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 20
    Architecture: 64bit

    Hostname: compute-f-17-09.o2.rc.hms.harvard.edu

    Git branch: modeling

    logging   : 0.5.1.2
    numpy     : 1.21.2
    pandas    : 1.3.2
    plotnine  : 0.8.0
    janitor   : 0.21.0
    matplotlib: 3.4.3
