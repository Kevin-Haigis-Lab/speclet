# Pipeline benchmarks

## Setup

```python
from pathlib import Path
from typing import Optional

import janitor  # noqa: F401
import pandas as pd
import seaborn as sns

from speclet.io import project_root
```

```python
%config InlineBackend.figure_format='retina'
```

```python
benchmark_dir = project_root() / "benchmarks"
assert benchmark_dir.exists()
assert benchmark_dir.is_dir()
```

## Data Processing

Organization of benchmarks directory:

1. pipeline
2. rules
3. individual runs

> I may want to add more information to the name of the rules to keep them separate and not overwritten.
> For instance, including the date would be useful or metadata such as the data size for SBC or debug status for the fitting pipeline.

```python
list(benchmark_dir.iterdir())
```

    [PosixPath('/Users/admin/Developer/haigis-lab/speclet/benchmarks/.DS_Store'),
     PosixPath('/Users/admin/Developer/haigis-lab/speclet/benchmarks/010_010_model-fitting-pipeline')]

```python
def process_benchmark_file(bench_f: Path) -> pd.DataFrame:
    return pd.read_csv(bench_f, sep="\t").assign(
        step=bench_f.name.replace(bench_f.suffix, "")
    )


def get_benchmark_data_for_rule_dir(
    rule_d: Path, pipeline_name: str
) -> Optional[pd.DataFrame]:
    bench_dfs: list[pd.DataFrame] = [
        process_benchmark_file(b) for b in rule_d.iterdir()
    ]
    if len(bench_dfs) == 0:
        return None

    return (
        pd.concat(bench_dfs)
        .assign(rule=rule_d.name, pipeline=pipeline_name)
        .clean_names()
    )


benchmark_df_list: list[pd.DataFrame] = []

for pipeline_dir in benchmark_dir.iterdir():
    if pipeline_dir.name in {".DS_Store"}:
        continue
    for rule_dir in pipeline_dir.iterdir():
        df = get_benchmark_data_for_rule_dir(rule_dir, pipeline_name=pipeline_dir.name)
        if df is not None:
            benchmark_df_list.append(df)

benchmark_df = pd.concat(benchmark_df_list).reset_index(drop=True)
benchmark_df.head()
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
      <th>s</th>
      <th>h_m_s</th>
      <th>max_rss</th>
      <th>max_vms</th>
      <th>max_uss</th>
      <th>max_pss</th>
      <th>io_in</th>
      <th>io_out</th>
      <th>mean_load</th>
      <th>cpu_time</th>
      <th>step</th>
      <th>rule</th>
      <th>pipeline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55.1094</td>
      <td>0:00:55</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.0</td>
      <td>0</td>
      <td>example-specification_chain0</td>
      <td>sample_pymc3_mcmc</td>
      <td>010_010_model-fitting-pipeline</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.5505</td>
      <td>0:00:55</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.0</td>
      <td>0</td>
      <td>example-specification_chain1</td>
      <td>sample_pymc3_mcmc</td>
      <td>010_010_model-fitting-pipeline</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50.6351</td>
      <td>0:00:50</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.0</td>
      <td>0</td>
      <td>example-specification_chain0</td>
      <td>sample_stan_mcmc</td>
      <td>010_010_model-fitting-pipeline</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49.6582</td>
      <td>0:00:49</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.0</td>
      <td>0</td>
      <td>example-specification_chain1</td>
      <td>sample_stan_mcmc</td>
      <td>010_010_model-fitting-pipeline</td>
    </tr>
    <tr>
      <th>4</th>
      <td>110.2113</td>
      <td>0:01:50</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.0</td>
      <td>0</td>
      <td>example-specification</td>
      <td>sample_pymc3_advi</td>
      <td>010_010_model-fitting-pipeline</td>
    </tr>
  </tbody>
</table>
</div>

## Data dictionary

| colname | type (unit) | description |
|-------- |-------------|-------------|
| s | float (seconds) | Running time in seconds. |
| h:m:s	| string (-) | Running time in hour, minutes, seconds format. |
| max_rss | float (MB) | Maximum "Resident Set Size”, this is the non-swapped physical memory a process has used. |
| max_vms | float (MB) | Maximum “Virtual Memory Size”, this is the total amount of virtual memory used by the process. |
| max_uss | float (MB) | “Unique Set Size”, this is the memory which is unique to a process and which would be freed if the process was terminated right now. |
| max_pss | float (MB) | “Proportional Set Size”, is the amount of memory shared with other processes, accounted in a way that the amount is divided evenly between the processes that share it (Linux only). |
| io_in | float (MB) | The number of MB read (cumulative). |
| io_out | float (MB) | The number of MB written (cumulative). |
| mean_load | float (-) | CPU usage over time, divided by the total running time (first row). |
| cpu_time | float (-) | CPU time summed for user and system. |

## Data analysis and visualization

```python
benchmark_df.groupby(["pipeline", "rule"]).mean().round(2)
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
      <th>s</th>
      <th>mean_load</th>
      <th>cpu_time</th>
    </tr>
    <tr>
      <th>pipeline</th>
      <th>rule</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">010_010_model-fitting-pipeline</th>
      <th>sample_pymc3_advi</th>
      <td>110.21</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sample_pymc3_mcmc</th>
      <td>55.33</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sample_stan_mcmc</th>
      <td>50.15</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
benchmark_df_long = (
    benchmark_df[
        ["pipeline", "rule", "step", "cpu_time", "max_rss", "mean_load", "cpu_time"]
    ]
    .pivot_longer(["pipeline", "rule", "step"])
    .query("value != '-'")
    .astype({"value": float})
)

benchmark_df_long.head()
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
      <th>pipeline</th>
      <th>rule</th>
      <th>step</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>010_010_model-fitting-pipeline</td>
      <td>sample_pymc3_mcmc</td>
      <td>example-specification_chain0</td>
      <td>cpu_time</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>010_010_model-fitting-pipeline</td>
      <td>sample_pymc3_mcmc</td>
      <td>example-specification_chain1</td>
      <td>cpu_time</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>010_010_model-fitting-pipeline</td>
      <td>sample_stan_mcmc</td>
      <td>example-specification_chain0</td>
      <td>cpu_time</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>010_010_model-fitting-pipeline</td>
      <td>sample_stan_mcmc</td>
      <td>example-specification_chain1</td>
      <td>cpu_time</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>010_010_model-fitting-pipeline</td>
      <td>sample_pymc3_advi</td>
      <td>example-specification</td>
      <td>cpu_time</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
sns.catplot(
    data=benchmark_df_long,
    x="rule",
    y="value",
    kind="box",
    row="variable",
    col="pipeline",
    sharey=False,
)
```

    <seaborn.axisgrid.FacetGrid at 0x1240f3dc0>

![png](benchmarks_files/benchmarks_13_1.png)

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-01-14

    Python implementation: CPython
    Python version       : 3.9.9
    IPython version      : 8.0.0

    Compiler    : Clang 11.1.0
    OS          : Darwin
    Release     : 21.2.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit

    Hostname: JHCookMac

    Git branch: refactor-stan

    janitor: 0.22.0
    seaborn: 0.11.2
    pandas : 1.3.5

```python

```
