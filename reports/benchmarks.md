```python
from pathlib import Path
from typing import Optional

import janitor
import pandas as pd
import plotnine as gg
import seaborn as sns

%config InlineBackend.figure_format='retina'
```

```python
benchmark_dir = Path("../benchmarks/")
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

    [PosixPath('../benchmarks/010_010_run-crc-sampling-snakemake'),
     PosixPath('../benchmarks/012_010_simulation-based-calibration-snakemake')]

```python
def process_benchmark_file(bench_f: Path) -> pd.DataFrame:
    return pd.read_csv(bench_f, sep="\t").assign(
        step=bench_f.name.replace(bench_f.suffix, "")
    )


def get_benchmark_data_for_rule_dir(rule_d: Path, pipeline_name: str) -> Optional[pd.DataFrame]:
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
      <td>900.0916</td>
      <td>0:15:00</td>
      <td>11876.86</td>
      <td>16598.31</td>
      <td>2643.14</td>
      <td>2643.29</td>
      <td>8.54</td>
      <td>144.07</td>
      <td>80.96</td>
      <td>138.08</td>
      <td>sp2-default_MCMC_perm106</td>
      <td>run_sbc</td>
      <td>012_010_simulation-based-calibration-snakemake</td>
    </tr>
    <tr>
      <th>1</th>
      <td>681.0140</td>
      <td>0:11:21</td>
      <td>11834.54</td>
      <td>16552.55</td>
      <td>2673.76</td>
      <td>2698.16</td>
      <td>8.98</td>
      <td>149.32</td>
      <td>67.21</td>
      <td>156.21</td>
      <td>sp2-default_MCMC_perm447</td>
      <td>run_sbc</td>
      <td>012_010_simulation-based-calibration-snakemake</td>
    </tr>
    <tr>
      <th>2</th>
      <td>745.1272</td>
      <td>0:12:25</td>
      <td>12174.35</td>
      <td>16889.28</td>
      <td>2686.37</td>
      <td>2686.53</td>
      <td>15.43</td>
      <td>160.75</td>
      <td>82.28</td>
      <td>131.27</td>
      <td>sp2-default_MCMC_perm10</td>
      <td>run_sbc</td>
      <td>012_010_simulation-based-calibration-snakemake</td>
    </tr>
    <tr>
      <th>3</th>
      <td>609.5455</td>
      <td>0:10:09</td>
      <td>12045.56</td>
      <td>16742.46</td>
      <td>2615.14</td>
      <td>2637.87</td>
      <td>4.54</td>
      <td>92.44</td>
      <td>133.23</td>
      <td>115.00</td>
      <td>sp2-default_MCMC_perm82</td>
      <td>run_sbc</td>
      <td>012_010_simulation-based-calibration-snakemake</td>
    </tr>
    <tr>
      <th>4</th>
      <td>605.5162</td>
      <td>0:10:05</td>
      <td>11209.66</td>
      <td>15945.14</td>
      <td>2464.32</td>
      <td>2477.39</td>
      <td>8.06</td>
      <td>84.99</td>
      <td>79.81</td>
      <td>119.25</td>
      <td>sp2-default_MCMC_perm24</td>
      <td>run_sbc</td>
      <td>012_010_simulation-based-calibration-snakemake</td>
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
      <th>max_rss</th>
      <th>max_vms</th>
      <th>max_uss</th>
      <th>max_pss</th>
      <th>io_in</th>
      <th>io_out</th>
      <th>mean_load</th>
      <th>cpu_time</th>
    </tr>
    <tr>
      <th>pipeline</th>
      <th>rule</th>
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
      <th rowspan="3" valign="top">012_010_simulation-based-calibration-snakemake</th>
      <th>collate_sbc</th>
      <td>272.46</td>
      <td>288.61</td>
      <td>1013.56</td>
      <td>286.13</td>
      <td>286.20</td>
      <td>2269.80</td>
      <td>0.22</td>
      <td>35.53</td>
      <td>95.78</td>
    </tr>
    <tr>
      <th>generate_mockdata</th>
      <td>8.94</td>
      <td>216.96</td>
      <td>929.21</td>
      <td>214.40</td>
      <td>214.44</td>
      <td>259.24</td>
      <td>0.29</td>
      <td>40.41</td>
      <td>3.72</td>
    </tr>
    <tr>
      <th>run_sbc</th>
      <td>655.81</td>
      <td>11195.34</td>
      <td>15717.18</td>
      <td>2582.34</td>
      <td>2592.93</td>
      <td>244.20</td>
      <td>122.06</td>
      <td>114.10</td>
      <td>128.91</td>
    </tr>
  </tbody>
</table>
</div>

```python
benchmark_df_long = benchmark_df[
    ["pipeline", "rule", "step", "cpu_time", "max_rss", "mean_load", "cpu_time"]
].pivot_longer(["pipeline", "rule", "step"])

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
      <td>012_010_simulation-based-calibration-snakemake</td>
      <td>run_sbc</td>
      <td>sp2-default_MCMC_perm106</td>
      <td>cpu_time</td>
      <td>138.08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>012_010_simulation-based-calibration-snakemake</td>
      <td>run_sbc</td>
      <td>sp2-default_MCMC_perm447</td>
      <td>cpu_time</td>
      <td>156.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>012_010_simulation-based-calibration-snakemake</td>
      <td>run_sbc</td>
      <td>sp2-default_MCMC_perm10</td>
      <td>cpu_time</td>
      <td>131.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>012_010_simulation-based-calibration-snakemake</td>
      <td>run_sbc</td>
      <td>sp2-default_MCMC_perm82</td>
      <td>cpu_time</td>
      <td>115.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>012_010_simulation-based-calibration-snakemake</td>
      <td>run_sbc</td>
      <td>sp2-default_MCMC_perm24</td>
      <td>cpu_time</td>
      <td>119.25</td>
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
);
```

    <seaborn.axisgrid.FacetGrid at 0x7fb62c7ed790>

![png](benchmarks_files/benchmarks_10_1.png)

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-07-26

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

    Hostname: compute-a-17-126.o2.rc.hms.harvard.edu

    Git branch: sbc-uniform-check

    seaborn : 0.11.1
    pandas  : 1.2.3
    janitor : 0.20.14
    plotnine: 0.7.1
