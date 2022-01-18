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

    [PosixPath('/n/data1/hms/dbmi/park/Cook/speclet/benchmarks/010_010_run-crc-sampling-snakemake'),
     PosixPath('/n/data1/hms/dbmi/park/Cook/speclet/benchmarks/010_010_model-fitting-pipeline'),
     PosixPath('/n/data1/hms/dbmi/park/Cook/speclet/benchmarks/012_010_simulation-based-calibration-snakemake')]

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
      <td>221.5452</td>
      <td>0:03:41</td>
      <td>1419.73</td>
      <td>2140.38</td>
      <td>1417.16</td>
      <td>1417.25</td>
      <td>908.23</td>
      <td>16.45</td>
      <td>17.16</td>
      <td>38.26</td>
      <td>simple-default_chain0</td>
      <td>sample_mcmc</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>1</th>
      <td>321.1937</td>
      <td>0:05:21</td>
      <td>465.15</td>
      <td>1062.94</td>
      <td>462.39</td>
      <td>462.47</td>
      <td>289.45</td>
      <td>104.28</td>
      <td>63.86</td>
      <td>205.30</td>
      <td>nine-noncentered_chain2</td>
      <td>sample_mcmc</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>2</th>
      <td>313.5864</td>
      <td>0:05:13</td>
      <td>407.55</td>
      <td>1052.50</td>
      <td>365.42</td>
      <td>385.26</td>
      <td>158.54</td>
      <td>77.25</td>
      <td>58.78</td>
      <td>174.10</td>
      <td>nine-noncentered_chain1</td>
      <td>sample_mcmc</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>3</th>
      <td>319.5655</td>
      <td>0:05:19</td>
      <td>423.44</td>
      <td>1051.12</td>
      <td>378.94</td>
      <td>387.15</td>
      <td>97.58</td>
      <td>124.66</td>
      <td>60.99</td>
      <td>197.01</td>
      <td>nine-default_chain1</td>
      <td>sample_mcmc</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>4</th>
      <td>224.0925</td>
      <td>0:03:44</td>
      <td>1419.79</td>
      <td>2140.32</td>
      <td>1417.19</td>
      <td>1417.29</td>
      <td>959.60</td>
      <td>12.40</td>
      <td>17.09</td>
      <td>38.29</td>
      <td>simple-default_chain1</td>
      <td>sample_mcmc</td>
      <td>010_010_run-crc-sampling-snakemake</td>
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
      <th rowspan="3" valign="top">010_010_model-fitting-pipeline</th>
      <th>sample_pymc3_advi</th>
      <td>273.26</td>
      <td>334.38</td>
      <td>945.97</td>
      <td>287.92</td>
      <td>309.51</td>
      <td>2.88</td>
      <td>132.27</td>
      <td>45.35</td>
      <td>123.36</td>
    </tr>
    <tr>
      <th>sample_pymc3_mcmc</th>
      <td>228.63</td>
      <td>385.99</td>
      <td>979.62</td>
      <td>340.08</td>
      <td>361.68</td>
      <td>9.98</td>
      <td>171.55</td>
      <td>12.26</td>
      <td>24.86</td>
    </tr>
    <tr>
      <th>sample_stan_mcmc</th>
      <td>47.53</td>
      <td>806.60</td>
      <td>2187.02</td>
      <td>602.32</td>
      <td>704.21</td>
      <td>117.32</td>
      <td>126.24</td>
      <td>31.34</td>
      <td>14.68</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">010_010_run-crc-sampling-snakemake</th>
      <th>sample_advi</th>
      <td>220.61</td>
      <td>1418.07</td>
      <td>2197.32</td>
      <td>1415.48</td>
      <td>1415.64</td>
      <td>967.38</td>
      <td>25.52</td>
      <td>19.70</td>
      <td>43.53</td>
    </tr>
    <tr>
      <th>sample_mcmc</th>
      <td>1419.45</td>
      <td>725.94</td>
      <td>1345.14</td>
      <td>708.15</td>
      <td>711.92</td>
      <td>372.02</td>
      <td>88.66</td>
      <td>59.05</td>
      <td>1265.54</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">012_010_simulation-based-calibration-snakemake</th>
      <th>collate_sbc</th>
      <td>641.62</td>
      <td>803.26</td>
      <td>2913.11</td>
      <td>799.93</td>
      <td>802.38</td>
      <td>26585.63</td>
      <td>0.86</td>
      <td>60.40</td>
      <td>399.18</td>
    </tr>
    <tr>
      <th>generate_mockdata</th>
      <td>20.44</td>
      <td>173.16</td>
      <td>880.76</td>
      <td>170.72</td>
      <td>170.77</td>
      <td>216.71</td>
      <td>0.37</td>
      <td>19.58</td>
      <td>3.01</td>
    </tr>
    <tr>
      <th>run_sbc</th>
      <td>602.97</td>
      <td>1744.51</td>
      <td>3399.60</td>
      <td>742.90</td>
      <td>751.64</td>
      <td>52.41</td>
      <td>198.71</td>
      <td>95.33</td>
      <td>86.48</td>
    </tr>
    <tr>
      <th>sbc_uniformity_test</th>
      <td>858.15</td>
      <td>5310.08</td>
      <td>6813.24</td>
      <td>5305.28</td>
      <td>5307.70</td>
      <td>47729.32</td>
      <td>0.38</td>
      <td>55.34</td>
      <td>488.02</td>
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
      <td>010_010_run-crc-sampling-snakemake</td>
      <td>sample_mcmc</td>
      <td>simple-default_chain0</td>
      <td>cpu_time</td>
      <td>38.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>010_010_run-crc-sampling-snakemake</td>
      <td>sample_mcmc</td>
      <td>nine-noncentered_chain2</td>
      <td>cpu_time</td>
      <td>205.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>010_010_run-crc-sampling-snakemake</td>
      <td>sample_mcmc</td>
      <td>nine-noncentered_chain1</td>
      <td>cpu_time</td>
      <td>174.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>010_010_run-crc-sampling-snakemake</td>
      <td>sample_mcmc</td>
      <td>nine-default_chain1</td>
      <td>cpu_time</td>
      <td>197.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>010_010_run-crc-sampling-snakemake</td>
      <td>sample_mcmc</td>
      <td>simple-default_chain1</td>
      <td>cpu_time</td>
      <td>38.29</td>
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

    <seaborn.axisgrid.FacetGrid at 0x7f5be94938e0>

![png](benchmarks_files/benchmarks_13_1.png)

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-01-18

    Python implementation: CPython
    Python version       : 3.9.9
    IPython version      : 8.0.0

    Compiler    : GCC 9.4.0
    OS          : Linux
    Release     : 3.10.0-1160.45.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-237.o2.rc.hms.harvard.edu

    Git branch: run-on-o2

    janitor: 0.22.0
    seaborn: 0.11.2
    pandas : 1.3.5

```python

```
