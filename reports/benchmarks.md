```python
from pathlib import Path

import pandas as pd
import altair as alt
import janitor
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
def process_benchmark_file(bench_f: Path) -> pd.DataFrame:
    return pd.read_csv(bench_f, sep="\t").assign(step=bench_f.name.replace(bench_f.suffix, ""))

def get_benchmark_data_for_rule_dir(rule_d: Path, pipeline_name: str) -> pd.DataFrame:
    bench_dfs: list[pd.DataFrame] = [process_benchmark_file(b) for b in rule_d.iterdir()]
    return pd.concat(bench_dfs).assign(rule=rule_d.name, pipeline=pipeline_name).clean_names()

benchmark_df_list: list[pd.DataFrame] = []

for pipeline_dir in benchmark_dir.iterdir():
    for rule_dir in pipeline_dir.iterdir():
        benchmark_df_list.append(get_benchmark_data_for_rule_dir(rule_dir, pipeline_name=pipeline_dir.name))

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
      <td>160.2274</td>
      <td>0:02:40</td>
      <td>1295.18</td>
      <td>1938.04</td>
      <td>1250.59</td>
      <td>1271.62</td>
      <td>39.25</td>
      <td>4.05</td>
      <td>16.19</td>
      <td>26.33</td>
      <td>sp6-default</td>
      <td>sample_advi</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1767.7202</td>
      <td>0:29:27</td>
      <td>1447.37</td>
      <td>2517.04</td>
      <td>1444.73</td>
      <td>1444.91</td>
      <td>803.91</td>
      <td>102.04</td>
      <td>93.20</td>
      <td>1647.72</td>
      <td>sp4-default-fullrank</td>
      <td>sample_advi</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131.3041</td>
      <td>0:02:11</td>
      <td>1285.27</td>
      <td>1924.70</td>
      <td>1282.27</td>
      <td>1282.48</td>
      <td>5.78</td>
      <td>3.93</td>
      <td>17.02</td>
      <td>22.32</td>
      <td>sp5-default</td>
      <td>sample_advi</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>3</th>
      <td>138.4463</td>
      <td>0:02:18</td>
      <td>1295.54</td>
      <td>1948.86</td>
      <td>1292.96</td>
      <td>1293.05</td>
      <td>34.23</td>
      <td>4.71</td>
      <td>20.34</td>
      <td>28.41</td>
      <td>sp4-centered-copynum</td>
      <td>sample_advi</td>
      <td>010_010_run-crc-sampling-snakemake</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135.3755</td>
      <td>0:02:15</td>
      <td>1281.48</td>
      <td>1930.82</td>
      <td>1278.91</td>
      <td>1278.98</td>
      <td>0.39</td>
      <td>4.70</td>
      <td>16.47</td>
      <td>22.57</td>
      <td>sp2-default</td>
      <td>sample_advi</td>
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
      <th rowspan="2" valign="top">010_010_run-crc-sampling-snakemake</th>
      <th>sample_advi</th>
      <td>309.92</td>
      <td>1304.34</td>
      <td>2023.15</td>
      <td>1297.18</td>
      <td>1299.43</td>
      <td>166.95</td>
      <td>14.03</td>
      <td>24.80</td>
      <td>186.06</td>
    </tr>
    <tr>
      <th>sample_mcmc</th>
      <td>1512.58</td>
      <td>1439.86</td>
      <td>2333.38</td>
      <td>1429.85</td>
      <td>1433.68</td>
      <td>320.52</td>
      <td>47.70</td>
      <td>80.13</td>
      <td>1267.32</td>
    </tr>
  </tbody>
</table>
</div>

```python
benchmark_df_long = benchmark_df[["pipeline", "rule", "step", "cpu_time", "max_rss", "mean_load", "cpu_time"]].pivot_longer(["pipeline", "rule", "step"])

(
    alt.Chart(benchmark_df_long)
    .mark_boxplot(size=50)
    .encode(
        x="rule",
        y=alt.Y("value", title=""),
        row=alt.Row("variable", title=""),
        column=alt.Column('pipeline'),
    )
    .properties(width=200, height=100)
    .resolve_scale(y='independent')
)
```

<div id="altair-viz-33994c88f7cd40eba16d460a67ce6b23"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-33994c88f7cd40eba16d460a67ce6b23") {
      outputDiv = document.getElementById("altair-viz-33994c88f7cd40eba16d460a67ce6b23");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-c630aeaeba6340b8780d3df4698030a8"}, "mark": {"type": "boxplot", "size": 50}, "encoding": {"column": {"type": "nominal", "field": "pipeline"}, "row": {"type": "nominal", "field": "variable", "title": ""}, "tooltip": [], "x": {"type": "nominal", "field": "rule"}, "y": {"type": "quantitative", "field": "value", "title": ""}}, "height": 100, "resolve": {"scale": {"y": "independent"}}, "width": 200, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-c630aeaeba6340b8780d3df4698030a8": [{"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp6-default", "variable": "cpu_time", "value": 26.33}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default-fullrank", "variable": "cpu_time", "value": 1647.72}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-default", "variable": "cpu_time", "value": 22.32}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-centered-copynum", "variable": "cpu_time", "value": 28.41}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp2-default", "variable": "cpu_time", "value": 22.57}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp7-default", "variable": "cpu_time", "value": 20.99}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-noncentered", "variable": "cpu_time", "value": 20.43}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default", "variable": "cpu_time", "value": 22.47}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered", "variable": "cpu_time", "value": 22.6}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered-copynum", "variable": "cpu_time", "value": 26.81}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain0", "variable": "cpu_time", "value": 611.22}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain0", "variable": "cpu_time", "value": 360.72}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain1", "variable": "cpu_time", "value": 329.62}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain1", "variable": "cpu_time", "value": 597.2}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain3", "variable": "cpu_time", "value": 614.21}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain3", "variable": "cpu_time", "value": 394.14}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain2", "variable": "cpu_time", "value": 374.15}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain2", "variable": "cpu_time", "value": 635.52}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain2", "variable": "cpu_time", "value": 1967.62}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain3", "variable": "cpu_time", "value": 2179.39}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain1", "variable": "cpu_time", "value": 1856.39}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain0", "variable": "cpu_time", "value": 1865.28}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain2", "variable": "cpu_time", "value": 1539.18}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain3", "variable": "cpu_time", "value": 874.76}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain1", "variable": "cpu_time", "value": 891.48}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain0", "variable": "cpu_time", "value": 1436.22}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain2", "variable": "cpu_time", "value": 1858.29}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain2", "variable": "cpu_time", "value": 1995.04}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain3", "variable": "cpu_time", "value": 1812.74}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain3", "variable": "cpu_time", "value": 3008.48}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain1", "variable": "cpu_time", "value": 1691.11}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain1", "variable": "cpu_time", "value": 2008.26}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain0", "variable": "cpu_time", "value": 1881.07}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain0", "variable": "cpu_time", "value": 1736.99}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain2", "variable": "cpu_time", "value": 163.77}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain3", "variable": "cpu_time", "value": 198.38}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain1", "variable": "cpu_time", "value": 197.8}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain0", "variable": "cpu_time", "value": 158.53}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain3", "variable": "cpu_time", "value": 2043.36}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain2", "variable": "cpu_time", "value": 2000.91}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain0", "variable": "cpu_time", "value": 2121.29}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain1", "variable": "cpu_time", "value": 1892.01}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain1", "variable": "cpu_time", "value": 971.0}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain0", "variable": "cpu_time", "value": 926.15}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain2", "variable": "cpu_time", "value": 1383.81}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain3", "variable": "cpu_time", "value": 1047.34}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp6-default", "variable": "max_rss", "value": 1295.18}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default-fullrank", "variable": "max_rss", "value": 1447.37}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-default", "variable": "max_rss", "value": 1285.27}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-centered-copynum", "variable": "max_rss", "value": 1295.54}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp2-default", "variable": "max_rss", "value": 1281.48}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp7-default", "variable": "max_rss", "value": 1301.98}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-noncentered", "variable": "max_rss", "value": 1277.66}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default", "variable": "max_rss", "value": 1285.94}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered", "variable": "max_rss", "value": 1284.8}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered-copynum", "variable": "max_rss", "value": 1288.16}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain0", "variable": "max_rss", "value": 1390.63}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain0", "variable": "max_rss", "value": 1370.79}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain1", "variable": "max_rss", "value": 1358.33}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain1", "variable": "max_rss", "value": 1402.98}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain3", "variable": "max_rss", "value": 1402.43}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain3", "variable": "max_rss", "value": 1378.41}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain2", "variable": "max_rss", "value": 1370.21}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain2", "variable": "max_rss", "value": 1390.21}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain2", "variable": "max_rss", "value": 1500.79}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain3", "variable": "max_rss", "value": 1462.01}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain1", "variable": "max_rss", "value": 1447.26}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain0", "variable": "max_rss", "value": 1552.29}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain2", "variable": "max_rss", "value": 1438.24}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain3", "variable": "max_rss", "value": 1410.36}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain1", "variable": "max_rss", "value": 1444.84}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain0", "variable": "max_rss", "value": 1425.09}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain2", "variable": "max_rss", "value": 1501.35}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain2", "variable": "max_rss", "value": 1425.93}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain3", "variable": "max_rss", "value": 1469.34}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain3", "variable": "max_rss", "value": 1452.1}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain1", "variable": "max_rss", "value": 1451.43}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain1", "variable": "max_rss", "value": 1516.52}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain0", "variable": "max_rss", "value": 1430.52}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain0", "variable": "max_rss", "value": 1426.0}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain2", "variable": "max_rss", "value": 1454.91}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain3", "variable": "max_rss", "value": 1400.55}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain1", "variable": "max_rss", "value": 1393.86}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain0", "variable": "max_rss", "value": 1414.7}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain3", "variable": "max_rss", "value": 1601.46}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain2", "variable": "max_rss", "value": 1478.08}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain0", "variable": "max_rss", "value": 1570.75}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain1", "variable": "max_rss", "value": 1491.8}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain1", "variable": "max_rss", "value": 1387.11}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain0", "variable": "max_rss", "value": 1396.42}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain2", "variable": "max_rss", "value": 1396.11}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain3", "variable": "max_rss", "value": 1431.29}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp6-default", "variable": "mean_load", "value": 16.19}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default-fullrank", "variable": "mean_load", "value": 93.2}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-default", "variable": "mean_load", "value": 17.02}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-centered-copynum", "variable": "mean_load", "value": 20.34}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp2-default", "variable": "mean_load", "value": 16.47}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp7-default", "variable": "mean_load", "value": 16.32}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-noncentered", "variable": "mean_load", "value": 30.38}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default", "variable": "mean_load", "value": 11.58}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered", "variable": "mean_load", "value": 14.53}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered-copynum", "variable": "mean_load", "value": 11.94}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain0", "variable": "mean_load", "value": 82.84}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain0", "variable": "mean_load", "value": 59.99}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain1", "variable": "mean_load", "value": 73.85}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain1", "variable": "mean_load", "value": 61.44}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain3", "variable": "mean_load", "value": 63.0}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain3", "variable": "mean_load", "value": 51.76}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain2", "variable": "mean_load", "value": 86.11}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain2", "variable": "mean_load", "value": 64.37}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain2", "variable": "mean_load", "value": 82.35}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain3", "variable": "mean_load", "value": 83.8}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain1", "variable": "mean_load", "value": 83.53}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain0", "variable": "mean_load", "value": 81.67}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain2", "variable": "mean_load", "value": 97.15}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain3", "variable": "mean_load", "value": 77.08}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain1", "variable": "mean_load", "value": 94.23}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain0", "variable": "mean_load", "value": 78.45}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain2", "variable": "mean_load", "value": 87.76}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain2", "variable": "mean_load", "value": 87.13}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain3", "variable": "mean_load", "value": 85.6}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain3", "variable": "mean_load", "value": 90.79}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain1", "variable": "mean_load", "value": 94.05}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain1", "variable": "mean_load", "value": 87.62}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain0", "variable": "mean_load", "value": 97.59}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain0", "variable": "mean_load", "value": 85.41}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain2", "variable": "mean_load", "value": 35.09}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain3", "variable": "mean_load", "value": 75.05}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain1", "variable": "mean_load", "value": 76.15}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain0", "variable": "mean_load", "value": 71.34}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain3", "variable": "mean_load", "value": 83.56}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain2", "variable": "mean_load", "value": 82.97}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain0", "variable": "mean_load", "value": 83.31}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain1", "variable": "mean_load", "value": 95.97}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain1", "variable": "mean_load", "value": 90.44}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain0", "variable": "mean_load", "value": 79.7}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain2", "variable": "mean_load", "value": 85.08}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain3", "variable": "mean_load", "value": 88.62}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp6-default", "variable": "cpu_time", "value": 26.33}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default-fullrank", "variable": "cpu_time", "value": 1647.72}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-default", "variable": "cpu_time", "value": 22.32}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-centered-copynum", "variable": "cpu_time", "value": 28.41}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp2-default", "variable": "cpu_time", "value": 22.57}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp7-default", "variable": "cpu_time", "value": 20.99}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp5-noncentered", "variable": "cpu_time", "value": 20.43}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-default", "variable": "cpu_time", "value": 22.47}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered", "variable": "cpu_time", "value": 22.6}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_advi", "step": "sp4-noncentered-copynum", "variable": "cpu_time", "value": 26.81}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain0", "variable": "cpu_time", "value": 611.22}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain0", "variable": "cpu_time", "value": 360.72}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain1", "variable": "cpu_time", "value": 329.62}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain1", "variable": "cpu_time", "value": 597.2}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain3", "variable": "cpu_time", "value": 614.21}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain3", "variable": "cpu_time", "value": 394.14}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-default_chain2", "variable": "cpu_time", "value": 374.15}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp5-noncentered_chain2", "variable": "cpu_time", "value": 635.52}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain2", "variable": "cpu_time", "value": 1967.62}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain3", "variable": "cpu_time", "value": 2179.39}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain1", "variable": "cpu_time", "value": 1856.39}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp6-default_chain0", "variable": "cpu_time", "value": 1865.28}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain2", "variable": "cpu_time", "value": 1539.18}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain3", "variable": "cpu_time", "value": 874.76}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain1", "variable": "cpu_time", "value": 891.48}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-centered-copynum_chain0", "variable": "cpu_time", "value": 1436.22}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain2", "variable": "cpu_time", "value": 1858.29}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain2", "variable": "cpu_time", "value": 1995.04}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain3", "variable": "cpu_time", "value": 1812.74}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain3", "variable": "cpu_time", "value": 3008.48}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain1", "variable": "cpu_time", "value": 1691.11}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain1", "variable": "cpu_time", "value": 2008.26}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered-copynum_chain0", "variable": "cpu_time", "value": 1881.07}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-noncentered_chain0", "variable": "cpu_time", "value": 1736.99}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain2", "variable": "cpu_time", "value": 163.77}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain3", "variable": "cpu_time", "value": 198.38}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain1", "variable": "cpu_time", "value": 197.8}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp2-default_chain0", "variable": "cpu_time", "value": 158.53}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain3", "variable": "cpu_time", "value": 2043.36}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain2", "variable": "cpu_time", "value": 2000.91}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain0", "variable": "cpu_time", "value": 2121.29}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp7-default_chain1", "variable": "cpu_time", "value": 1892.01}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain1", "variable": "cpu_time", "value": 971.0}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain0", "variable": "cpu_time", "value": 926.15}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain2", "variable": "cpu_time", "value": 1383.81}, {"pipeline": "010_010_run-crc-sampling-snakemake", "rule": "sample_mcmc", "step": "sp4-default_chain3", "variable": "cpu_time", "value": 1047.34}]}}, {"mode": "vega-lite"});
</script>

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```
