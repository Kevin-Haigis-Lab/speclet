# Data preparation

Raw data is stored in ["data/"](../data) and the prepared data is saved to ["modeling_data/"](../modeling_data).
(The ["cache/"](../cache) directory is for saving intermediate data of analyses.)
The DepMap 2021 Q2 data can be downloaded by running the ["data/download-data.sh"](../data/download-data.sh) script from the root directory.

A single Snakemake workflow prepares all of the data.
It can be run using the following command in the root directory of the project.

```bash
make munge
```

If running on O2, the jobs can be parallelized over the HPC using the following command, instead.

```bash
make munge_o2
```

Below is the DAG of the pipeline (scaled down to just 5 cell lines).

![munge-pipeline-dag](munge-dag.png)
