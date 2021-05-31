# Data preparation

Raw data is stored in `data/` and the prepared data is saved to `modeling_data/`.
(The `cache` is for saving intermediate data of analyses.)
The DepMap 2020 Q3 data can be downloaded from [FigShare](https://figshare.com/articles/dataset/public_20q3/12931238/1) using the ["data/download-depmap-data.sh"](../data/download-depmap-data.sh) script.
A single Snakemake workflow prepares all of the data.
It can be run using the following command in the root directory of the project:

```bash
make munge
```
