# Data preparation

Raw data is stored in `data/` and the prepared data is saved to `modeling_data/`. (The `cache` is for saving intermediate data of analyses.)
The DepMap 2020 Q3 data can be downloaded from [FigShare](https://figshare.com/articles/dataset/public_20q3/12931238/1) using the "download-depmap-data.sh" script.

Each piece of raw data is first prepared in a single notebook and then a SnakeMake workflow is run to merge everything together into a single data frame that will be used for modeling.

1. [Prepare DepMap data](005_prepare-depmap-data.md)
2. [Prepare modeling data SnakeMake workflow](010_prepare-modeling-data_snakemake.py)
