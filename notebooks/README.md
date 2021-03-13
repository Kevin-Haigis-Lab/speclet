# Analysis

Modeling construction and fitting.

## Experimentation with model design

> Notebook series `005`

These notebooks are for experimenting with various model designs, each one using different covariates and model structures.
They primarily use synthetic data and test whether the known values can be recovered.

1. [Model 1.](005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
2. [Model 2.](005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
3. [Model 3.](005_009_model-experimentation-m3.md) (Failed) A hierarhcical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
4. [Model 4.](005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
5. [Model 5.](005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribtuion for each gene.
6. [Model 6.](005_015_model-experimentation-m6.md) A heirarchcial model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
7. [Model 7.](005_017_model-experimentation-m7.md) A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene. A model with two varying intercepts was also successfully fit in this notebook.

## Experimentation with a subset of real data

> Notebook series `010`

<span style="color:#93152E">This series of notebooks was removed in commit [(put commit SHA here)]().</span>

This series of notebooks were the first attempt at modeling the data using PyMC3.
They are no longer under development nor testing, but exist primarily for reference.
There is a good chance that they are no longer reproducible and will likely be removed in the future.

1. [Exploratory data analysis.](010_005_exploratory-data-analysis.md) Exploration of the subset of data.
2. [Mimic CERES.](010_010_ceres-replicate.md) Replicate the results of the CERES model by recreating the model using PyMC3.
3. [Hierarchical modeling.](010_013_hierarchical-model-subsample.md) Building various *speclet* models with a subset of the DepMap data.
4. [2D covariates.](010_015_gene-lineage-hierarchical-matrix.md) An experiment with using 2D covariates with various hierarhcical levels.

#### Conclusions

There were many issues with the approach used in these notebooks.
First, to model all of the data in a single PyMC3 model would require astronomical system resources (i.e. RAM and time).
Also, the CERES model was not fit using a probabilistic programming language, but instead used a custom, piece-wise procedure that reduced the flexibility in covariate parameters.
Series `015` is the second generation and is under active development.

## Modeling CRC data

> Notebook series `015`

The purpose of this series is to experiment with various ways of modeling the CRC data using PyMC3 models.
Various models will be designed, fit, and compared.
The results will also be compared to those from the CERES model produced by DepMap and the results of using the CERES dependency scores for modeling.

1. [Exploratory data analysis.](015_005_exploratory-data-analysis.md) Exploration of the CRC cell line data.
2. [Designing models.](015_010_model-design.md) A scratch-pad for experimenting with model design.
3. [Workflow to fit CRC models.](015_017_run-crc-sampling-snakemake.py) A [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow for fitting CRC models, caching the results, and running the analysis notebook.
4. [Model analysis.](015_020_crc-model-analysis.md) Analyzing the posterior distributions of the fit CRC models.

## Miscellaneous experimentation

> Notebook series `999`

These are miscellaneous, small, experimental notebooks that exist to quickly test ideas for other uses.
They may not be fully reproducible, but exist primarily for reference.

1. [Multiple varying intercepts example.](999_005_experimentation.md) An example model for fitting multiple varying intercepts.
2. [Saving and loading PyMC3 models and samples](999_010_saving-and-loading-models.md) Testing various methods for wrapping `pm.sample()` for automatic caching and re-loading.
3. [Fititng splines](999_015_splines-in-pymc3.md) How to fit splines with PyMC3. I have yet to get a working multi-level model.
