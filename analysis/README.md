# Analysis

Modeling construction and fitting.

## Experimentation with model design

> Notebook series `005`

1. [Model 1.](005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
2. [Model 2.](005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
3. [Model 3.](005_009_model-experimentation-m3.md) (Failed) A hierarhcical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
4. [Model 4.](005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
5. [Model 5.](005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribtuion for each gene.
6. [Model 6.](005_015_model-experimentation-m6.md) A heirarchcial model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
7. [Model 7.](005_017_model-experimentation-m7.md) A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene. A model with two varying intercepts was also successfully fit in this notebook.

## Experimentation with a subset of real data

> Notebook series `010`

1. [Exploratory data analysis.](010_005_exploratory-data-analysis.md) Exploration of the subset of data.
2. [Mimic CERES.](010_010_ceres-replicate.md) Replicate the results of the CERES model by recreating the model using PyMC3.
3. [Hierarchical modeling.](010_013_hierarchical-model-subsample.md) Building various *speclet* models with a subset of the DepMap data.

## Miscellaneous experimentation

> Notebook series `999`

1. [Multiple varying intercepts example.](999_005_experimentation.md) An example model for fitting multiple varying intercepts.
2. [Saving and loading PyMC3 models and samples](999_010_saving-and-loading-models.md) Testing various methods for wrapping `pm.sample()` for automatic caching and re-loading.
3. [Fititng splines](999_015_splines-in-pymc3.md) How to fit splines with PyMC3. I have yet to get a working multi-level model.
