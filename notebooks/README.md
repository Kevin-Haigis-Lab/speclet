# Notebooks

Notebooks should be used for experimentation and analysis, not for modeling fitting – model fitting should be performed in a the modeling fitting pipeline.
Each subdirectory contains notebooks for a related task.

Many of these notebooks can no longer be run from start to finish because the shared code has been altered and the notebook has not been updated.
This is not an issue for notebooks that are for experimentation or reference.
The expected behavior is indicated for each group of notebooks.

## [Data exploration](data-exploration)

![reproducibility-full](https://img.shields.io/badge/reproducibility-full-brightgreen.svg?style=flat)

Notebooks for general exploratory data analysis.

1. [Basic data statistics and plots](001_001_basic-data-statistics-and-plots.md)

## [Model design experimentation](model_design_experimentation)

![reproducibility-limited](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

These notebooks are for experimenting with various model designs, each one using different covariates and model structures.
They primarily use synthetic data and test whether the known values can be recovered.
Limited use the 'speclet' library should be used so as to limit coupling of code.

> These were early notebooks and are retained for reference only.

1. [Model 1.](model_design_experimentation/005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
2. [Model 2.](model_design_experimentation/005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
3. [Model 3.](model_design_experimentation/005_009_model-experimentation-m3.md) (Failed) A hierarchical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
4. [Model 4.](model_design_experimentation/005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
5. [Model 5.](model_design_experimentation/005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribution for each gene.
6. [Model 6.](model_design_experimentation/005_015_model-experimentation-m6.md) A hierarchical model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
7. [Model 7.](model_design_experimentation/005_017_model-experimentation-m7.md) A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene. A model with two varying intercepts was also successfully fit in this notebook.

## [Negative binomial modeling](negative-binomial-modeling)

![reproducibility-limited](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

A series of notebooks to experimenting with Negative Binomial (NB) likelihood distributions.

1. [NB distributions.](005_004_negative-binomial-distribution-experimentation.md). Some initial experiments with NB distributions.
1. [NB likelihooods.](005_005_basic-experimentation.md). Basic practice with generalized linear models with NB likelihoods.
1. [NB GLMs on CRISPR screen data.](005_010_simulation-nb-crispr.md). Generalized linear models with a NB likelihood fit with example CRISPR screen data.
1. [Exposure in CRISPR screen models.](005_013_different-exposure-methods.md). Experimentation with different measures of "exposure" for a NB model of CRISPR screen data.
1. [Initial CRISPR screen models.](005_015_simple-models-real-data.md). Some simpler GLMs with a NB likelihood and some interesting covariates for modeling CRISPR data.
1. [Comparing LM to GLM](005_020_compare-nb-to-normal.md). A comparison of similar GLMs except with either an identity or exponential link function and Gaussian or NB likelihood.

## [Miscellaneous experimentation](experimentation)

![reproducibility-limited](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

These are miscellaneous, small, experimental notebooks that exist to quickly test ideas for other uses.
They may not be fully reproducible, but exist primarily for reference.

1. [Multiple varying intercepts example.](experimentation/999_005_experimentation.md) An example model for fitting multiple varying intercepts.
2. [Saving and loading PyMC3 models and samples.](experimentation/999_010_saving-and-loading-models.md) Testing various methods for wrapping `pm.sample()` for automatic caching and re-loading.
3. [Fitting splines.](experimentation/999_015_splines-in-pymc3.md) How to fit splines with PyMC3. I have yet to get a working multi-level model.
4. [Simple SBC example.](999_020_simulation-based-calibration.md) A quick proof-of-concept for simulation-based calibration workflow.
5. [Combining MCMC chains.](999_025_combining-chains.md) How to combine MCMC chains run separately into a single ArviZ `InferenceData` object.
6. [Scaling copy number data.](999_030_scaling-copy-number.md) Effects of different transformations on copy number data.
7. [Scaling RNA expression data.](999_031_scaling-rna-expression.md) Effects of different transformations on RNA expression data.
8. [Mixing centered and non-centered parameterizations.](999_032_mixed-centered-parameterization-pymc3-model.md)
