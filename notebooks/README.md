# Notebooks

Notebooks should be used for experimentation and analysis, not for modeling fitting – model fitting should be performed in a the modeling fitting pipeline.
Each subdirectory contains notebooks for a related task.

Many of these notebooks can no longer be run from start to finish because the shared code has been altered and the notebook has not been updated.
This is not an issue for notebooks that are for experimentation or reference.
The expected behavior is indicated for each group of notebooks.

## [Model design experimentation](model_design_experimentation)

![not-reproducible](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

These notebooks are for experimenting with various model designs, each one using different covariates and model structures.
They primarily use synthetic data and test whether the known values can be recovered.

1. [Model 1.](model_design_experimentation/005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
2. [Model 2.](model_design_experimentation/005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
3. [Model 3.](model_design_experimentation/005_009_model-experimentation-m3.md) (Failed) A hierarchical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
4. [Model 4.](model_design_experimentation/005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
5. [Model 5.](model_design_experimentation/005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribution for each gene.
6. [Model 6.](model_design_experimentation/005_015_model-experimentation-m6.md) A hierarchical model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
7. [Model 7.](model_design_experimentation/005_017_model-experimentation-m7.md) A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene. A model with two varying intercepts was also successfully fit in this notebook.

## [Modeling CRC data](crc_pipeline)

![reproduciblity-expected](https://img.shields.io/badge/reproducibility-expected-yellow.svg?style=flat)

The purpose of this series is to experiment with various ways of modeling the CRC data using PyMC3 models.
Various models will be designed, fit, and compared.
The results will also be compared to those from the CERES model produced by DepMap and the results of using the CERES dependency scores for modeling.

1. [Exploratory data analysis.](crc_pipeline/015_005_exploratory-data-analysis.md) Exploration of the CRC cell line data.
2. [Designing Model 2.](crc_pipeline/015_010_m2-design.md) A scratch-pad for experimenting with designing Model 2.
3. [LOO-CV.](015_012_loo-cv-experimentation.md) Experimentation with some analysis of the LOO-CV of fit models to be included in the final summary report.
4. [Copy number covariate in CERES mimic.](015_014_ceres-with-cn-covariate.md) Adding an optional covariate in the CERES mimic model.
5. [Debugging SpecletThree.](020_010_debug-speclet_three.md) Debugging issues with `src.models.SpecletThree`.
6. [Non-centered reparameterization.](020_015_noncentered-paramaterization.md) Implementing a non-centered parameterization of the speclet models.

## [CRC modeling analysis](crc_model_analysis)

![reproduciblity-full](https://img.shields.io/badge/reproducibility-full-brightgreen.svg?style=flat)

1. [Initial model exploration.](010_010_initial-model-exploration.md)
2. [Analysis of SpecletOne fit.](010_015_analysis-of-specletone-fit.md)
3. [Analysis of SpecletTwo fit.](010_020_analysis-of-speclettwo.md)
4. [Comparing models with and without the *KRAS* covariate.](010_025_compare-kras-covariate.md)
5. [Analysis of SpecletThree fit.](010_030_analysis-of-specletthree.md)
6. [Analysis of SpecletSix SBC.](020_010_experimentation-speclet6-sbc.md)
7. [Experimentation with SpecletSeven.](020_015_experimentation_speclet7.md)

## [Miscellaneous experimentation](experimentation)

![not-reproducible](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

These are miscellaneous, small, experimental notebooks that exist to quickly test ideas for other uses.
They may not be fully reproducible, but exist primarily for reference.

1. [Multiple varying intercepts example.](experimentation/999_005_experimentation.md) An example model for fitting multiple varying intercepts.
2. [Saving and loading PyMC3 models and samples.](experimentation/999_010_saving-and-loading-models.md) Testing various methods for wrapping `pm.sample()` for automatic caching and re-loading.
3. [Fitting splines.](experimentation/999_015_splines-in-pymc3.md) How to fit splines with PyMC3. I have yet to get a working multi-level model.
4. [Simple SBC example.](999_020_simulation-based-calibration.md) A quick proof-of-concept for simulation-based calibration workflow.
5. [Combining MCMC chains.](999_025_combining-chains.md) How to combine MCMC chains run separately into a single ArviZ `InferenceData` object.
6. [Scaling copy number data.](999_030_scaling-copy-number.md) Effects of different transformations on copy number data.
7. [Scaling RNA expression data.](999_031_scaling-rna-expression.md) Effects of different transformations on RNA expression data.
