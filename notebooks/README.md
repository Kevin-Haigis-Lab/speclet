# Notebooks

Notebooks were used for experimentation and analysis, not for modeling fitting – model fitting was performed in a the modeling fitting pipeline.
Each subdirectory contains notebooks for a set of analyses.

Many of these notebooks can no longer be run from start to finish because the shared code has been altered and the notebook has not been updated.
This is not an issue for notebooks that are for experimentation or reference.
The expected behavior is indicated for each group of notebooks.
Note, the provided links are to the Markdown files generated from each notebook.

## [Data exploration](data-exploration)

![reproducibility-full](https://img.shields.io/badge/reproducibility-full-brightgreen.svg?style=flat)

Notebooks for general exploratory data analysis.

1. [Basic data statistics and plots.](data-exploration/001_001_basic-data-statistics-and-plots.md)
1. [Exploration of molecular variates.](data-exploration/001_005_molecular-covariates-eda.md)
1. [Cell line lineages and lineage subtypes.](data-exploration/001_020_lineage-exploration.md) Understanding the lineage and lineage subtype relationships.
1. [Exploration of data by cell line lineage.](data-exploration/005_100_lineage-data-exploration.md) Testing how well the cell lines can be clustered by the raw log-fold change data.

## [Model design experimentation](model_design_experimentation)

![reproducibility-limited](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

These notebooks are for experimenting with various model designs, each one using different covariates and model structures.
They primarily use synthetic data and test whether the known values can be recovered.

> These were early notebooks and are retained for reference only.

1. [Model 1.](model_design_experimentation/005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
1. [Model 2.](model_design_experimentation/005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
1. [Model 3.](model_design_experimentation/005_009_model-experimentation-m3.md) (Failed) A hierarchical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
1. [Model 4.](model_design_experimentation/005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
1. [Model 5.](model_design_experimentation/005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribution for each gene.
1. [Model 6.](model_design_experimentation/005_015_model-experimentation-m6.md) A hierarchical model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
1. [Model 7.](model_design_experimentation/005_017_model-experimentation-m7.md) A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene. A model with two varying intercepts was also successfully fit in this notebook.

## [Model construction](model-construction/)

![reproducibility-limited](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

A series of notebooks that build the final model based off of the earlier experimentation.
The model was constructed one piece at a time, experimenting with the effect of the latest addition on the model fit and other variables.

1. [NB distributions.](model-construction/005_004_negative-binomial-distribution-experimentation.md) Some initial experiments with NB distributions.
1. [NB likelihooods.](model-construction/005_005_basic-experimentation.md) Basic practice with generalized linear models with NB likelihoods.
1. [NB GLMs on CRISPR screen data.](model-construction/005_010_simulation-nb-crispr.md) Generalized linear models with a NB likelihood fit with example CRISPR screen data.
1. [Exposure in CRISPR screen models.](model-construction/005_013_different-exposure-methods.md) Experimentation with different measures of "exposure" for a NB model of CRISPR screen data.
1. [Initial CRISPR screen models.](model-construction/005_015_simple-models-real-data.md) Some simpler GLMs with a NB likelihood and some interesting covariates for modeling CRISPR data.
1. [Comparing LM to GLM.](model-construction/005_020_compare-nb-to-normal.md) A comparison of similar GLMs except with either an identity or exponential link function and Gaussian or NB likelihood.
1. [Add CN covariate.](model-construction/010_005_adding-copy-number-covar.md) Introducing the copy number of the target location as a covariate to the model.
1. [Add screen source covariate.](model-construction/010_010_hnb-add-screen-and-intermediates.md) Experimenting with adding the source of the screen data as a batch effect. *This was later removed and only a single screen was used (from the Broad).*
1. [Cancer gene comutation matrix example.](model-construction/010_020_cancer-gene-matrix-covariate.md) Experimenting with how to build the comutation matrix and integrate it into the model.
1. [Cancer gene comutation matrix into the current model.](model-construction/010_022_cancer-gene-matrix-covariate_implementation.md) Example implementation of the cancer gene comutation covariate.
1. [Cancer gene comutation with CGC list.](model-construction/015_005_hnb-cgc-matrix-covariate_subsample.md) Building the cancer gene comutation covariate from the CGC cancer gene list.
1. [Experimentation with current model structure.](model-construction/020_single-lineage-model-testing.md) Some general exploration of the current model and how it behaves with real data.
1. [Experimenting with a simpler model.](model-construction/050_simplify-single-lineage-model.md) The model was simplified in order to move the project along.
1. [Introduce a chromosome-varying effect.](model-construction/055_chromosome-varying-effects.md) Introduction of another hierarchical layer in the cell line-effect variables to account for differential sensitivity of each chromosome of each cell line.

The following notebooks were analyses of running the model on the lineages prostate, liver, and colorectal.
These lineages were chosen because they represent examples of datasets of different sizes and complexities (e.g. number of cancer genes in the comutation matrix.)
The model of MCMC parameters were altered slightly from version to version.
(The posterior data for the models no longer exists, so these notebooks are retained primarily for reference.)

1. [Experimental run with prostate (model v001).](model-construction/022_single-lineage-prostate-inspection_001.md)
1. [Experimental run with prostate (model v002).](model-construction/023_single-lineage-prostate-inspection_002.md)
1. [Experimental run with prostate (model v003).](model-construction/024_single-lineage-prostate-inspection_003.md)
1. [Experimental run with prostate (model v004).](model-construction/025_single-lineage-prostate-inspection_004.md)
1. [Experimental run with multiple lineages (model v004).](model-construction/026_single-lineage-multiple-inspection_004.md)
1. [Experimental run with colorectal (model v004).](model-construction/027_single-lineage-colorectal-inspection_004.md)
1. [Experimental run with colorectal (model v005).](model-construction/028_single-lineage-colorectal-inspection_005.md)
1. [Experimental run with prostate (model v006).](model-construction/029_single-lineage-prostate-inspection_006.md)
1. [Experimental run with prostate (model v007).](model-construction/030_single-lineage-prostate-inspection_007.md)
1. [Experimental run with liver (model v007).](model-construction/031_single-lineage-liver-inspection_007.md)
1. [Experimental run with colorectal (model v007).](model-construction/032_single-lineage-colorectal-inspection_007.md)
1. [Experimental run with prostate (model v008).](model-construction/033_single-lineage-prostate-inspection_008.md)
1. [Experimental run with prostate (model v009).](model-construction/034_single-lineage-prostate-inspection_009.md)
1. [Experimental run with colorectal (model v009).](model-construction/035_single-lineage-colorectal-inspection_009.md)

## [Model Analysis](model-analysis)

![reproducibility-full](https://img.shields.io/badge/reproducibility-full-brightgreen.svg?style=flat)

Final analyses of fitting the final model to all of the cell line lineages.
Many of the figures were generated using the results of these analyses.

1. [First look at fit models](model-analysis/100_100_lineage-models-analysis.md). First, preliminary look at the fit models for all of the lineages.
1. [Model diagnositcs.](model-analysis/100_101_model-diagnostics.md) Collect MCMC and model-fit diagnostics.
1. [Simple descriptions of models.](model-analysis/100_102_model-descriptions.md) Summary statistics on the dynamic features of the models (e.g. number of cancer genes included).
1. [Gene essentiality.](model-analysis/100_105_essentiality-comparisons.md) Analyze base "essentiality" of genes by lineage.
1. [Molecular and cellular covariates.](model-analysis/100_106_molecular-cellular-covariates.md) Analysis of the covariates for molecular data on the gene and cell line levels.
1. [Gene mutation effects.](model-analysis/100_110_gene-mutation-effect.md) Effects of the mutation of the target gene and discovery of putative driver genes.
1. [Cancer gene comutation analysis.](model-analysis/100_120_cancer-gene-comut-analysis.md) Analysis of the cancer gene comutation variables to discover possible synthetic lethal interactions.

Some other notebooks are in this directory though they were from earlier analyses and can be ignored.

1. [Preliminary analysis with a few lineages.](model-anaysis/005_010_hierarchical-nb-analysis.md)
1. [Small analysis with more lineages.](model-analysis/005_015_brief-analysis-with-larger-data.md)
1. [Analyzing the molecular covariates with the preliminary models.](model-analysis/010_005_hnb-molecular-covs-largesubsample.md)


## [Experimentation](experimentation)

![reproducibility-limited](https://img.shields.io/badge/reproducibility-limited-orange.svg?style=flat)

These are small, experimental notebooks that exist to quickly test ideas for other uses.
They may not be fully reproducible, but exist primarily for reference.

1. [Multiple varying intercepts example.](experimentation/999_005_experimentation.md) An example model for fitting multiple varying intercepts.
1. [Saving and loading PyMC3 models and samples.](experimentation/999_010_saving-and-loading-models.md) Testing various methods for wrapping `pm.sample()` for automatic caching and re-loading.
1. [Fitting splines.](experimentation/999_015_splines-in-pymc3.md) How to fit splines with PyMC3. I have yet to get a working multi-level model.
1. [Simple SBC example.](999_020_simulation-based-calibration.md) A quick proof-of-concept for simulation-based calibration workflow.
1. [Combining MCMC chains.](999_025_combining-chains.md) How to combine MCMC chains run separately into a single ArviZ `InferenceData` object.
1. [Scaling copy number data.](999_030_scaling-copy-number.md) Effects of different transformations on copy number data.
1. [Scaling RNA expression data.](999_031_scaling-rna-expression.md) Effects of different transformations on RNA expression data.
1. [Mixing centered and non-centered parameterizations.](999_032_mixed-centered-parameterization-pymc3-model.md)
1. [PyMC vs. Stan.](experimentation/999_005_experimentation.md) Simple comparison of MCMC speed and performance between PyMC and Stan building the same models.

## [Miscellaneous](misc)

Miscellaneous notebooks for testing ideas not related to model design.

1. [PyMC custom callback.](misc/010_010_pymc3-callback-tracking.md)
1. [Different PyMC backends.](010_015_pymc-backends.md)
1. [Different PyMC backends on O2.](misc/010_015_pymc-backends_o2.md)
