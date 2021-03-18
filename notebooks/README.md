# Notebooks

Modeling construction and fitting.

## [Model design experimentation](model_design_experimentation/)

These notebooks are for experimenting with various model designs, each one using different covariates and model structures.
They primarily use synthetic data and test whether the known values can be recovered.

1. [Model 1.](model_design_experimentation/005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
2. [Model 2.](model_design_experimentation/005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
3. [Model 3.](model_design_experimentation/005_009_model-experimentation-m3.md) (Failed) A hierarchical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
4. [Model 4.](model_design_experimentation/005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
5. [Model 5.](model_design_experimentation/005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribution for each gene.
6. [Model 6.](model_design_experimentation/005_015_model-experimentation-m6.md) A hierarchical model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
7. [Model 7.](model_design_experimentation/005_017_model-experimentation-m7.md) A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene. A model with two varying intercepts was also successfully fit in this notebook.

## [Modeling CRC data](crc_pipeline/)

The purpose of this series is to experiment with various ways of modeling the CRC data using PyMC3 models.
Various models will be designed, fit, and compared.
The results will also be compared to those from the CERES model produced by DepMap and the results of using the CERES dependency scores for modeling.

1. [Exploratory data analysis.](crc_pipeline/015_005_exploratory-data-analysis.md) Exploration of the CRC cell line data.
2. [Designing Model 2](crc_pipeline/015_010_m2-design.md) A scratch-pad for experimenting with designing Model 2.

## [Miscellaneous experimentation](experimentation/)

These are miscellaneous, small, experimental notebooks that exist to quickly test ideas for other uses.
They may not be fully reproducible, but exist primarily for reference.

1. [Multiple varying intercepts example.](experimentation/999_005_experimentation.md) An example model for fitting multiple varying intercepts.
2. [Saving and loading PyMC3 models and samples](experimentation/999_010_saving-and-loading-models.md) Testing various methods for wrapping `pm.sample()` for automatic caching and re-loading.
3. [Fitting splines](experimentation/999_015_splines-in-pymc3.md) How to fit splines with PyMC3. I have yet to get a working multi-level model.
