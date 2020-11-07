# Analysis

Modeling construction and fitting.

## Building increasingly complex models

1. [Model 1.](005_005_model-experimentation-m1.md) Standard linear model of one gene using RNA expression as a predictor.
2. [Model 2.](005_007_model-experimentation-m2.md) A hierarchical linear model of multiple genes with a varying intercept and slope on RNA expression.
3. [Model 3.](005_009_model-experimentation-m3.md) (Failed) A hierarhcical model of multiple genes and cell lines with a varying intercept for each. The gene level model consisted of an intercept and slope for RNA expression.
4. [Model 4.](005_011_model-experimentation-m4.md) A hierarchical model of multiple genes and cell lines with a varying intercept for each and a slope for each gene on RNA expression.
5. [Model 5.](005_013_model-experimentation-m5.md) A multi-level hierarchical model with the main level consisting of two varying intercepts, one for the sgRNA and one for the cell line. The sgRNA varying intercept had an additional level where each guide came from a distribtuion for each gene.
6. [Model 6.](005_015_model-experimentation-m6.md) A heirarchcial model with varying effects for the intercept and slope pooling across sgRNA and gene. The slope is on synthetic copy number data.
7. [Model 7.](005_017_model-experimentation-m7.md) **(in progress)** A model with a single 2D varying intercept with one dimension for sgRNA and one for cell line. Then try to have the sgRNA dimension vary by gene.
