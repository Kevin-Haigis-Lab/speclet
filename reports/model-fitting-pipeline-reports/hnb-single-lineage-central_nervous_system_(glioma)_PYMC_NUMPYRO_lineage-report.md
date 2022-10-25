# Single-lineage Hierarchical Model Report

## Setup

### Imports


```python
import logging
from math import ceil
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
```


```python
from speclet.analysis.arviz_analysis import describe_mcmc, extract_coords_param_names
from speclet.bayesian_models.lineage_hierarchical_nb import LineageHierNegBinomModel
from speclet.data_processing.common import head_tail
from speclet.data_processing.vectors import squish_array
from speclet.io import project_root
from speclet.loggers import set_console_handler_level
from speclet.managers.posterior_data_manager import PosteriorDataManager
from speclet.project_configuration import arviz_config
from speclet.project_enums import ModelFitMethod
```

    WARNING (aesara.tensor.blas): Using NumPy C-API based implementation for BLAS functions.



```python
set_console_handler_level(logging.INFO)
%config InlineBackend.figure_format = "retina"
arviz_config()
```

Parameters for papermill:

- `MODEL_NAME`: name of the model
- `FIT_METHOD`: method used to fit the model
- `CONFIG_PATH`: path to configuration file
- `ROOT_CACHE_DIR`: path to the root caching directory

### Papermill parameters


```python
CONFIG_PATH = ""
MODEL_NAME = ""
FIT_METHOD = ""
ROOT_CACHE_DIR = ""
```


```python
# Parameters
MODEL_NAME = "hnb-single-lineage-central_nervous_system_(glioma)"
FIT_METHOD = "PYMC_NUMPYRO"
CONFIG_PATH = "models/model-configs.yaml"
ROOT_CACHE_DIR = "models"
```


```python
_fit_method = ModelFitMethod(FIT_METHOD)
postman = PosteriorDataManager(
    name=MODEL_NAME,
    fit_method=_fit_method,
    config_path=project_root() / Path(CONFIG_PATH),
    posterior_dir=project_root() / Path(ROOT_CACHE_DIR),
)
```


```python
assert isinstance(postman.bayes_model, LineageHierNegBinomModel)
```


```python
print(postman.read_description())
```

    config. name: 'hnb-single-lineage-central_nervous_system_(glioma)'
    model name: 'LineageHierNegBinomModel'
    model version: '0.1.3'
    model description: A hierarchical negative binomial generalized linear model for one lineage.
    fit method: 'PYMC_NUMPYRO'

    --------------------------------------------------------------------------------

    CONFIGURATION

    {
        "name": "hnb-single-lineage-central_nervous_system_(glioma)",
        "description": " Single lineage hierarchical negative binomial model for central_nervous_system_(glioma) data. ",
        "active": true,
        "model": "LINEAGE_HIERARCHICAL_NB",
        "data_file": "modeling_data/sublineage-broad-modeling-data/depmap-modeling-data_central_nervous_system_(glioma).csv",
        "model_kwargs": {
            "lineage": "central_nervous_system_(glioma)",
            "min_n_cancer_genes": 4,
            "min_frac_cancer_genes": 0.05,
            "top_n_cancer_genes": 10
        },
        "sampling_kwargs": {
            "pymc_mcmc": null,
            "pymc_advi": null,
            "pymc_numpyro": {
                "draws": 1000,
                "tune": 2000,
                "chains": 4,
                "target_accept": 0.99,
                "progress_bar": true,
                "chain_method": "parallel",
                "postprocessing_backend": "cpu",
                "idata_kwargs": {
                    "log_likelihood": false
                },
                "nuts_kwargs": {
                    "step_size": 0.01
                }
            }
        },
        "split_posterior_when_combining_chains": false
    }

    --------------------------------------------------------------------------------

    POSTERIOR

    <xarray.Dataset>
    Dimensions:                    (chain: 4, draw: 1000, delta_genes_dim_0: 14,
                                    delta_genes_dim_1: 18119, sgrna: 71062,
                                    delta_cells_dim_0: 2, delta_cells_dim_1: 51,
                                    cell_chrom: 1173, genes_chol_cov_dim_0: 105,
                                    cells_chol_cov_dim_0: 3,
                                    genes_chol_cov_corr_dim_0: 14,
                                    genes_chol_cov_corr_dim_1: 14,
                                    genes_chol_cov_stds_dim_0: 14, gene: 18119,
                                    cancer_gene: 10, cells_chol_cov_corr_dim_0: 2,
                                    cells_chol_cov_corr_dim_1: 2,
                                    cells_chol_cov_stds_dim_0: 2, cell_line: 51)
    Coordinates: (12/19)
      * chain                      (chain) int64 0 1 2 3
      * draw                       (draw) int64 0 1 2 3 4 5 ... 995 996 997 998 999
      * delta_genes_dim_0          (delta_genes_dim_0) int64 0 1 2 3 ... 10 11 12 13
      * delta_genes_dim_1          (delta_genes_dim_1) int64 0 1 2 ... 18117 18118
      * sgrna                      (sgrna) object 'AAAAAAATCCAGCAATGCAG' ... 'TTT...
      * delta_cells_dim_0          (delta_cells_dim_0) int64 0 1
        ...                         ...
      * gene                       (gene) object 'A1BG' 'A1CF' ... 'ZZEF1' 'ZZZ3'
      * cancer_gene                (cancer_gene) object 'APC' 'CDKN2C' ... 'TP53'
      * cells_chol_cov_corr_dim_0  (cells_chol_cov_corr_dim_0) int64 0 1
      * cells_chol_cov_corr_dim_1  (cells_chol_cov_corr_dim_1) int64 0 1
      * cells_chol_cov_stds_dim_0  (cells_chol_cov_stds_dim_0) int64 0 1
      * cell_line                  (cell_line) object 'ACH-000036' ... 'ACH-001624'
    Data variables: (12/35)
        mu_mu_a                    (chain, draw) float64 ...
        mu_b                       (chain, draw) float64 ...
        delta_genes                (chain, draw, delta_genes_dim_0, delta_genes_dim_1) float64 ...
        delta_a                    (chain, draw, sgrna) float64 ...
        mu_mu_m                    (chain, draw) float64 ...
        delta_cells                (chain, draw, delta_cells_dim_0, delta_cells_dim_1) float64 ...
        ...                         ...
        sigma_mu_k                 (chain, draw) float64 ...
        sigma_mu_m                 (chain, draw) float64 ...
        mu_k                       (chain, draw, cell_line) float64 ...
        mu_m                       (chain, draw, cell_line) float64 ...
        k                          (chain, draw, cell_chrom) float64 ...
        m                          (chain, draw, cell_chrom) float64 ...
    Attributes:
        created_at:           2022-10-02 13:39:36.391602
        arviz_version:        0.12.1
        model_name:           LineageHierNegBinomModel
        model_version:        0.1.3
        model_doc:            A hierarchical negative binomial generalized linear...
        previous_created_at:  ['2022-10-02 13:39:36.391602', '2022-10-02T17:12:10...

    --------------------------------------------------------------------------------

    SAMPLE STATS

    <xarray.Dataset>
    Dimensions:          (chain: 4, draw: 1000)
    Coordinates:
      * chain            (chain) int64 0 1 2 3
      * draw             (draw) int64 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
    Data variables:
        acceptance_rate  (chain, draw) float64 ...
        step_size        (chain, draw) float64 ...
        diverging        (chain, draw) bool ...
        energy           (chain, draw) float64 ...
        n_steps          (chain, draw) int64 ...
        tree_depth       (chain, draw) int64 ...
        lp               (chain, draw) float64 ...
    Attributes:
        created_at:           2022-10-02 13:39:36.391602
        arviz_version:        0.12.1
        previous_created_at:  ['2022-10-02 13:39:36.391602', '2022-10-02T17:12:10...

    --------------------------------------------------------------------------------

    MCMC DESCRIPTION

    date created: 2022-10-02 13:39
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.731, 0.687, 0.708, 0.723
    avg. step size: 0.008, 0.007, 0.007, 0.008
    avg. accept prob.: 0.987, 0.99, 0.99, 0.986
    avg. tree depth: 9.0, 9.0, 9.0, 9.0



```python
postman.load_all()
```

    [INFO] 2022-10-02 19:40:36 [(lineage_hierarchical_nb.py:data_processing_pipeline:323] Processing data for modeling.
    [INFO] 2022-10-02 19:40:36 [(lineage_hierarchical_nb.py:data_processing_pipeline:324] LFC limits: (-5.0, 5.0)
    [WARNING] 2022-10-02 19:42:49 [(lineage_hierarchical_nb.py:data_processing_pipeline:382] number of data points dropped: 71170
    [INFO] 2022-10-02 19:42:52 [(lineage_hierarchical_nb.py:target_gene_is_mutated_vector:630] number of genes mutated in all cells lines: 0
    [INFO] 2022-10-02 19:42:56 [(cancer_gene_mutation_matrix.py:_trim_cancer_genes:77] Dropping 14 cancer genes.


## Fit diagnostics


```python
def _plot_rhat_boxplots(pm: PosteriorDataManager) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=pm.posterior_summary, x="var_name", y="r_hat", ax=ax)
    ax.tick_params("x", rotation=90)
    ax.set_ylabel(r"$\widehat{R}$")
    ax.set_ylim(0.999, None)
    plt.show()


def _plot_ess_hist(pm: PosteriorDataManager) -> None:
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 4)
    )
    sns.histplot(data=pm.posterior_summary, x="ess_bulk", ax=axes[0])
    axes[0].set_title("ESS (bulk)")
    sns.histplot(data=pm.posterior_summary, x="ess_tail", ax=axes[1])
    axes[1].set_title("ESS (tail)")
    for ax in axes.flatten():
        ax.set_xlim(0, None)
    fig.tight_layout()
    plt.show()
```


```python
if postman.fit_method in {ModelFitMethod.PYMC_NUMPYRO, ModelFitMethod.PYMC_MCMC}:
    _plot_rhat_boxplots(postman)
    _plot_ess_hist(postman)
    print("=" * 60)
    describe_mcmc(postman.trace)
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_16_0.png)





![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_16_1.png)



    ============================================================
    date created: 2022-10-02 13:39
    sampled 4 chains with (unknown) tuning steps and 1,000 draws
    num. divergences: 0, 0, 0, 0
    percent divergences: 0.0, 0.0, 0.0, 0.0
    BFMI: 0.731, 0.687, 0.708, 0.723
    avg. step size: 0.008, 0.007, 0.007, 0.008
    avg. accept prob.: 0.987, 0.99, 0.99, 0.986
    avg. tree depth: 9.0, 9.0, 9.0, 9.0




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_16_3.png)



## Posterior analysis


```python
ignore_vars = ["mu_a", "a", "b", "d", "f", "h", "k", "m"]
ignore_vars = [f"~^{v}$" for v in ignore_vars]
ignore_vars += postman.bayes_model.vars_regex(postman.fit_method)
az.plot_trace(postman.trace, var_names=ignore_vars, filter_vars="regex")
plt.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_18_0.png)




```python
sigma_vars = list(
    {v for v in postman.posterior_summary["var_name"].unique() if "sigma" in v}
)
sigma_vars.sort()

axes = az.plot_forest(
    postman.trace,
    var_names=sigma_vars,
    combined=True,
    figsize=(6, len(sigma_vars) * 0.6),
)
axes[0].axvline(0, color="k", lw=0.8, zorder=1)
plt.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_19_0.png)




```python
sgrna_to_gene_map = (
    postman.data.copy()[["hugo_symbol", "sgrna"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
```


```python
_cgs = postman.trace.posterior.coords.get("cancer_gene")
cancer_genes: list[str] = [] if _cgs is None else _cgs.values.tolist()
print(cancer_genes)
```

    ['APC', 'CDKN2C', 'EGFR', 'KMT2C', 'KMT2D', 'MLH1', 'MTOR', 'NF1', 'PTEN', 'TP53']



```python
example_genes = ["KRAS", "BRAF", "CTNNB1", "TP53", "PTEN", "STK11", "MDM2"]

var_names = ["mu_mu_a", "mu_a", "a", "mu_b", "b", "d", "f"]
if len(cancer_genes) > 0:
    var_names += ["h"]

for example_gene in example_genes:
    display(Markdown(f"🧬 **target gene: *{example_gene}***"))
    example_gene_sgrna = sgrna_to_gene_map.query(f"hugo_symbol == '{example_gene}'")[
        "sgrna"
    ].tolist()
    axes = az.plot_forest(
        postman.trace,
        var_names=var_names,
        coords={"gene": [example_gene], "sgrna": example_gene_sgrna},
        combined=True,
        figsize=(6, 4 + 0.3 * len(cancer_genes)),
    )
    axes[0].axvline(0, color="k", lw=0.8, zorder=1)
    plt.show()
```


🧬 **target gene: *KRAS***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_1.png)




🧬 **target gene: *BRAF***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_3.png)




🧬 **target gene: *CTNNB1***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_5.png)




🧬 **target gene: *TP53***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_7.png)




🧬 **target gene: *PTEN***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_9.png)




🧬 **target gene: *STK11***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_11.png)




🧬 **target gene: *MDM2***




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_22_13.png)




```python
if len(cancer_genes) > 0:
    h_post_summary = (
        postman.posterior_summary.query("var_name == 'h'")
        .reset_index(drop=True)
        .pipe(
            extract_coords_param_names,
            names=["hugo_symbol", "cancer_gene"],
            col="parameter",
        )
    )

    _, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=h_post_summary, x="mean", hue="cancer_gene", ax=ax)
    ax.set_xlabel(r"$\bar{h}_g$ posterior")
    ax.set_ylabel("density")
    ax.get_legend().set_title("cancer gene\ncomut.")
    plt.show()

    h_post_df = (
        postman.posterior_summary.query("var_name == 'h'")
        .reset_index(drop=True)
        .pipe(
            extract_coords_param_names,
            names=["target_gene", "cancer_gene"],
            col="parameter",
        )
        .pivot_wider("target_gene", names_from="cancer_gene", values_from="mean")
        .set_index("target_gene")
    )

    h_gene_var = h_post_df.values.var(axis=1)
    idx = h_gene_var >= np.quantile(h_gene_var, q=0.8)
    h_post_df_topvar = h_post_df.loc[idx, :]

    width = max(2, len(cancer_genes) * 0.5)
    sns.clustermap(
        h_post_df_topvar,
        cmap="seismic",
        center=0,
        figsize=(width, 12),
        col_cluster=len(cancer_genes) > 1,
        cbar_pos=(1, 0.4, 0.05, 0.2),
        dendrogram_ratio=(0.1, 0.1 if len(cancer_genes) > 1 else 0),
    )
    plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_23_0.png)



    /home/jc604/.conda/envs/speclet_smk/lib/python3.10/site-packages/seaborn/matrix.py:654: UserWarning: Clustering large matrix with scipy. Installing `fastcluster` may give better performance.
      warnings.warn(msg)




![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_23_2.png)




```python
tp53_muts = (
    postman.valid_data.query("hugo_symbol == 'TP53'")[["depmap_id", "is_mutated"]]
    .copy()
    .drop_duplicates()
    .rename(columns={"is_mutated": "TP53 mut."})
    .reset_index(drop=True)
)

cell_line_vars = ["mu_k", "mu_m"]
cell_line_effects = (
    az.summary(postman.trace, var_names=cell_line_vars, kind="stats")
    .pipe(extract_coords_param_names, names=["depmap_id"])
    .assign(var_name=lambda d: [x.split("[")[0] for x in d.index.values])
    .pivot_wider(
        index="depmap_id",
        names_from="var_name",
        values_from=["mean", "hdi_5.5%", "hdi_94.5%"],
    )
    .merge(tp53_muts, on="depmap_id")
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.axhline(0, color="k", zorder=1)
ax.axvline(0, color="k", zorder=1)

ax.vlines(
    x=cell_line_effects["mean_mu_k"],
    ymin=cell_line_effects["hdi_5.5%_mu_m"],
    ymax=cell_line_effects["hdi_94.5%_mu_m"],
    alpha=0.5,
    color="gray",
)
ax.hlines(
    y=cell_line_effects["mean_mu_m"],
    xmin=cell_line_effects["hdi_5.5%_mu_k"],
    xmax=cell_line_effects["hdi_94.5%_mu_k"],
    alpha=0.5,
    color="gray",
)
mut_pal = {True: "tab:red", False: "tab:blue"}
sns.scatterplot(
    data=cell_line_effects,
    x="mean_mu_k",
    y="mean_mu_m",
    hue="TP53 mut.",
    palette=mut_pal,
    ax=ax,
    zorder=10,
)

sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_24_0.png)




```python
cell_chromosome_map = (
    postman.valid_data[["depmap_id", "sgrna_target_chr", "cell_chrom"]]
    .drop_duplicates()
    .sort_values("cell_chrom")
    .reset_index(drop=True)
)
chromosome_effect_post = (
    az.summary(postman.trace, var_names=["k", "m"], kind="stats")
    .pipe(extract_coords_param_names, names="cell_chrom")
    .assign(var_name=lambda d: [p[0] for p in d.index.values])
    .merge(cell_chromosome_map, on="cell_chrom")
)

ncells = chromosome_effect_post["depmap_id"].nunique()
ncols = 3
nrows = ceil(ncells / 3)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, nrows * 3))
for ax, (cell, data_c) in zip(
    axes.flatten(), chromosome_effect_post.groupby("depmap_id")
):
    plot_df = data_c.pivot_wider(
        index="sgrna_target_chr",
        names_from="var_name",
        values_from=["mean", "hdi_5.5%", "hdi_94.5%"],
    )
    cell_eff = (
        cell_line_effects.copy().query(f"depmap_id == '{cell}'").reset_index(drop=True)
    )

    ax.axhline(0, color="k", zorder=1)
    ax.axvline(0, color="k", zorder=1)

    ax.vlines(
        x=plot_df["mean_k"],
        ymin=plot_df["hdi_5.5%_m"],
        ymax=plot_df["hdi_94.5%_m"],
        alpha=0.5,
        zorder=5,
    )
    ax.hlines(
        y=plot_df["mean_m"],
        xmin=plot_df["hdi_5.5%_k"],
        xmax=plot_df["hdi_94.5%_k"],
        alpha=0.5,
        zorder=5,
    )
    sns.scatterplot(data=plot_df, x="mean_k", y="mean_m", ax=ax, zorder=10)

    ax.vlines(
        x=cell_eff["mean_mu_k"],
        ymin=cell_eff["hdi_5.5%_mu_m"],
        ymax=cell_eff["hdi_94.5%_mu_m"],
        alpha=0.8,
        zorder=15,
        color="red",
    )
    ax.hlines(
        y=cell_eff["mean_mu_m"],
        xmin=cell_eff["hdi_5.5%_mu_k"],
        xmax=cell_eff["hdi_94.5%_mu_k"],
        alpha=0.8,
        zorder=15,
        color="red",
    )
    sns.scatterplot(
        data=cell_eff, x="mean_mu_k", y="mean_mu_m", ax=ax, zorder=20, color="red"
    )
    ax.set_title(cell)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

for ax in axes[-1, :]:
    ax.set_xlabel("$k$")
for ax in axes[:, 0]:
    ax.set_ylabel("$m$")
for ax in axes.flatten()[ncells:]:
    ax.axis("off")

fig.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_25_0.png)




```python
for v, data_v in chromosome_effect_post.groupby("var_name"):
    df = (
        data_v.copy()
        .reset_index(drop=True)
        .pivot_wider(
            index="depmap_id", names_from="sgrna_target_chr", values_from="mean"
        )
        .set_index("depmap_id")
    )
    height = max(2.5, df.shape[0] * 0.15)
    cg = sns.clustermap(
        df,
        cmap="seismic",
        figsize=(7, height),
        cbar_pos=(1, 0.4, 0.05, 0.2),
        dendrogram_ratio=(0.1, 0.1),
        center=0,
    )
    cg.ax_col_dendrogram.set_title(f"variable: ${v}$")
    plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_26_0.png)





![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_26_1.png)




```python
chrom_target_positions = (
    postman.valid_data.copy()[
        [
            "hugo_symbol",
            "sgrna",
            "sgrna_target_chr",
            "sgrna_target_pos",
            "copy_number",
            "cn_gene",
            "depmap_id",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

d_post = (
    postman.posterior_summary.query("var_name == 'd'")
    .reset_index(drop=True)
    .pipe(extract_coords_param_names, names=["hugo_symbol"], col="parameter")
    .merge(chrom_target_positions, on="hugo_symbol", validate="one_to_many")
    .assign(cn_effect=lambda d: d["mean"] * d["cn_gene"])
)
```


```python
n_chroms = d_post["sgrna_target_chr"].nunique()
n_cols = 4
n_rows = ceil(n_chroms / n_cols)

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(n_cols * 3, n_rows * 2),
    sharex=False,
    sharey=True,
)

for ax, (chrom, data_c) in zip(axes.flatten(), d_post.groupby("sgrna_target_chr")):
    ax.set_title(f"chromsome {chrom}")

    # Plot CN for each cell line over the chromosome.
    cell_cn_data = (
        chrom_target_positions.copy()
        .query(f"sgrna_target_chr == '{chrom}'")
        .reset_index(drop=True)
        .assign(
            copy_number=lambda d: squish_array(d["copy_number"] - 1, lower=-1, upper=1)
        )
    )
    for _, data_cell in cell_cn_data.groupby("depmap_id"):
        sns.lineplot(
            data=data_cell,
            x="sgrna_target_pos",
            y="copy_number",
            alpha=0.2,
            lw=0.5,
            color="k",
            drawstyle="steps-pre",
            ax=ax,
            zorder=1,
        )

    # Plot the gene copy number effects over the chromosome.
    sns.scatterplot(
        data=data_c,
        x="sgrna_target_pos",
        y="cn_effect",
        ax=ax,
        color="tab:blue",
        s=2,
        alpha=0.3,
        edgecolor=None,
        zorder=5,
    )

    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xlim(data_c["sgrna_target_pos"].min(), data_c["sgrna_target_pos"].max())
    # break

for ax in axes.flatten()[n_chroms:]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel("$d$")
for ax in axes.flatten()[(n_chroms - 4) : n_chroms]:
    ax.set_xlabel("chromosome position")

fig.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_28_0.png)




```python
def _plot_corr_heatmap(
    trace: az.InferenceData, corr_var_name: str, var_names: list[str]
) -> None:
    corr_post = (
        az.summary(trace, var_names=[corr_var_name], kind="stats")
        .pipe(extract_coords_param_names, names=["d1", "d2"])
        .astype({"d1": int, "d2": int})
        .assign(
            p1=lambda d: [var_names[i] for i in d["d1"]],
            p2=lambda d: [var_names[i] for i in d["d2"]],
        )
        .assign(
            p1=lambda d: pd.Categorical(
                d["p1"], categories=d["p1"].unique(), ordered=True
            )
        )
        .assign(
            p2=lambda d: pd.Categorical(
                d["p2"], categories=d["p1"].cat.categories, ordered=True
            )
        )
    )
    plot_df = corr_post.pivot_wider("p1", "p2", "mean").set_index("p1")
    ax = sns.heatmap(
        plot_df,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    ax.set_ylabel(None)
    plt.show()
    return None
```


```python
# TODO: change get from the trace when included in `posterior.coords`.
genes_var_names = ["mu_a", "b", "d", "f"] + [f"h[{g}]" for g in cancer_genes]
_plot_corr_heatmap(postman.trace, "genes_chol_cov_corr", var_names=genes_var_names)

# TODO: change to use "cells_params" when that is included as a coord.
cells_var_names = postman.model_data_struct.coords["cells_params"]
_plot_corr_heatmap(postman.trace, "cells_chol_cov_corr", var_names=cells_var_names)
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_30_0.png)





![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_30_1.png)




```python
if len(cancer_genes) > 1:
    cancer_gene_mutants = (
        postman.valid_data.filter_column_isin("hugo_symbol", cancer_genes)[
            ["hugo_symbol", "depmap_id", "is_mutated"]
        ]
        .drop_duplicates()
        .sort_values(["hugo_symbol", "depmap_id"])
        .pivot_wider("depmap_id", names_from="hugo_symbol", values_from="is_mutated")
        .set_index("depmap_id")
    )

    sns.clustermap(
        cancer_gene_mutants,
        cmap="gray_r",
        xticklabels=1,
        yticklabels=1,
        figsize=(3, 9),
        cbar_pos=(1, 0.4, 0.05, 0.2),
    )
    plt.show()

    sns.clustermap(
        cancer_gene_mutants.corr(),
        cmap="seismic",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(len(cancer_genes) * 0.5, len(cancer_genes) * 0.5),
    )
    plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_31_0.png)





![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_31_1.png)




```python
top_n = 5
top_b_hits = (
    postman.posterior_summary.query("var_name == 'b'")
    .sort_values("mean")
    .reset_index(drop=True)
    .pipe(extract_coords_param_names, names=["hugo_symbol"], col="parameter")
    .pipe(head_tail, n=top_n)
)

negative_b = top_b_hits["hugo_symbol"][:top_n].values
positive_b = top_b_hits["hugo_symbol"][top_n:].values


fig, axes = plt.subplots(2, top_n, figsize=(12, 6))

for i, genes in enumerate([positive_b, negative_b]):
    for j, gene in enumerate(genes):
        ax = axes[i, j]
        ax.set_title(gene)
        obs_data = postman.valid_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(
            data=obs_data, x="m_rna_gene", y="lfc", ax=ax, edgecolor=None, s=20
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("log RNA expression")
fig.supylabel("log-fold change")

fig.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_32_0.png)




```python
top_n = 5
top_d_hits = (
    postman.posterior_summary.query("var_name == 'd'")
    .sort_values("mean")
    .reset_index(drop=True)
    .pipe(extract_coords_param_names, names=["hugo_symbol"], col="parameter")
    .pipe(head_tail, n=top_n)
)

negative_d = top_d_hits["hugo_symbol"][:top_n].values
positive_d = top_d_hits["hugo_symbol"][top_n:].values


fig, axes = plt.subplots(2, top_n, figsize=(12, 6))

for i, genes in enumerate([positive_d, negative_d]):
    for j, gene in enumerate(genes):
        ax = axes[i, j]
        ax.set_title(gene)
        obs_data = postman.valid_data.query(f"hugo_symbol == '{gene}'")
        sns.scatterplot(data=obs_data, x="cn_gene", y="lfc", ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel(None)


fig.supxlabel("copy number")
fig.supylabel("log-fold change")
fig.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_33_0.png)



## Model predictions


```python
n_examples = 40
n_chains, n_draws, n_data = postman.trace.posterior_predictive["ct_final"].shape
ex_draws_idx = np.random.choice(
    np.arange(n_draws), size=n_examples // n_chains, replace=False
)
example_ppc_draws = postman.trace.posterior_predictive["ct_final"][
    :, ex_draws_idx, :
].values.reshape(-1, n_data)

fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharex=False, sharey=False)
ax1 = axes[0]
ax2 = axes[1]

pp_avg = postman.trace.posterior_predictive["ct_final"].mean(axis=(0, 1))

for i in range(example_ppc_draws.shape[0]):
    sns.kdeplot(
        x=np.log10(example_ppc_draws[i, :] + 1), alpha=0.2, color="tab:blue", ax=ax1
    )

sns.kdeplot(x=np.log10(pp_avg + 1), color="tab:orange", ax=ax1)
sns.kdeplot(x=np.log10(postman.valid_data["counts_final"] + 1), color="k", ax=ax1)
ax1.set_xlabel("log10(counts final + 1)")
ax1.set_ylabel("density")

x_max = 1000
x_cut = x_max * 5
for i in range(example_ppc_draws.shape[0]):
    x = example_ppc_draws[i, :]
    x = x[x < x_cut]
    sns.kdeplot(x=x, alpha=0.2, color="tab:blue", ax=ax2)

sns.kdeplot(x=pp_avg[pp_avg < x_cut], color="tab:orange", ax=ax2)
_obs = postman.valid_data["counts_final"].values
_obs = _obs[_obs < x_cut]
sns.kdeplot(x=_obs, color="k", ax=ax2)
ax2.set_xlabel("counts final")
ax2.set_ylabel("density")
ax2.set_xlim(0, x_max)

fig.suptitle("PPC")
fig.tight_layout()
plt.show()
```



![png](hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_files/hnb-single-lineage-central_nervous_system_%28glioma%29_PYMC_NUMPYRO_lineage-report_35_0.png)



---


```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2022-10-02

    Python implementation: CPython
    Python version       : 3.10.5
    IPython version      : 8.4.0

    Compiler    : GCC 10.3.0
    OS          : Linux
    Release     : 3.10.0-1160.76.1.el7.x86_64
    Machine     : x86_64
    Processor   : x86_64
    CPU cores   : 28
    Architecture: 64bit

    Hostname: compute-e-16-184.o2.rc.hms.harvard.edu

    Git branch: figures

    logging   : 0.5.1.2
    arviz     : 0.12.1
    matplotlib: 3.5.2
    numpy     : 1.23.1
    pandas    : 1.4.3
    seaborn   : 0.11.2
