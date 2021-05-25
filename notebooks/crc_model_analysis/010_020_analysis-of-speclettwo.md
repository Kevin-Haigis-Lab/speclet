```python
%load_ext autoreload
%autoreload 2
```

```python
import re
import string
import warnings
from pathlib import Path
from time import time

import arviz as az
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from theano import tensor as tt
```

```python
from src.data_processing import achilles as achelp
from src.data_processing import common as dphelp
from src.io import cache_io
from src.modeling import pymc3_analysis as pmanal
from src.modeling import pymc3_sampling_api as pmapi
from src.models.speclet_two import SpecletTwo
from src.plot.color_pal import FitMethodColors, ModelColors, SeabornColor
```

```python
notebook_tic = time()

warnings.simplefilter(action="ignore", category=UserWarning)

gg.theme_set(
    gg.theme_bw()
    + gg.theme(
        figure_size=(4, 4),
        axis_ticks_major=gg.element_blank(),
        strip_background=gg.element_blank(),
    )
)
%config InlineBackend.figure_format = "retina"

RANDOM_SEED = 255
np.random.seed(RANDOM_SEED)

cache_dir = cache_io.default_cache_dir() / "pymc3_model_cache"
cache_dir

HDI_PROB = 0.89
```

```python
eb = gg.element_blank()
```

```python
sp_two = SpecletTwo(
    "SpecletTwo-debug",
    root_cache_dir=cache_dir / "SpecletTwo-debug",
    debug=True,
    kras_cov=False,
)
sp_two.advi_sampling_params.n_iterations = 40000
```

```python
sp_two.build_model()
if sp_two.cache_manager.mcmc_cache_exists():
    _ = sp_two.mcmc_sample_model()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[04/29/21 10:59:44] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Calling `model_specification<span style="font-weight: bold">()</span>` method.     <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:155</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Beginning PyMC3 model specification.           <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/models/speclet_two.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_two.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:78</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[04/29/21 10:59:45] </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping data points of sgRNA that    <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:125</span>
                             map to multiple genes.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #800000; text-decoration-color: #800000">WARNING </span> Dropping data points with missing     <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/managers/model_data_managers.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">model_data_managers.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:131</span>
                             copy number.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Getting Theano shared variables.               <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/models/speclet_two.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_two.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:85</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">                    </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Creating PyMC3 model.                          <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/models/speclet_two.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_two.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:93</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[04/29/21 11:00:37] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Returning results from cache.               <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:247</span>
</pre>

```python
_ = sp_two.advi_sample_model(ignore_cache=False)
```

    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/data.py:316: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
    /home/jc604/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/data.py:316: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[04/29/21 11:01:41] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Returning results from cache.               <a href="file:///n/data2/dfci/cancerbio/haigis/Cook/speclet/src/models/speclet_model.py"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">speclet_model.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:367</span>
</pre>

```python
az_sptwo_mcmc = pmapi.convert_samples_to_arviz(sp_two.model, sp_two.mcmc_results)
az_sptwo_advi = pmapi.convert_samples_to_arviz(sp_two.model, sp_two.advi_results)
```

    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-9-b63253ec3df2> in <module>
    ----> 1 az_sptwo_mcmc = pmapi.convert_samples_to_arviz(sp_two.model, sp_two.mcmc_results)
          2 az_sptwo_advi = pmapi.convert_samples_to_arviz(sp_two.model, sp_two.advi_results)


    /n/data2/dfci/cancerbio/haigis/Cook/speclet/src/modeling/pymc3_sampling_api.py in convert_samples_to_arviz(model, res)
         68         az.InferenceData: A standard ArviZ data object.
         69     """
    ---> 70     return az.from_pymc3(
         71         trace=res.trace,
         72         model=model,


    ~/.conda/envs/speclet/lib/python3.9/site-packages/arviz/data/io_pymc3.py in from_pymc3(trace, prior, posterior_predictive, log_likelihood, coords, dims, model, save_warmup, density_dist_obs)
        561     InferenceData
        562     """
    --> 563     return PyMC3Converter(
        564         trace=trace,
        565         prior=prior,


    ~/.conda/envs/speclet/lib/python3.9/site-packages/arviz/data/io_pymc3.py in to_inference_data(self)
        495         """
        496         id_dict = {
    --> 497             "posterior": self.posterior_to_xarray(),
        498             "sample_stats": self.sample_stats_to_xarray(),
        499             "log_likelihood": self.log_likelihood_to_xarray(),


    ~/.conda/envs/speclet/lib/python3.9/site-packages/arviz/data/base.py in wrapped(cls, *args, **kwargs)
         44                 if all([getattr(cls, prop_i) is None for prop_i in prop]):
         45                     return None
    ---> 46             return func(cls, *args, **kwargs)
         47
         48         return wrapped


    ~/.conda/envs/speclet/lib/python3.9/site-packages/arviz/data/io_pymc3.py in posterior_to_xarray(self)
        264             if self.posterior_trace:
        265                 data[var_name] = np.array(
    --> 266                     self.posterior_trace.get_values(var_name, combine=False, squeeze=False)
        267                 )
        268         return (


    ~/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/backends/base.py in get_values(self, varname, burn, thin, combine, chains, squeeze)
        485         varname = get_var_name(varname)
        486         try:
    --> 487             results = [self._straces[chain].get_values(varname, burn, thin) for chain in chains]
        488         except TypeError:  # Single chain passed.
        489             results = [self._straces[chains].get_values(varname, burn, thin)]


    ~/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/backends/base.py in <listcomp>(.0)
        485         varname = get_var_name(varname)
        486         try:
    --> 487             results = [self._straces[chain].get_values(varname, burn, thin) for chain in chains]
        488         except TypeError:  # Single chain passed.
        489             results = [self._straces[chains].get_values(varname, burn, thin)]


    ~/.conda/envs/speclet/lib/python3.9/site-packages/pymc3/backends/ndarray.py in get_values(self, varname, burn, thin)
        330         A NumPy array
        331         """
    --> 332         return self.samples[varname][burn::thin]
        333
        334     def _slice(self, idx):


    KeyError: 'μ_α'

```python
pm.model_to_graphviz(sp_two.model)
```

## MCMC Diagnostics

```python
hyperparams_var_names = ["μ_η", "σ_η", "μ_ɑ", "σ_ɑ", "σ_σ"]
az.plot_trace(az_sptwo_mcmc, var_names=hyperparams_var_names);
```

```python
az.plot_energy(az_sptwo_mcmc);
```

```python
for kind in ["quantile", "local", "evolution"]:
    print(kind.upper())
    az.plot_ess(az_sptwo_mcmc, var_names=hyperparams_var_names, kind=kind)
    plt.show()
    print("-" * 80)
```

## ADVI Diagnostics

```python
advi_loss = sp_two.advi_results.approximation.hist
advi_loss.shape
```

```python
ax = sns.lineplot(x=np.arange(len(advi_loss)), y=advi_loss)
ax.set_yscale("log")
```

## Comparing paramerter estimates

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
for ax, az_obj in zip(axes.flatten(), [az_sptwo_mcmc, az_sptwo_advi]):
    az.plot_forest(az_obj, var_names="η", hdi_prob=HDI_PROB, ax=ax)
plt.show()
```

```python
alpha_posterior = pd.DataFrame()

for fit_method, az_obj in zip(("MCMC", "ADVI"), [az_sptwo_mcmc, az_sptwo_advi]):
    df = az.summary(az_obj, var_names="ɑ", hdi_prob=HDI_PROB, kind="stats").reset_index(
        drop=False
    )
    df = pmanal.extract_matrix_variable_indices(
        df,
        col="index",
        idx1=sp_two.data_manager.get_data()["hugo_symbol"].cat.categories,
        idx2=sp_two.data_manager.get_data()["depmap_id"].cat.categories,
        idx1name="hugo_symbol",
        idx2name="depmap_id",
    ).assign(fit_method=fit_method)
    alpha_posterior = pd.concat([alpha_posterior, df])
```

```python
fit_method_pal = {
    "MCMC": FitMethodColors.pymc3_mcmc,
    "ADVI": FitMethodColors.pymc3_advi,
}

(
    gg.ggplot(alpha_posterior, gg.aes(x="depmap_id", y="mean", color="fit_method"))
    + gg.geom_hline(yintercept=0)
    + gg.geom_boxplot(gg.aes(fill="fit_method"), outlier_size=0.2, alpha=0.5)
    + gg.scale_color_manual(values=fit_method_pal)
    + gg.scale_fill_manual(values=fit_method_pal)
    + gg.theme(figure_size=(8, 3), axis_text_x=gg.element_text(angle=90))
    + gg.labs(x="cell line DepMap ID", y="mean posterior of α", color="fit method")
)
```

```python
SELECT_CL = "ACH-000009"

pos = gg.position_dodge(width=0.8)

(
    gg.ggplot(
        alpha_posterior.query(f"depmap_id == '{SELECT_CL}'"),
        gg.aes(x="hugo_symbol", y="mean", color="fit_method"),
    )
    + gg.geom_hline(yintercept=0, alpha=0.5)
    + gg.geom_linerange(
        gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"),
        position=pos,
        size=0.4,
        alpha=0.75,
        show_legend=False,
    )
    + gg.geom_point(position=pos, size=0.7)
    + gg.scale_color_manual(values=fit_method_pal)
    + gg.theme(
        figure_size=(11, 3),
        axis_text_x=gg.element_text(angle=90, size=6),
        legend_position=(0.192, 0.80),
        legend_direction="horizontal",
        legend_background=eb,
    )
    + gg.labs(x="gene", y="posterior of α (89% CI)", color="fit method")
)
```

```python
sp_two.data_manager.get_data().query(f"depmap_id == '{SELECT_CL}'").query("n_muts > 0")[
    ["hugo_symbol", "n_muts"]
].reset_index(drop=True).drop_duplicates()
```

```python
sp_two.data_manager.get_data().query("n_muts > 0")[
    ["depmap_id", "hugo_symbol"]
].drop_duplicates().groupby("hugo_symbol")[["depmap_id"]].count().reset_index(
    drop=False
).sort_values(
    "depmap_id", ascending=False
).reset_index(
    drop=True
).head(
    8
)
```

```python
SELECT_GENE = "WASF3"
assert SELECT_GENE in alpha_posterior["hugo_symbol"].unique()

gene_mutants = (
    sp_two.data_manager.get_data()
    .query(f"hugo_symbol == '{SELECT_GENE}'")
    .query("n_muts > 0")
    .depmap_id.unique()
)
print(f"Num. mutants: {len(gene_mutants)}")

plot_df = (
    alpha_posterior.query(f"hugo_symbol == '{SELECT_GENE}'")
    .sort_values("mean")
    .reset_index(drop=True)
    .assign(
        depmap_id=lambda d: dphelp.make_cat(d, "depmap_id", ordered=True)["depmap_id"],
        is_mutated=lambda d: d.depmap_id.isin(gene_mutants),
    )
)

(
    gg.ggplot(
        plot_df,
        gg.aes(x="depmap_id", y="mean", color="fit_method"),
    )
    + gg.geom_hline(yintercept=0, alpha=0.5)
    + gg.geom_linerange(
        gg.aes(ymin="hdi_5.5%", ymax="hdi_94.5%"),
        position=pos,
        size=0.4,
        alpha=0.75,
        show_legend=False,
    )
    + gg.geom_point(gg.aes(size="is_mutated"), position=pos)
    + gg.scale_color_manual(values=fit_method_pal)
    + gg.scale_size_discrete(range=(1, 3))
    + gg.theme(
        figure_size=(8, 3),
        axis_text_x=gg.element_text(angle=90, size=8),
    )
    + gg.labs(x="gene", y="posterior of α (89% CI)", color="fit method")
)
```

```python
gene_data = sp_two.data_manager.get_data()[
    ["hugo_symbol", "depmap_id", "gene_cn", "z_log2_cn", "is_mutated", "rna_expr"]
].assign(
    hugo_symbol=lambda d: d.hugo_symbol.astype(str),
    depmap_id=lambda d: d.depmap_id.astype(str),
)
```

```python
alpha_posterior_extra = alpha_posterior.merge(
    gene_data, on=["depmap_id", "hugo_symbol"]
)
alpha_posterior_extra.head()
```

```python
alpha_posterior_extra["gene_cn_rnd"] = [
    np.min([6.0, x]) for x in np.round(alpha_posterior_extra.gene_cn)
]

avg_cn_effect = (
    alpha_posterior_extra.groupby("gene_cn_rnd")[["mean"]]
    .mean()
    .reset_index(drop=False)
)

(
    gg.ggplot(
        alpha_posterior_extra, gg.aes(x="factor(gene_cn_rnd.astype(int))", y="mean")
    )
    + gg.geom_boxplot(outlier_size=0.2, outlier_alpha=0.2)
    + gg.geom_point(data=avg_cn_effect, color=SeabornColor.red)
    + gg.geom_line(
        group="a", data=avg_cn_effect, color=SeabornColor.orange, alpha=1, size=0.7
    )
    + gg.theme(figure_size=(3, 3))
)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

## Comparing prediciton accuracy

```python
for fit_method, az_obj in zip(("MCMC", "ADVI"), [az_sptwo_mcmc, az_sptwo_advi]):
    ax = az.plot_ppc(az_sptwo_mcmc, num_pp_samples=50)
    ax.set_title(fit_method)
plt.show()
```

```python
post_preds = sp_two.data_manager.get_data().copy()

for fit_method, res in zip(
    ("mcmc", "advi"), [sp_two.mcmc_results, sp_two.advi_results]
):
    post_preds = pmanal.summarize_posterior_predictions(
        res.posterior_predictive["lfc"],
        hdi_prob=HDI_PROB,
        merge_with=post_preds,
        calc_error=True,
        observed_y="lfc",
    ).rename(
        columns={
            "pred_mean": f"{fit_method}_pred_mean",
            "pred_hdi_low": f"{fit_method}_pred_hdi_low",
            "pred_hdi_high": f"{fit_method}_pred_hdi_high",
            "error": f"{fit_method}_error",
        }
    )

post_preds.head()
```

```python
plot_df = post_preds[["lfc", "advi_pred_mean", "mcmc_pred_mean"]].melt(
    id_vars=["lfc"],
    value_vars=["advi_pred_mean", "mcmc_pred_mean"],
    var_name="fit_method",
    value_name="pred",
)


def fit_method_labeller(x):
    return x.replace("_pred_mean", "").upper()


(
    gg.ggplot(plot_df, gg.aes(x="pred", y="lfc"))
    + gg.facet_wrap("~ fit_method", labeller=fit_method_labeller, nrow=1)
    + gg.geom_point(size=0.2, alpha=0.2)
    + gg.geom_abline(slope=1, intercept=0, linetype="--", color=SeabornColor.orange)
    + gg.geom_smooth(method="lm", color=SeabornColor.blue)
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0))
    + gg.theme(figure_size=(6, 3))
    + gg.labs(x="posterior prediction (mean)", y="observed LFC")
)
```

```python
(
    gg.ggplot(post_preds, gg.aes(x="mcmc_pred_mean", y="advi_pred_mean"))
    + gg.geom_point(size=0.2, alpha=0.2)
    + gg.geom_abline(slope=1, intercept=0, linetype="--", color=SeabornColor.orange)
    + gg.geom_smooth(method="lm", color=SeabornColor.blue)
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0))
    + gg.labs(x="MCMC pred (mean)", y="ADVI pred (mean)")
)
```

```python

```

```python

```

```python

```

---

```python
notebook_toc = time()
print(f"execution time: {(notebook_toc - notebook_tic) / 60:.2f} minutes")
```

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```
