# Prepare data for modeling


```python
raise "TODO: make faster!"
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-1-03b94e7188ef> in <module>
    ----> 1 raise "TODO: make faster!"
    

    TypeError: exceptions must derive from BaseException



```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import janitor
from pathlib import Path
import re
import plotnine as gg
```


```python
gg.theme_set(gg.theme_minimal())
```


```python
data_dir = Path("../data")
save_dir = Path("../modeling_data")
```

### Setup dask


```python
import dask
import dask.dataframe as dd
```


```python
from dask.distributed import Client, progress

client = Client(n_workers=2, threads_per_worker=2, memory_limit="20GB")
client
```




<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://127.0.0.1:41948</li>
  <li><b>Dashboard: </b><a href='http://127.0.0.1:8787/status' target='_blank'>http://127.0.0.1:8787/status</a></li>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Cluster</h3>
<ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
  <li><b>Workers: </b>2</li>
  <li><b>Cores: </b>4</li>
  <li><b>Memory: </b>40.00 GB</li>
</ul>
</td>
</tr>
</table>



## Cell line information


```python
def show_counts(df, col):
    if type(col) != list:
        col = [col]

    return (
        df[col + ["depmap_id"]]
        .drop_duplicates()
        .groupby(col)
        .count()
        .sort_values("depmap_id", ascending=False)
    )
```


```python
sample_info = pd.read_csv(save_dir / "sample_info.csv")
show_counts(sample_info, "lineage")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>depmap_id</th>
    </tr>
    <tr>
      <th>lineage</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lung</th>
      <td>273</td>
    </tr>
    <tr>
      <th>blood</th>
      <td>132</td>
    </tr>
    <tr>
      <th>skin</th>
      <td>113</td>
    </tr>
    <tr>
      <th>lymphocyte</th>
      <td>109</td>
    </tr>
    <tr>
      <th>central_nervous_system</th>
      <td>107</td>
    </tr>
    <tr>
      <th>colorectal</th>
      <td>83</td>
    </tr>
    <tr>
      <th>breast</th>
      <td>82</td>
    </tr>
    <tr>
      <th>upper_aerodigestive</th>
      <td>76</td>
    </tr>
    <tr>
      <th>bone</th>
      <td>75</td>
    </tr>
    <tr>
      <th>ovary</th>
      <td>74</td>
    </tr>
    <tr>
      <th>soft_tissue</th>
      <td>71</td>
    </tr>
    <tr>
      <th>pancreas</th>
      <td>59</td>
    </tr>
    <tr>
      <th>kidney</th>
      <td>57</td>
    </tr>
    <tr>
      <th>gastric</th>
      <td>49</td>
    </tr>
    <tr>
      <th>peripheral_nervous_system</th>
      <td>47</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>45</td>
    </tr>
    <tr>
      <th>bile_duct</th>
      <td>43</td>
    </tr>
    <tr>
      <th>fibroblast</th>
      <td>43</td>
    </tr>
    <tr>
      <th>uterus</th>
      <td>39</td>
    </tr>
    <tr>
      <th>urinary_tract</th>
      <td>39</td>
    </tr>
    <tr>
      <th>esophagus</th>
      <td>38</td>
    </tr>
    <tr>
      <th>plasma_cell</th>
      <td>34</td>
    </tr>
    <tr>
      <th>liver</th>
      <td>27</td>
    </tr>
    <tr>
      <th>cervix</th>
      <td>22</td>
    </tr>
    <tr>
      <th>thyroid</th>
      <td>21</td>
    </tr>
    <tr>
      <th>prostate</th>
      <td>13</td>
    </tr>
    <tr>
      <th>eye</th>
      <td>9</td>
    </tr>
    <tr>
      <th>engineered</th>
      <td>5</td>
    </tr>
    <tr>
      <th>engineered_bone</th>
      <td>4</td>
    </tr>
    <tr>
      <th>embryo</th>
      <td>4</td>
    </tr>
    <tr>
      <th>engineered_kidney</th>
      <td>3</td>
    </tr>
    <tr>
      <th>engineered_breast</th>
      <td>2</td>
    </tr>
    <tr>
      <th>engineered_prostate</th>
      <td>1</td>
    </tr>
    <tr>
      <th>engineered_ovary</th>
      <td>1</td>
    </tr>
    <tr>
      <th>engineered_lung</th>
      <td>1</td>
    </tr>
    <tr>
      <th>engineered_central_nervous_system</th>
      <td>1</td>
    </tr>
    <tr>
      <th>engineered_blood</th>
      <td>1</td>
    </tr>
    <tr>
      <th>adrenal_cortex</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
noncancerous_lineages = ["unknown", "embryo"]
engineered_lineages = sample_info[
    sample_info.lineage.str.contains("engineer")
].lineage.to_list()

ignore_lineages = engineered_lineages + noncancerous_lineages
sample_info = sample_info[~sample_info.lineage.isin(ignore_lineages)]

sample_info_columns = [
    "depmap_id",
    "primary_or_metastasis",
    "lineage",
    "lineage_subtype",
]
sample_info = sample_info[sample_info_columns].drop_duplicates()
```

## *KRAS* mutations


```python
# Remove all cell lines with no mutation data.
all_samples_with_mutation_data = pd.read_csv(
    save_dir / "ccle_mutations.csv", low_memory=False
).depmap_id.unique()

sample_info = sample_info.pipe(
    lambda x: x[x.depmap_id.isin(all_samples_with_mutation_data)]
)
```


```python
kras_mutations = pd.read_csv(save_dir / "kras_mutations.csv")
kras_mutations = kras_mutations[["depmap_id", "kras_mutation"]]

sample_info = sample_info.merge(kras_mutations, on="depmap_id", how="left").assign(
    kras_mutation=lambda x: x.kras_mutation.fillna("WT")
)
```


```python
sample_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>depmap_id</th>
      <th>primary_or_metastasis</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>kras_mutation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACH-000001</td>
      <td>Metastasis</td>
      <td>ovary</td>
      <td>ovary_adenocarcinoma</td>
      <td>WT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACH-000002</td>
      <td>Primary</td>
      <td>blood</td>
      <td>AML</td>
      <td>WT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACH-000003</td>
      <td>NaN</td>
      <td>colorectal</td>
      <td>colorectal_adenocarcinoma</td>
      <td>WT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACH-000004</td>
      <td>NaN</td>
      <td>blood</td>
      <td>AML</td>
      <td>WT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACH-000005</td>
      <td>NaN</td>
      <td>blood</td>
      <td>AML</td>
      <td>WT</td>
    </tr>
  </tbody>
</table>
</div>



## Screen data


```python
achilles_lfc = dd.read_csv(save_dir / "achilles_logfold_change.csv").compute()
```


```python
achilles_guide_map = dd.read_csv(save_dir / "achilles_guide_map.csv").compute()
```


```python
modeling_data = pd.merge(
    left=achilles_lfc, right=sample_info, how="inner", on=["depmap_id"]
)

modeling_data = pd.merge(
    left=modeling_data, right=achilles_guide_map, how="inner", on=["sgrna"]
)

modeling_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sgrna</th>
      <th>replicate_id</th>
      <th>lfc</th>
      <th>depmap_id</th>
      <th>pdna_batch</th>
      <th>passes_qc</th>
      <th>primary_or_metastasis</th>
      <th>lineage</th>
      <th>lineage_subtype</th>
      <th>kras_mutation</th>
      <th>genome_alignment</th>
      <th>n_alignments</th>
      <th>hugo_symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAAAAAATCCAGCAATGCAG</td>
      <td>143b-311cas9_repa_p6_batch3</td>
      <td>0.289694</td>
      <td>ACH-001001</td>
      <td>3</td>
      <td>True</td>
      <td>Primary</td>
      <td>bone</td>
      <td>osteosarcoma</td>
      <td>G12S</td>
      <td>chr10_110964620_+</td>
      <td>1</td>
      <td>SHOC2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AAAAAAATCCAGCAATGCAG</td>
      <td>2313287-311cas9_repa_p5_batch3</td>
      <td>0.171917</td>
      <td>ACH-000948</td>
      <td>3</td>
      <td>True</td>
      <td>Primary</td>
      <td>gastric</td>
      <td>gastric_adenocarcinoma</td>
      <td>WT</td>
      <td>chr10_110964620_+</td>
      <td>1</td>
      <td>SHOC2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AAAAAAATCCAGCAATGCAG</td>
      <td>2313287-311cas9_repb_p5_batch3</td>
      <td>-0.522717</td>
      <td>ACH-000948</td>
      <td>3</td>
      <td>True</td>
      <td>Primary</td>
      <td>gastric</td>
      <td>gastric_adenocarcinoma</td>
      <td>WT</td>
      <td>chr10_110964620_+</td>
      <td>1</td>
      <td>SHOC2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AAAAAAATCCAGCAATGCAG</td>
      <td>253j-311cas9_repa_p5_batch3</td>
      <td>-0.211690</td>
      <td>ACH-000011</td>
      <td>3</td>
      <td>True</td>
      <td>Metastasis</td>
      <td>urinary_tract</td>
      <td>bladder_carcinoma</td>
      <td>WT</td>
      <td>chr10_110964620_+</td>
      <td>1</td>
      <td>SHOC2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAAAAAATCCAGCAATGCAG</td>
      <td>42-mg-ba-311cas9_repa_p6_batch3</td>
      <td>-1.067942</td>
      <td>ACH-000323</td>
      <td>3</td>
      <td>True</td>
      <td>Primary</td>
      <td>central_nervous_system</td>
      <td>glioma</td>
      <td>WT</td>
      <td>chr10_110964620_+</td>
      <td>1</td>
      <td>SHOC2</td>
    </tr>
  </tbody>
</table>
</div>



## Copy number at each guide target


```python
def get_segment_mean(cn_data, depmap_id, chromosome, pos):
    d = cn_data[(cn_data.depmap_id == depmap_id) & (cn_data.chromosome == chromosome)]
    d = d[(d.start <= pos) & (pos <= d.end)]
    if len(d) == 0:
        return None
    elif len(d) > 1:
        raise Exception(f"Data contains more than 1 row of data: {len(d)}")
    else:
        return d.segment_mean.to_list()[0]


def segmentmean_to_copynumber(seg_mean):
    if seg_mean == None:
        return None
    return 2.0 ** seg_mean


def parse_genome_location(gloc):
    d = gloc.split("_")
    d[0] = d[0].replace("chr", "")
    return (d[0], int(d[1]))


cn_segments = dd.read_csv(save_dir / "ccle_semgent_cn.csv").compute()

for i in range(0, len(modeling_data)):
    genome_loc = modeling_data.loc[i, "genome_alignment"]
    depmap_id = modeling_data.loc[i, "depmap_id"]
    g_chr, g_pos = parse_genome_location(genome_loc)

    seg_mean = get_segment_mean(
        cn_data=cn_segments, depmap_id=depmap_id, chromosome=g_chr, pos=g_pos
    )

    modeling_data.at[i, "chromosome"] = g_chr
    modeling_data.at[i, "chr_position"] = g_pos
    modeling_data.at[i, "copy_number"] = segmentmean_to_copynumber(seg_mean)

modeling_data.head(10)
```


```python
eg_cellline = "ACH-001001"

chromosome_order = [str(a) for a in range(1, 23)] + ["X"]

d = modeling_data[modeling_data.depmap_id == eg_cellline]
d["chromosome"] = pd.Categorical(d.chromosome, categories=chromosome_order)

gg.options.set_option("figure_size", (12, 8))
gg.options.set_option("dpi", 400)

(
    gg.ggplot(d, gg.aes("chr_position", "copy_number", color="chromosome"))
    + gg.facet_wrap("chromosome", ncol=4, scales="free")
    + gg.geom_line(alpha=0.5, size=0.5)
    + gg.geom_point(alpha=0.5, size=0.5)
    + gg.theme_minimal()
    + gg.theme(
        axis_text_x=gg.element_blank(),
        legend_position="none",
        subplots_adjust={"hspace": 0.25, "wspace": 0.25},
        axis_text_y=gg.element_text(hjust=2),
    )
    + gg.labs(
        x="chromosome position",
        y="copy number",
        title=f"Example of copy number variation for a single cell line: {eg_cellline}",
    )
)
```

## Mutation data of each gene


```python
full_mutations_df = pd.read_csv(
    save_dir / "ccle_mutations.csv", dtype={"chromosome": str}, low_memory=False
)

mutations_data_columns = [
    "depmap_id",
    "hugo_symbol",
    "variant_classification",
    "variant_type",
    "isdeleterious",
    "istcgahotspot",
    "iscosmichotspot",
]


def mod_variant_classifications(d):
    d.variant_classification = d.variant_classification.fillna("unknown")
    d.variant_classification = d.variant_classification.astype("str")
    d.variant_classification = [a.lower() for a in d.variant_classification]
    return d


mutations_df = (
    full_mutations_df[mutations_data_columns]
    .rename(
        columns={
            "isdeleterious": "is_deleterious",
            "istcgahotspot": "is_tcga_hotspot",
            "iscosmichotspot": "is_cosmic_hotspot",
        }
    )
    .pipe(mod_variant_classifications)
)


def any_mutations_true(mut_data, col_name):
    return [any([a[col_name] for a in m]) for m in mut_data]


mutations_df = (
    pd.DataFrame(
        mutations_df.set_index(["depmap_id", "hugo_symbol"])
        .apply(lambda d: d.to_dict(), axis=1)
        .groupby(["depmap_id", "hugo_symbol"])
        .agg(mutation_data=lambda x: list(x))
    )
    .assign(n_muts=lambda d: [len(a) for a in d.mutation_data])
    .reset_index()
    .assign(
        any_deleterious=lambda df: any_mutations_true(
            df.mutation_data, "is_deleterious"
        ),
        any_tcga_hotspot=lambda df: any_mutations_true(
            df.mutation_data, "is_tcga_hotspot"
        ),
        any_cosmic_hotspot=lambda df: any_mutations_true(
            df.mutation_data, "is_cosmic_hotspot"
        ),
    )
)

mutations_df.head()
```

Check that there is one row per `depmap_id` x `hugo_symbol` pair.
An error will be raised if this is not true.


```python
x = (
    mutations_df.groupby(["depmap_id", "hugo_symbol"])
    .count()
    .query("mutation_data > 1")
)
if x.shape[0] > 0:
    raise "More than one row per cell line x gene pair"
```


```python
modeling_data = pd.merge(
    left=modeling_data, right=mutations_df, how="left", on=["depmap_id", "hugo_symbol"]
).fillna(
    value={
        "n_muts": 0,
        "any_deleterious": False,
        "any_tcga_hotspot": False,
        "any_cosmic_hotspot": False,
    }
)
```


```python
modeling_data.head()
```

## A column to indicate if there is a mutation at the guide target location


```python
modeling_data["mutated_at_target"] = False

for i in range(modeling_data.shape[0]):
    g_chr = modeling_data.chromosome[i]
    g_pos = modeling_data.chr_position[i]
    g_depmapid = modeling_data.depmap_id[i]

    mut_d = (
        full_mutations_df.query(f"depmap_id == '{g_depmapid}'")
        .query(f"chromosome == '{g_chr}'")
        .query(f"start_position <= {g_pos} <= end_position")
    )

    if mut_d.shape[0] >= 1:
        modeling_data.loc[i, "mutated_at_target"] = True
```


```python
modeling_data.groupby("mutated_at_target").count()
```

## RNA expression of the target gene


```python
rna_df = (
    dd.read_csv(save_dir / "ccle_expression.csv")
    .rename(columns={"dep_map_id": "depmap_id"})
    .compute()
)
rna_df.head()
```


```python
modeling_data = pd.merge(
    left=modeling_data, right=rna_df, how="left", on=["depmap_id", "hugo_symbol"]
)
```

---

## Final data frame


```python
modeling_data.head()
```


```python
modeling_data.to_csv(save_dir / "depmap_modeling_dataframe.csv")
```


```python

```
