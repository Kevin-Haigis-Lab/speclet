# Make a smaller data set for testing models


```R
library(nakedpipe)
library(tidyverse)
```

    ── [1mAttaching packages[22m ─────────────────────────────────────── tidyverse 1.3.0 ──
    
    [32m✔[39m [34mggplot2[39m 3.3.2     [32m✔[39m [34mpurrr  [39m 0.3.4
    [32m✔[39m [34mtibble [39m 3.0.3     [32m✔[39m [34mdplyr  [39m 1.0.1
    [32m✔[39m [34mtidyr  [39m 1.1.1     [32m✔[39m [34mstringr[39m 1.4.0
    [32m✔[39m [34mreadr  [39m 1.3.1     [32m✔[39m [34mforcats[39m 0.5.0
    
    ── [1mConflicts[22m ────────────────────────────────────────── tidyverse_conflicts() ──
    [31m✖[39m [34mdplyr[39m::[32mfilter()[39m masks [34mstats[39m::filter()
    [31m✖[39m [34mdplyr[39m::[32mlag()[39m    masks [34mstats[39m::lag()
    



```R
data_dir <- file.path("..", "modeling_data")
modeling_data_path <- file.path(data_dir, "depmap_modeling_dataframe.csv")
out_path <- file.path(data_dir, "depmap_modeling_dataframe_subsample.csv")
```


```R
modeling_data <- read_csv(modeling_data_path, guess_max = 1e5)
head(modeling_data)
```

    Parsed with column specification:
    cols(
      .default = col_character(),
      lfc = [32mcol_double()[39m,
      pdna_batch = [32mcol_double()[39m,
      passes_qc = [33mcol_logical()[39m,
      n_alignments = [32mcol_double()[39m,
      chrom_pos = [32mcol_double()[39m,
      segment_mean = [32mcol_double()[39m,
      copy_number = [32mcol_double()[39m,
      n_muts = [32mcol_double()[39m,
      any_deleterious = [33mcol_logical()[39m,
      mutated_at_guide_location = [33mcol_logical()[39m,
      rna_expr = [32mcol_double()[39m
    )
    
    See spec(...) for full column specifications.
    



<table>
<caption>A tibble: 6 × 25</caption>
<thead>
	<tr><th scope=col>sgrna</th><th scope=col>replicate_id</th><th scope=col>lfc</th><th scope=col>pdna_batch</th><th scope=col>passes_qc</th><th scope=col>depmap_id</th><th scope=col>primary_or_metastasis</th><th scope=col>lineage</th><th scope=col>lineage_subtype</th><th scope=col>kras_mutation</th><th scope=col>⋯</th><th scope=col>segment_mean</th><th scope=col>copy_number</th><th scope=col>n_muts</th><th scope=col>any_deleterious</th><th scope=col>variant_classification</th><th scope=col>is_deleterious</th><th scope=col>is_tcga_hotspot</th><th scope=col>is_cosmic_hotspot</th><th scope=col>mutated_at_guide_location</th><th scope=col>rna_expr</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;lgl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>⋯</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;lgl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;lgl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>AAAAAAATCCAGCAATGCAG</td><td>143b-311cas9_repa_p6_batch3</td><td> 0.2896938</td><td>3</td><td>TRUE</td><td>ACH-001001</td><td>Primary</td><td>bone</td><td>osteosarcoma</td><td>G12S</td><td>⋯</td><td>1.1434280</td><td>2.209053</td><td>0</td><td>FALSE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>FALSE</td><td>4.1009776</td></tr>
	<tr><td>AAAAAACCCGTAGATAGCCT</td><td>143b-311cas9_repa_p6_batch3</td><td> 0.1701723</td><td>3</td><td>TRUE</td><td>ACH-001001</td><td>Primary</td><td>bone</td><td>osteosarcoma</td><td>G12S</td><td>⋯</td><td>0.8850587</td><td>1.846840</td><td>0</td><td>FALSE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>FALSE</td><td>7.4709435</td></tr>
	<tr><td>AAAAAAGAAGAAAAAACCAG</td><td>143b-311cas9_repa_p6_batch3</td><td>-0.6959467</td><td>3</td><td>TRUE</td><td>ACH-001001</td><td>Primary</td><td>bone</td><td>osteosarcoma</td><td>G12S</td><td>⋯</td><td>0.8946238</td><td>1.859125</td><td>0</td><td>FALSE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>FALSE</td><td>4.6270231</td></tr>
	<tr><td>AAAAAAGCTCAAGAAGGAGG</td><td>143b-311cas9_repa_p6_batch3</td><td>-0.3249354</td><td>3</td><td>TRUE</td><td>ACH-001001</td><td>Primary</td><td>bone</td><td>osteosarcoma</td><td>G12S</td><td>⋯</td><td>1.0461620</td><td>2.065029</td><td>0</td><td>FALSE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>FALSE</td><td>4.7750505</td></tr>
	<tr><td>AAAAAAGGCTGTAAAAGCGT</td><td>143b-311cas9_repa_p6_batch3</td><td> 0.1428739</td><td>3</td><td>TRUE</td><td>ACH-001001</td><td>Primary</td><td>bone</td><td>osteosarcoma</td><td>G12S</td><td>⋯</td><td>0.8744568</td><td>1.833318</td><td>0</td><td>FALSE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>FALSE</td><td>0.2265085</td></tr>
	<tr><td>AAAAAAGGGCTCCAAAAAGG</td><td>143b-311cas9_repa_p6_batch3</td><td>-0.2998787</td><td>3</td><td>TRUE</td><td>ACH-001001</td><td>Primary</td><td>bone</td><td>osteosarcoma</td><td>G12S</td><td>⋯</td><td>1.0855960</td><td>2.122252</td><td>0</td><td>FALSE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>FALSE</td><td>0.3785116</td></tr>
</tbody>
</table>




```R
glimpse(modeling_data)
```

    Rows: 120,235,860
    Columns: 25
    $ sgrna                     [3m[90m<chr>[39m[23m "AAAAAAATCCAGCAATGCAG", "AAAAAACCCGTAGATAGC…
    $ replicate_id              [3m[90m<chr>[39m[23m "143b-311cas9_repa_p6_batch3", "143b-311cas…
    $ lfc                       [3m[90m<dbl>[39m[23m 0.28969381, 0.17017231, -0.69594665, -0.324…
    $ pdna_batch                [3m[90m<dbl>[39m[23m 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3…
    $ passes_qc                 [3m[90m<lgl>[39m[23m TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, T…
    $ depmap_id                 [3m[90m<chr>[39m[23m "ACH-001001", "ACH-001001", "ACH-001001", "…
    $ primary_or_metastasis     [3m[90m<chr>[39m[23m "Primary", "Primary", "Primary", "Primary",…
    $ lineage                   [3m[90m<chr>[39m[23m "bone", "bone", "bone", "bone", "bone", "bo…
    $ lineage_subtype           [3m[90m<chr>[39m[23m "osteosarcoma", "osteosarcoma", "osteosarco…
    $ kras_mutation             [3m[90m<chr>[39m[23m "G12S", "G12S", "G12S", "G12S", "G12S", "G1…
    $ genome_alignment          [3m[90m<chr>[39m[23m "chr10_110964620_+", "chr12_95003615_+", "c…
    $ n_alignments              [3m[90m<dbl>[39m[23m 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
    $ hugo_symbol               [3m[90m<chr>[39m[23m "SHOC2", "NDUFA12", "SDAD1", "FAM98A", "ZNF…
    $ chromosome                [3m[90m<chr>[39m[23m "10", "12", "4", "2", "19", "6", "14", "1",…
    $ chrom_pos                 [3m[90m<dbl>[39m[23m 110964620, 95003615, 75970356, 33588446, 19…
    $ segment_mean              [3m[90m<dbl>[39m[23m 1.1434280, 0.8850587, 0.8946238, 1.0461620,…
    $ copy_number               [3m[90m<dbl>[39m[23m 2.209053, 1.846840, 1.859125, 2.065029, 1.8…
    $ n_muts                    [3m[90m<dbl>[39m[23m 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0…
    $ any_deleterious           [3m[90m<lgl>[39m[23m FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, F…
    $ variant_classification    [3m[90m<chr>[39m[23m NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
    $ is_deleterious            [3m[90m<chr>[39m[23m NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
    $ is_tcga_hotspot           [3m[90m<chr>[39m[23m NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
    $ is_cosmic_hotspot         [3m[90m<chr>[39m[23m NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
    $ mutated_at_guide_location [3m[90m<lgl>[39m[23m FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, F…
    $ rna_expr                  [3m[90m<dbl>[39m[23m 4.1009776, 7.4709435, 4.6270231, 4.7750505,…



```R
# Number of unique guides.
n_distinct(modeling_data$sgrna)
```


72133



```R
# Number of unique cell lines.
n_distinct(modeling_data$depmap_id)
```


767



```R
# Number of genes.
n_distinct(modeling_data$hugo_symbol)
```


18524



```R
# Some specific genes to subset.
specific_genes <- c("KRAS", "BRAF", "PIK3CA", "PTK2", "MDM2", "TP53")

set.seed(0)
num_random_genes <- 20
subsample_genes <- c(
    sample(unique(modeling_data$hugo_symbol), num_random_genes),
    specific_genes
)

# Check that all genes are actually in the modeling data.
all(subsample_genes %in% modeling_data$hugo_symbol)
```


TRUE



```R
subsample_genes
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'PHACTR3'</li><li>'ZSWIM8'</li><li>'DPH7'</li><li>'LGALS7B'</li><li>'DISP1'</li><li>'KIF3C'</li><li>'RNF125'</li><li>'LGALS4'</li><li>'SLC7A14'</li><li>'UQCRC1'</li><li>'SCMH1'</li><li>'SMAD7'</li><li>'GHSR'</li><li>'IQCK'</li><li>'NDUFAF3'</li><li>'FAM43B'</li><li>'PDE5A'</li><li>'HIST1H2BO'</li><li>'ADAMTS13'</li><li>'CXCL2'</li><li>'KRAS'</li><li>'BRAF'</li><li>'PIK3CA'</li><li>'PTK2'</li><li>'MDM2'</li><li>'TP53'</li></ol>




```R
subsample_modeling_data <- modeling_data %>% filter(hugo_symbol %in% subsample_genes)
pryr::object_size(subsample_modeling_data)
```


    31.5 MB



```R
nrow(subsample_modeling_data)
```


166345



```R
write_csv(subsample_modeling_data, out_path)
```


```R

```
