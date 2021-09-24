# Tidy and clean Project SCORE raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


#### ---- Data tidying functions ---- ####


read_replicate_map <- function(f) {
  readr::read_csv(f) %>%
    janitor::clean_names() %>%
    rename(depmap_id = dep_map_id)
}

read_guide_map <- function(f) {
  readr::read_csv(f) %>%
    extract_hugo_gene_name(gene) %>%
    rename(hugo_symbol = gene)
}

tidy_log_fold_change <- function(lfc_file,
                                 guide_map_file,
                                 replicate_map_file,
                                 out_file) {
  replicate_map <- read_replicate_map(replicate_map_file)
  guide_map <- read_guide_map(guide_map_file)

  readr::read_csv(lfc_file) %>%
    flatten_wide_df_by_gene(values_to = "lfc") %>%
    rename(sgrna = depmap_id, replicate_id = hugo_symbol) %>%
    inner_join(replicate_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    readr::write_csv(out_file)
}

tidy_read_counts <- function(counts_file,
                             guide_map_file,
                             replicate_map_file,
                             out_file) {
  # TODO
  stop("Not yet implemented.")
}

#### ---- Function calls ---- ####

print("---- Tidying SCORE log fold change. ----")
tidy_log_fold_change(
  lfc_file = snakemake@input[["log_fold_change"]],
  guide_map_file = snakemake@input[["guide_map"]],
  replicate_map_file = snakemake@input[["replicate_map"]],
  out_file = snakemake@output[["log_fold_change"]]
)

print("---- Tidying SCORE read counts. ----")
tidy_read_counts(
  counts_file = snakemake@input[["score_raw_readcounts"]],
  guide_map_file = snakemake@input[["guide_map"]],
  replicate_map_file = snakemake@input[["replicate_map"]],
  out_file = snakemake@output[["score_read_counts"]]
)
