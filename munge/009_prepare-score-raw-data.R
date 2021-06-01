#!/usr/bin/env Rscript

# Tidy and clean Project SCORE raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


#### ---- Data tidying functions ---- ####

tidy_gene_copynumber <- function(in_file, out_file) {
  read_csv(in_file, n_max = 1e4, col_types = c("Chromosome" = "c")) %>%
    janitor::clean_names() %>%
    rename(depmap_id = dep_map_id, start_pos = start, end_pos = end) %>%
    write_csv(out_file)
}


tidy_gene_effect <- function(scaled_ge_file, unscaled_ge_file, out_file) {
  scaled_ge <- read_csv(scaled_ge_file, n_max = 10) %>%
    flatten_wide_df_by_gene(values_to = "gene_effect")
  unscaled_ge <- read_csv(unscaled_ge_file, n_max = 10) %>%
    flatten_wide_df_by_gene(values_to = "gene_effect_unscaled")

  combined_ge <- inner_join(
    scaled_ge,
    unscaled_ge,
    by = c("depmap_id", "hugo_symbol")
  )

  if (nrow(scaled_ge) != nrow(combined_ge) || nrow(unscaled_ge) != nrow(combined_ge)) {
    stop("Lost data when merging scaled and unscaled gene effect values.")
  }

  write_csv(combined_ge, out_file)
}

tidy_log_fold_change <- function(lfc_file,
                                 guide_map_file,
                                 replicate_map_file,
                                 out_file) {
  replicate_map <- read_csv(replicate_map_file, n_max = 10) %>%
    janitor::clean_names() %>%
    rename(depmap_id = dep_map_id)

  guide_map <- read_csv(guide_map_file, n_max = 10) %>%
    extract_hugo_gene_name(gene) %>%
    rename(hugo_symbol = gene)

  read_csv(lfc_file, n_max = 10) %>%
    flatten_wide_df_by_gene(values_to = "lfc") %>%
    rename(sgrna = depmap_id, replicate_id = hugo_symbol) %>%
    inner_join(replicate_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    write_csv(out_file)
}



#### ---- Function calls ---- ####

print("Tidying SCORE copy number data.")
tidy_gene_copynumber(
  in_file = snakemake@input[["copy_number"]],
  out_file = snakemake@output[["copy_number"]]
)

print("Tidying SCORE gene effect.")
tidy_gene_effect(
  scaled_ge_file = snakemake@input[["gene_effect"]],
  unscaled_ge_file = snakemake@input[["gene_effect_unscaled"]],
  out_file = snakemake@output[["gene_effect"]]
)

print("Tidying SCORE log fold change.")
tidy_log_fold_change(
  lfc_file = snakemake@input[["log_fold_change"]],
  guide_map_file = snakemake@input[["guide_map"]],
  replicate_map_file = snakemake@input[["replicate_map"]],
  out_file = snakemake@output[["log_fold_change"]]
)
