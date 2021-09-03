#!/usr/bin/env Rscript

# Tidy and clean DepMap raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


#### ---- Data tidying functions ---- ####


make_known_essentials_and_nonessentials <- function(essentials_file,
                                                    nonessentials_file,
                                                    out_file) {
  essentials <- read_csv(essentials_file) %>%
    extract_hugo_gene_name() %>%
    add_column(is_essential = TRUE)
  nonessentials <- read_csv(nonessentials_file) %>%
    extract_hugo_gene_name() %>%
    add_column(is_essential = FALSE)

  bind_rows(essentials, nonessentials) %>%
    write_csv(out_file)
}


remove_columns <- function(df, cols_to_drop) {
  return(df[, !(colnames(df) %in% cols_to_drop)])
}


read_replicate_map <- function(f) {
  read_csv(f) %>%
    janitor::clean_names() %>%
    rename(depmap_id = dep_map_id)
}

get_dropped_replicates <- function(rep_map) {
  rep_map %>%
    filter(!passes_qc) %>%
    pull(replicate_id) %>%
    unlist() %>%
    unique()
}

read_guide_map <- function(f) {
  read_csv(f) %>%
    janitor::clean_names() %>%
    extract_hugo_gene_name() %>%
    rename(hugo_symbol = gene)
}

check_all_pass_qc <- function(df) {
  if (!all(df$passes_qc)) {
    stop("Not all batches passes QC.")
  }
}

tidy_log_fold_change <- function(lfc_file,
                                 replicate_map,
                                 dropped_guides,
                                 guide_map,
                                 out_file) {
  guide_map <- read_guide_map(guide_map)

  rep_map <- read_replicate_map(replicate_map)
  dropped_reps <- get_dropped_replicates(rep_map)
  print(paste("Number of dropped batches:", length(dropped_reps)))

  dropped_guides <- read_csv(dropped_guides)$X1
  print(paste("Number of dropped guides:", length(dropped_guides)))

  lfc_df <- read_csv(lfc_file) %>%
    rename(sgrna = `Construct Barcode`) %>%
    filter(!sgrna %in% !!dropped_guides) %>%
    remove_columns(dropped_reps) %>%
    pivot_longer(-sgrna, names_to = "replicate_id", values_to = "lfc") %>%
    inner_join(rep_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    rename(n_sgrna_alignments = n_alignments)

  check_all_pass_qc(lfc_df)
  write_csv(lfc_df, out_file)
}


tidy_read_counts <- function(rc_file, replicate_map, dropped_guides, guide_map, out_file) {
  guide_map <- read_guide_map(guide_map)

  rep_map <- read_replicate_map(replicate_map)
  dropped_reps <- get_dropped_replicates(rep_map)
  print(paste("Number of dropped batches:", length(dropped_reps)))

  dropped_guides <- read_csv(dropped_guides)$X1
  print(paste("Number of dropped guides:", length(dropped_guides)))

  read_counts_df <- read_csv(rc_file) %>%
    rename(sgrna = `Construct Barcode`) %>%
    filter(!sgrna %in% !!dropped_guides) %>%
    remove_columns(dropped_reps) %>%
    pivot_longer(-sgrna, names_to = "replicate_id", values_to = "read_counts") %>%
    inner_join(rep_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    rename(n_sgrna_alignments = n_alignments)

  check_all_pass_qc(read_counts_df)
  write_csv(read_counts_df, out_file)
}


tidy_achilles_gene_effect <- function(gene_effect_scaled, gene_effect_unscaled, out_file) {
  ge_scaled <- read_csv(gene_effect_scaled) %>%
    flatten_wide_df_by_gene(values_to = "gene_effect")
  ge_unscaled <- read_csv(gene_effect_unscaled) %>%
    rename(X1 = DepMap_ID) %>%
    flatten_wide_df_by_gene(values_to = "gene_effect_unscaled")
  ge_combined <- inner_join(
    ge_scaled,
    ge_unscaled,
    by = c("depmap_id", "hugo_symbol")
  )

  if (nrow(ge_scaled) != nrow(ge_combined) || nrow(ge_unscaled) != nrow(ge_combined)) {
    stop("Lost data when merging scaled and unscaled gene effect values.")
  }

  write_csv(ge_combined, out_file)
}


tidy_chronos_gene_effect <- function(chronos_gene_effect, out_file) {
  read_csv(chronos_gene_effect) %>%
    rename(X1 = DepMap_ID) %>%
    flatten_wide_df_by_gene(values_to = "chronos_gene_effect") %>%
    write_csv(out_file)
}



#### ---- Function calls ---- ####


print("---- Tidying known essential and non-essential genes. ----")
make_known_essentials_and_nonessentials(
  essentials_file = snakemake@input[["common_essentials"]],
  nonessentials_file = snakemake@input[["nonessentials"]],
  out_file = snakemake@output[["known_essentials"]]
)

print("---- Tidying log fold change data. ----")
tidy_log_fold_change(
  lfc_file = snakemake@input[["achilles_logfold_change"]],
  replicate_map = snakemake@input[["achilles_replicate_map"]],
  dropped_guides = snakemake@input[["achilles_dropped_guides"]],
  guide_map = snakemake@input[["achilles_guide_map"]],
  out_file = snakemake@output[["achilles_log_fold_change"]]
)

print("---- Tidying read count data. ----")
tidy_read_counts(
  rc_file = snakemake@input[["achilles_raw_readcounts"]],
  replicate_map = snakemake@input[["achilles_replicate_map"]],
  dropped_guides = snakemake@input[["achilles_dropped_guides"]],
  guide_map = snakemake@input[["achilles_guide_map"]],
  out_file = snakemake@output[["achilles_read_counts"]]
)

print("---- Tidying Achilles gene effect. ----")
tidy_achilles_gene_effect(
  gene_effect_scaled = snakemake@input[["achilles_gene_effect"]],
  gene_effect_unscaled = snakemake@input[["achilles_gene_effect_unscaled"]],
  out_file = snakemake@output[["achilles_gene_effect"]]
)

print("---- Tidying Chronos gene effect. ----")
tidy_chronos_gene_effect(
  chronos_gene_effect = snakemake@input[["all_gene_effect_chronos"]],
  out_file = snakemake@output[["chronos_gene_effect"]]
)
