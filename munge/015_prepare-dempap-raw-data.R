# Tidy and clean DepMap raw data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


# ---- Data tidying functions ----


make_known_essentials_and_nonessentials <- function(essentials_file,
                                                    nonessentials_file,
                                                    out_file) {
  essentials <- readr::read_csv(essentials_file) %>%
    extract_hugo_gene_name() %>%
    add_column(is_essential = TRUE)
  nonessentials <- readr::read_csv(nonessentials_file) %>%
    extract_hugo_gene_name() %>%
    add_column(is_essential = FALSE)

  bind_rows(essentials, nonessentials) %>%
    readr::write_csv(out_file)
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
  guide_map <- read_achilles_guide_map(guide_map)

  rep_map <- read_achilles_replicate_map(replicate_map)
  dropped_reps <- get_dropped_replicates(rep_map)
  print(paste("Number of dropped batches:", length(dropped_reps)))

  dropped_guides <- readr::read_csv(dropped_guides)$X1
  print(paste("Number of dropped guides:", length(dropped_guides)))

  lfc_df <- readr::read_csv(lfc_file) %>%
    rename(sgrna = `Construct Barcode`) %>%
    filter(!sgrna %in% !!dropped_guides) %>%
    remove_columns(dropped_reps) %>%
    pivot_longer(-sgrna, names_to = "replicate_id", values_to = "lfc") %>%
    inner_join(rep_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    rename(n_sgrna_alignments = n_alignments)

  check_all_pass_qc(lfc_df)
  readr::write_csv(lfc_df, out_file)
}


tidy_read_counts <- function(rc_file,
                             replicate_map,
                             dropped_guides,
                             guide_map,
                             out_file) {
  guide_map <- read_achilles_guide_map(guide_map)

  rep_map <- read_achilles_replicate_map(replicate_map)
  dropped_reps <- get_dropped_replicates(rep_map)
  print(paste("Number of dropped batches:", length(dropped_reps)))

  dropped_guides <- readr::read_csv(dropped_guides)$X1
  print(paste("Number of dropped guides:", length(dropped_guides)))

  read_counts_df <- readr::read_csv(rc_file) %>%
    rename(sgrna = `Construct Barcode`) %>%
    filter(!sgrna %in% !!dropped_guides) %>%
    remove_columns(dropped_reps) %>%
    pivot_longer(-sgrna, names_to = "replicate_id", values_to = "read_counts") %>%
    inner_join(rep_map, by = "replicate_id") %>%
    inner_join(guide_map, by = "sgrna") %>%
    rename(n_sgrna_alignments = n_alignments)

  check_all_pass_qc(read_counts_df)
  readr::write_csv(read_counts_df, out_file)
}


tidy_crispr_gene_effect <- function(chronos_ge_file,
                                    ceres_ge_file,
                                    out_file) {
  chronos_ge <- readr::read_csv(chronos_ge_file) %>%
    flatten_wide_df_by_gene(
      values_to = "chronos_gene_effect",
      id_col_name = "DepMap_ID",
      rename_id_col_to = depmap_id,
      col_names_to = hugo_symbol
    )

  ceres_ge <- readr::read_csv(ceres_ge_file) %>%
    flatten_wide_df_by_gene(
      values_to = "ceres_gene_effect",
      id_col_name = "DepMap_ID",
      rename_id_col_to = depmap_id,
      col_names_to = hugo_symbol
    )

  ge_combined <- full_join(
    chronos_ge,
    ceres_ge,
    by = c("depmap_id", "hugo_symbol")
  )

  message(
    "Note: some data may be missing from the Chronos gene effect because Chronos does
not model guides that target multiple genes."
  )

  readr::write_csv(ge_combined, out_file)
}


# ---- Function calls ----

done <- function() {
  print("\nDONE\n\n")
}

print("---- Tidying known essential and non-essential genes. ----")
make_known_essentials_and_nonessentials(
  essentials_file = snakemake@input[["common_essentials"]],
  nonessentials_file = snakemake@input[["nonessentials"]],
  out_file = snakemake@output[["known_essentials"]]
)
done()

print("---- Tidying log fold change data. ----")
tidy_log_fold_change(
  lfc_file = snakemake@input[["achilles_logfold_change"]],
  replicate_map = snakemake@input[["achilles_replicate_map"]],
  dropped_guides = snakemake@input[["achilles_dropped_guides"]],
  guide_map = snakemake@input[["achilles_guide_map"]],
  out_file = snakemake@output[["achilles_log_fold_change"]]
)
done()

print("---- Tidying read count data. ----")
tidy_read_counts(
  rc_file = snakemake@input[["achilles_raw_readcounts"]],
  replicate_map = snakemake@input[["achilles_replicate_map"]],
  dropped_guides = snakemake@input[["achilles_dropped_guides"]],
  guide_map = snakemake@input[["achilles_guide_map"]],
  out_file = snakemake@output[["achilles_read_counts"]]
)
done()

print("---- Tidying CRISPR gene effect. ----")
tidy_crispr_gene_effect(
  chronos_ge_file = snakemake@input[["crispr_gene_effect_chronos"]],
  ceres_ge_file = snakemake@input[["crispr_gene_effect_ceres"]],
  out_file = snakemake@output[["crispr_gene_effect"]]
)
done()
