#!/usr/bin/env Rscript

# Prepare some of the auxillary DepMap files.


#### ---- Setup ---- ####

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)


#### ---- guide efficacy --- ####

prepare_guide_efficacy_data <- function(guide_efficacy_path, guide_efficacy_dest) {
  if (!file.exists(guide_efficacy_path)) {
    stop("Guide efficacy file not found.")
  }

  read_csv(guide_efficacy_path) %>%
    write_csv(guide_efficacy_dest)
  invisible(NULL)
}



#### ---- Essentials and non-essential genes ---- ####

prepare_essentials_and_nonessentials <- function(essentials_df_path, nonessentials_df_path, dest_path) {
  essentials_df <- read_csv(essentials_df_path) %>%
    mutate(
      gene = unlist(stringr::str_split_fixed(gene, " ", 2)[, 1]),
      essential = TRUE
    )
  nonessentials_df <- read_csv(nonessentials_df_path) %>%
    mutate(
      gene = unlist(stringr::str_split_fixed(gene, " ", 2)[, 1]),
      essential = FALSE
    )

  if (any(essentials_df$gene %in% nonessentials_df$gene)) {
    stop("A gene cannot be essential and non-essential.")
    print(intersect(unlist(essentials_df$gene), unlist(nonessentials_df$gene)))
  }

  bind_rows(essentials_df, nonessentials_df) %>%
    write_csv(dest_path)
}


#### ---- MAIN ---- ####

prepare_guide_efficacy_data(
  guide_efficacy_path = snakemake@input[["guide_efficacy"]],
  guide_efficacy_dest = snakemake@output[["guide_efficacy"]]
)

prepare_essentials_and_nonessentials(
  essentials_df_path = snakemake@input[["common_essentials"]],
  nonessentials_df_path = snakemake@input[["nonessentials"]],
  dest_path = snakemake@output[["achilles_essential_genes"]]
)
