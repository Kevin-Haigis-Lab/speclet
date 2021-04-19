#!/usr/bin/env Rscript

# Prepare some of the auxillary DepMap files.


#### ---- Setup ---- ####

library(magrittr)
library(tidyverse)


data_dir <- file.path("data", "depmap_20q3")
destination_dir <- file.path("modeling_data")

#### ---- guide efficacy --- ####


prepare_guide_efficacy_data <- function() {
  guide_efficacy_path <- file.path(data_dir, "Achilles_guide_efficacy.csv")
  guide_efficacy_dest <- file.path(destination_dir, "achilles_guide_efficacy.csv")

  if (!file.exists(guide_efficacy_path)) {
    stop("Guide efficacy file not found.")
  }

  read_csv(guide_efficacy_path) %>%
    write_csv(guide_efficacy_dest)
  invisible(NULL)
}



#### ---- Essentials and non-essential genes ---- ####

prepare_essentials_and_nonessentials <- function() {
  essentials_df <- read_csv(file.path(data_dir, "common_essentials.csv")) %>%
    mutate(
      gene = unlist(stringr::str_split_fixed(gene, " ", 2)[, 1]),
      essential = TRUE
    )
  nonessentials_df <- read_csv(file.path(data_dir, "nonessentials.csv")) %>%
    mutate(
      gene = unlist(stringr::str_split_fixed(gene, " ", 2)[, 1]),
      essential = FALSE
    )

  if (any(essentials_df$gene %in% nonessentials_df$gene)) {
    stop("A gene cannot be essential and non-essential.")
    print(intersect(unlist(essentials_df$gene), unlist(nonessentials_df$gene)))
  }

  bind_rows(essentials_df, nonessentials_df) %>%
    write_csv(file.path(destination_dir, "achilles_essential_genes.csv"))
}


#### ---- MAIN ---- ####

prepare_guide_efficacy_data()
prepare_essentials_and_nonessentials()
