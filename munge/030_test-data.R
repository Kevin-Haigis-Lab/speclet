#!/usr/bin/env Rscript

# Prepare data to use is tests.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)

data_file <- snakemake@input[["depmap_modeling_df"]]
output_file <- snakemake@output[["output_dest"]]

filter_lineages <- function(df, pos) {
  KRAS_ALLELES <- c("WT", "G12D", "G13D", "G12C")
  LINEAGES <- c("lung", "colorectal")
  df %>%
    filter(lineage %in% LINEAGES) %>%
    filter(kras_mutation %in% KRAS_ALLELES)
}

data <- read_csv_chunked(
  data_file,
  callback = DataFrameCallback$new(filter_lineages),
  chunk_size = 10e6
)

set.seed(527)

cell_lines <- c(
  "ACH-000421",
  "ACH-000470",
  "ACH-000007",
  "ACH-001345",
  "ACH-000286",
  "ACH-000350",
  "ACH-000957",
  "ACH-000552",
  "ACH-000963",
  "ACH-000009",
  "ACH-000900",
  "ACH-000496",
  "ACH-000438",
  "ACH-000667",
  "ACH-000339",
  "ACH-000757",
  "ACH-000787",
  "ACH-000901",
  "ACH-000890",
  "ACH-001555"
)

# cell_lines <- data %>%
#   distinct(depmap_id, lineage, kras_mutation) %>%
#   arrange(lineage, kras_mutation) %>%
#   knitr::kable()

genes <- data %>%
  filter(depmap_id %in% !!cell_lines) %>%
  pull(hugo_symbol) %>%
  unlist() %>%
  unique() %>%
  sample(10, replace = FALSE)

sgrnas <- data %>%
  filter(hugo_symbol %in% !!genes) %>%
  distinct(hugo_symbol, sgrna) %>%
  group_by(hugo_symbol) %>%
  sample_n(3) %>%
  pull(sgrna) %>%
  unlist()

data %>%
  filter(depmap_id %in% !!cell_lines) %>%
  filter(sgrna %in% !!sgrnas) %>%
  slice_sample(1, prop = 1.0, replace = FALSE) %>%
  write_csv(output_file)
