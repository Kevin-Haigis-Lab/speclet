#!/usr/bin/env Rscript

# Isolate the DepMap data for CRC cell lines.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)


data <- data.table::fread(
  snakemake@input[["depmap_modeling_data"]],
  showProgress = FALSE
)
data <- as_tibble(data)

cell_lines <- data %>%
  distinct(
    depmap_id, lineage, lineage_subtype, primary_or_metastasis, kras_mutation
  )

sort(table(cell_lines$lineage, useNA = "ifany"))

CRC_LINEAGES <- "colorectal"

crc_data <- data %>%
  filter(lineage %in% !!CRC_LINEAGES)

write_csv(crc_data, snakemake@output[["depmap_CRC_data"]])


#### ---- Create a sub-sample for testing ---- ####

set.seed(0)
GENES <- sample(unique(crc_data$hugo_symbol), 50)

CELL_LINES <- crc_data %>%
  distinct(depmap_id, kras_mutation) %>%
  filter(kras_mutation %in% c("WT", "G12D", "G13D", "G12V")) %>%
  # group_by(kras_mutation) %>%
  # sample_n(3) %>%
  pull(depmap_id) %>%
  unlist()

crc_subsample <- crc_data %>%
  filter(hugo_symbol %in% !!GENES) %>%
  filter(depmap_id %in% !!CELL_LINES)

SGRNAS <- crc_subsample %>%
  distinct(hugo_symbol, sgrna) %>%
  # group_by(hugo_symbol) %>%
  # sample_n(3) %>%
  pull(sgrna) %>%
  unlist()

crc_subsample %>%
  filter(sgrna %in% SGRNAS) %>%
  write_csv(snakemake@output[["depmap_CRC_data_subsample"]])



#### ---- Create a larger sub-sample for testing ---- ####

set.seed(1)
GENES <- c(
  sample(unique(crc_data$hugo_symbol), 1000),
  "KRAS", "BRAF", "PIK3CA", "TP53", "JAK2", "RCL1", "MYCN",
  "TRPS1", "ESR1", "FOXA1", "GATA3", "GRHL2", "TFAP2C", "SPDEF", "ZNF652"
)
GENES <- unique(unlist(GENES))

large_crc_subsample <- crc_data %>%
  filter(hugo_symbol %in% !!GENES) %T>%
  write_csv(snakemake@output[["depmap_CRC_data_largesubsample"]])

gene_check_idx <- GENES %in% large_crc_subsample$hugo_symbol
if (!all(gene_check_idx)) {
  print("Not all expected genes in large data subsample.")
  print(GENES[!gene_check_idx])
}
