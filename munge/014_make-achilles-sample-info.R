#!/usr/bin/env Rscript

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(nakedpipe)
library(tidyverse)


modeling_data_dir <- "modeling_data"


# mut_file_path <- file.path(modeling_data_dir, "ccle_mutations.csv")
mut_file_path <- snakemake@input[["ccle_mutations"]]
all_samples_with_mutation_data <- read_csv(
  mut_file_path,
  col_types = cols("chromosome" = col_character())
) %>%
  pull(depmap_id) %>%
  unlist() %>%
  unique()

# kras_mutations_path <- file.path(modeling_data_dir, "kras_mutations.csv")
kras_mutations_path <- snakemake@input[["kras_mutations"]]
kras_mutations <- read_csv(kras_mutations_path) %>%
  select(depmap_id, kras_mutation)


noncancerous_lineages <- c("unknown", "embryo")

# sample_info_path <- file.path(modeling_data_dir, "sample_info.csv")
sample_info_path <- snakemake@input[["sample_info"]]
sample_info <- read_csv(sample_info_path) %.% {
  filter(!(lineage %in% !!noncancerous_lineages))
  filter(!str_detect(lineage, "engineer"))
  filter(depmap_id %in% !!all_samples_with_mutation_data)
  distinct(depmap_id, primary_or_metastasis, lineage, lineage_subtype)
  left_join(kras_mutations, by = "depmap_id")
  mutate(kras_mutation = ifelse(is.na(kras_mutation), "WT", kras_mutation))
}

qs::qsave(sample_info, file.path(snakemake@output[["out_file"]]))
