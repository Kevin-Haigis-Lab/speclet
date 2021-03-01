#!/usr/bin/env Rscript

.libPaths("/home/jc604/R-4.0/library")

library(nakedpipe)
library(tidyverse)

if (basename(getwd()) == "munge") {
  setwd("..")
}


modeling_data_dir <- "modeling_data"



mut_file_path <- file.path(modeling_data_dir, "ccle_mutations.csv")
all_samples_with_mutation_data <- read_csv(mut_file_path, col_types = cols("chromosome" = col_character())) %>%
  pull(depmap_id) %>%
  unlist() %>%
  unique()

kras_mutations_path <- file.path(modeling_data_dir, "kras_mutations.csv")
kras_mutations <- read_csv(kras_mutations_path) %>%
  select(depmap_id, kras_mutation)


noncancerous_lineages <- c("unknown", "embryo")

sample_info_path <- file.path(modeling_data_dir, "sample_info.csv")
sample_info <- read_csv(sample_info_path) %.% {
  filter(!(lineage %in% !!noncancerous_lineages))
  filter(!str_detect(lineage, "engineer"))
  filter(depmap_id %in% !!all_samples_with_mutation_data)
  distinct(depmap_id, primary_or_metastasis, lineage, lineage_subtype)
  left_join(kras_mutations, by = "depmap_id")
  mutate(kras_mutation = ifelse(is.na(kras_mutation), "WT", kras_mutation))
}

qs::qsave(sample_info, file.path(snakemake@output[["out_file"]]))
