# Make various sub-samples of the DepMap modeling data.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)

set.seed(1009)

genes_of_interest <- c(
  "KRAS", "BRAF", "NRAS", "PIK3CA", "TP53", "MDM2", "MDM4", "APC", "FBXW7",
  "STK11", "PTK2", "CTNNB1", "KLF5", "GATA6"
)


# --- Load data ---

modeling_df_path <- snakemake@input[["modeling_df"]]

data <- data.table::fread(modeling_df_path, showProgress = FALSE)
data <- as_tibble(data)


# --- CRC data ---

crc_lineages <- c("colorectal")

crc_data <- data %>%
  filter(lineage %in% !!crc_lineages)

write_csv(crc_data, snakemake@output[["crc_subset"]])


# --- Sub-sample of CRC data ---

crc_cell_lines <- sample(unique(crc_data$depmap_id), 10)
genes <- sample(unique(crc_data$hugo_symbol), 100)

genes <- c(genes, genes_of_interest)
genes <- unique(genes)

sample_sgrna_from_gene <- function(df, genes, n_sgrna) {
  df %>%
    filter(hugo_symbol %in% !!genes) %>%
    distinct(hugo_symbol, sgrna) %>%
    slice_sample(prop = 1) %>%
    group_by(hugo_symbol) %>%
    slice_head(n = n_sgrna) %>%
    pull(sgrna) %>%
    unlist()
}

sgrnas <- sample_sgrna_from_gene(crc_data, genes, 3)

crc_data %>%
  filter(depmap_id %in% !!crc_cell_lines) %>%
  filter(sgrna %in% !!sgrnas) %>%
  write_csv(snakemake@output[["crc_subsample"]])


# --- CRC + BONE data ---

bone_lineages <- c("bone")

bone_data <- data %>%
  filter(lineage %in% !!bone_lineages)

write_csv(bone_data, snakemake@output[["bone_subset"]])
bind_rows(crc_data, bone_data) %>%
  write_csv(snakemake@output[["crc_bone_subset"]])


# --- Sub-sample of CRC + BONE data ---

bone_cell_lines <- sample(
  unique(bone_data$depmap_id),
  min(5, dplyr::n_distinct(bone_data$depmap_id))
)

print("BONE CELL LINES:")
print(bone_cell_lines)

cell_lines <- c(bone_cell_lines, sample(crc_cell_lines, 5))

data %>%
  filter(depmap_id %in% !!cell_lines) %>%
  filter(sgrna %in% !!sgrnas) %>%
  write_csv(snakemake@output[["crc_bone_subsample"]])


# --- Larger sub-sample of CRC + BONE data ---

cell_lines <- c(unique(bone_data$depmap_id), unique(crc_data$depmap_id))
genes <- sample(unique(data$hugo_symbol), 2000)

data %>%
  filter(depmap_id %in% !!cell_lines) %>%
  filter(hugo_symbol %in% !!genes) %>%
  write_csv(snakemake@output[["crc_bone_large_subsample"]])


# --- Larger sub-sample of colorectal + pancreas + cervix ---

lineages <- c("colorectal", "pancreas", "cervix")
cervix_genes <- c("POLRMT", "LATS1")
pancreas_genes <- c("EEF2", "CDKN2A")

data %>%
  filter(lineage %in% !!lineages) %>%
  filter(hugo_symbol %in% c(genes, genes_of_interest, cervix_genes, pancreas_genes)) %>%
  write_csv(snakemake@output[["crc_panc_cervix_large_subsample"]])


# --- Small random subset for Python module testing ---

lineages <- sample(unique(data$lineage), 2)
cell_lines <- data %>%
  filter(lineage %in% !!lineages) %>%
  distinct(lineage, depmap_id) %>%
  group_by(lineage) %>%
  sample_n(3) %T>%
  print() %>%
  pull(depmap_id) %>%
  unlist()


test_data <- data %>%
  filter(depmap_id %in% !!cell_lines)

genes <- sample(unique(crc_data$hugo_symbol), 10)
sgrnas <- sample_sgrna_from_gene(test_data, genes, 2)

test_data %>%
  filter(hugo_symbol %in% !!genes) %>%
  filter(sgrna %in% sgrnas) %>%
  write_csv(snakemake@output[["test_data"]])
