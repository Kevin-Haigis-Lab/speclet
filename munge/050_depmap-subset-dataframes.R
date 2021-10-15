
# Isolate the DepMap data for CRC cell lines.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)



# ---- Load data ----

modeling_df_path <- snakemake@input[["modeling_df"]]

data <- data.table::fread(
  modeling_df_path,
  showProgress = FALSE
)
data <- as_tibble(data)


# ---- CRC data ----

crc_lineages <- c("colorectal")

crc_data <- data %>%
  filter(lineage %in% !!crc_lineages)

write_csv(crc_data, snakemake@output[["crc_subset"]])


# ---- Sub-sample of CRC data ----

set.seed(0)

crc_cell_lines <- sample(unique(crc_data$depmap_id), 10)
genes <- sample(unique(crc_data$hugo_symbol), 100)

genes <- c(
  genes, "KRAS", "BRAF", "NRAS", "PIK3CA", "TP53", "MDM2", "MDM4", "APC", "FBXW7",
  "STK11", "PTK2", "CTNNB1", "KLF5",
  "GATA6"
)
genes <- unique(genes)

sample_sgrna_from_gene <- function(df, genes, n_sgrna) {
  df %>%
    filter(hugo_symbol %in% !!genes) %>%
    distinct(hugo_symbol, sgrna) %>%
    group_by(hugo_symbol) %>%
    sample_n(n_sgrna) %>%
    pull(sgrna) %>%
    unlist()
}

sgrnas <- sample_sgrna_from_gene(crc_data, genes, 3)

crc_data %>%
  filter(depmap_id %in% !!crc_cell_lines) %>%
  filter(sgrna %in% !!sgrnas) %>%
  write_csv(snakemake@output[["crc_subsample"]])


# ---- CRC + BONE data ----

bone_lineages <- c("bone")

bone_data <- data %>%
  filter(lineage %in% !!bone_lineages)

write_csv(bone_data, snakemake@output[["bone_subset"]])
bind_rows(crc_data, bone_data) %>%
  write_csv(snakemake@output[["crc_bone_subset"]])


# ---- Sub-sample of CRC + BONE data ----

bone_cell_lines <- sample(
  unique(crc_data$depmap_id),
  min(5, dplyr::n_distinct(crc_data$depmap_id))
)

cell_lines <- c(bone_cell_lines, sample(crc_cell_lines, 5))

data %>%
  filter(depmap_id %in% !!cell_lines) %>%
  filter(sgrna %in% !!sgrnas) %>%
  write_csv(snakemake@output[["crc_bone_subsample"]])


# ---- Small random subset for Python module testing ----

set.seed(123)

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
