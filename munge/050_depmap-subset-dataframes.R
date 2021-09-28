
# Isolate the DepMap data for CRC cell lines.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)



# ---- Load data ----

modeling_df_path <- snakemake@input[["modeling_df"]]
modeling_df_path <- "modeling_data/depmap_modeling_dataframe.csv"

data <- data.table::fread(
  modeling_df_path,
  showProgress = FALSE
)
data <- as_tibble(data)


CRC_LINEAGES <- c("colorectal")

crc_data <- data %>%
  filter(lineage %in% !!CRC_LINEAGES)

write_csv(crc_data, snakemake@output[["crc_subset"]])


# ---- Create a sub-sample for testing ----

set.seed(0)

CELL_LINES <- sample(unique(crc_data$depmap_id), 10)
GENES <- sample(unique(crc_data$hugo_symbol), 100)

GENES <- c(
  GENES, "KRAS", "BRAF", "NRAS", "PIK3CA", "TP53", "MDM2", "MDM4", "APC", "FBXW7",
  "STK11", "PTK2", "CTNNB1", "KLF5",
  "GATA6"
)
GENES <- unique(GENES)


sample_sgrna_from_gene <- function(df, genes, n_sgrna) {
  df %>%
    filter(hugo_symbol %in% !!genes) %>%
    distinct(hugo_symbol, sgrna) %>%
    group_by(hugo_symbol) %>%
    sample_n(n_sgrna) %>%
    pull(sgrna) %>%
    unlist()
}

SGRNAS <- sample_sgrna_from_gene(crc_data, GENES, 3)

crc_data %>%
  filter(depmap_id %in% !!CELL_LINES) %>%
  filter(sgrna %in% !!SGRNAS) %>%
  write_csv(snakemake@output[["crc_subsample"]])


# ---- Small random subset for Python module testing ----

set.seed(123)

LINEAGES <- sample(unique(data$lineage), 2)
CELL_LINES <- data %>%
  filter(lineage %in% !!LINEAGES) %>%
  distinct(lineage, depmap_id) %>%
  group_by(lineage) %>%
  sample_n(3) %T>%
  print() %>%
  pull(depmap_id) %>%
  unlist()


test_data <- data %>%
  filter(depmap_id %in% !!CELL_LINES)

GENES <- sample(unique(crc_data$hugo_symbol), 10)
SGRNAS <- SGRNAS <- sample_sgrna_from_gene(test_data, GENES, 2)

test_data %>%
  filter(hugo_symbol %in% !!GENES) %>%
  filter(sgrna %in% SGRNAS) %>%
  write_csv(snakemake@output[["test_data"]])
