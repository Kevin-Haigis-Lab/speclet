# A table of information for the cell lines in modeling data.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)

# --- Snakemake interface ---

# Input
modeling_data_file <- snakemake@input[["modeling_df"]]
# Output
cell_line_info_file <- snakemake@output[["cell_line_info"]]


# --- Load data ---

data <- data.table::fread(modeling_data_file, showProgress = FALSE)
data <- as_tibble(data)


# --- DepMap modeling data cell line information ---

data %>%
  distinct(
    screen, depmap_id, lineage, lineage_subtype, is_male, primary_or_metastasis, age,
    replicate_id, p_dna_batch,
    hugo_symbol
  ) %>%
  count(
    screen, depmap_id, lineage, lineage_subtype, is_male, primary_or_metastasis, age,
    replicate_id, p_dna_batch,
    name = "n_genes"
  ) %>%
  arrange(lineage, lineage_subtype, depmap_id, screen, replicate_id, p_dna_batch) %>%
  write_csv(cell_line_info_file)
