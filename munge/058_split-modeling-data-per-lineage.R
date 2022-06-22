# Split the modeling data into files per lineage.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(magrittr)
library(tidyverse)

# --- Snakemake interface ---

# Input
cell_line_info_file <- snakemake@input[["cell_line_info"]]
modeling_df_file <- snakemake@input[["modeling_df"]]
# Parameters
file_name_template <- snakemake@params[["file_name_template"]]
# Output
split_lineage_dir <- snakemake@output[["split_lineage_dir"]]


if (!dir.exists(split_lineage_dir)) {
    dir.create(split_lineage_dir, recursive=FALSE)
}


# --- Read data ---

modeling_data <- data.table::fread(modeling_df_file, showProgress = FALSE)
modeling_data <- as_tibble(modeling_data)


# --- Write data per lineage ---

lineages <- unqiue(modeling_data$lineage)

lineage_data_file_name <- function(lineage) {
    file.path(split_lineage_dir, glue::glue(file_name_template))
}

for (lineage in lineages) {
    modeling_data %>%
        filter(lineage == !!lineage) %>%
        write_csv(lineage_data_file_name(lineage))
}
