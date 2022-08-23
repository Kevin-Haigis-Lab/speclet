# Split the modeling data into files per sublineage and filter for Broad data only.

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(glue)
library(magrittr)
library(tidyverse)


MIN_NUM_CELLS <- 4
LUMP_LINEAGES <- c(
  "breast", "colorectal", "pancreas", "prostate", "upper_aerodigestive", "urinary_tract"
)


file_name_template <- snakemake@params[["file_name_template"]]

make_data_file_name <- function(lineage) {
  file.path(split_lineage_dir, glue(file_name_template, .open = "::", .close = "::"))
}


# Filter and write lineage data to file.
#
# Input:
#   modeling_data: full modeling data frame
#   lineage: name of the lineage
filter_and_write_lineage_data <- function(modeling_data, lineage) {
  lineage_data <- modeling_data %>% filter(lineage == !!lineage)
  n_cells <- length(unique(lineage_data$depmap_id))
  if (n_cells < MIN_NUM_CELLS) {
    print(glue("Not enough cells lines in '{lineage}': {n_cells}"))
    return(NULL)
  }
  fname <- make_data_file_name(lineage)
  print("Saving data for '{lineage}' ({n_cells} cell lines) to '{fname}'")
  write_csv(lineage_data, fname)
  return(NULL)
}


# Modify lineage names with the sublineage.
#
# Modifies the `lineage` column with `{lineage}_({sublineage})`.
#
# Input:
#   modeling_data: full modeling data frame
#   lineage: name of the lineage
#
# Return:
#   Lineage data with the modified `lineage` column.
modify_lineage_sublineage_names <- function(modeling_data, lineage) {
  lineage_df <- modeling_data %>% filter(lineage == !!lineage)
  names_df <- lineage_df %>%
    distinct(lineage, sublineage) %>%
    mutate(lineage = glue("{lineage}_({sublineage})"))
  lineage_df <- lineage_df %>%
    select(-lineage) %>%
    left_join(names_df, by="sublineage")
  return(lineage_df)
}


main <- function(modeling_df_file, out_dir) {
  modeling_data <- data.table::fread(modeling_df_file, showProgress = FALSE) %>%
    as_tibble() %>%
    filter(screen == "broad")

  lineages <- unique(modeling_data)
  for (lineage in lineages) {
    print(glue("Processing lineage '{lineage}'."))
    if (lineage in LUMP_LINEAGES) {
      filter_and_write_lineage_data(modeling_data, lineage)
    } else {
      mod_lineage_data <- modify_lineage_sublineage_names(modeling_data, lineage)
      new_lineages <- unique(mod_lineage_data$lineage)
      for (new_lineage in new_lineages) {
        filter_and_write_lineage_data(mod_lineage_data, new_lineage)
      }
    }
  }
}

# --- Run ---

split_lineage_dir <- snakemake@output[["split_lineage_dir"]]

if (!dir.exists(split_lineage_dir)) {
  dir.create(split_lineage_dir, recursive=FALSE)
}


# cell_line_info_file <- snakemake@input[["cell_line_info"]]
main(
  modeling_df_file=snakemake@input[["modeling_df"]],
  out_dir=split_lineage_dir
)
