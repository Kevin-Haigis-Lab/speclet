
# Prepare Sanger CGC data.

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(glue)
library(tidyverse)


#### ---- Snakemake interfacing ---- ####

raw_cgc_file <- unlist(snakemake@input["cgc_input"])
cgc_out_file <- unlist(snakemake@output["cgc_output"])


#### ---- Data Processing ---- ####

# Skip empty strings in an array of strings.
skip_empty <- function(l) {
  return(l[l != ""])
}

# Condense an array of strings into a single, clean string.
condense_array_of_strings <- function(l, collapse = ";") {
  l %>%
    unlist() %>%
    str_squish() %>%
    unique() %>%
    skip_empty() %>%
    sort() %>%
    paste0(collapse = ";")
}

# Clean `tumor_types_somatic` column.
clean_tumor_types_somatic <- function(tts) {
  y <- str_split(tts, ",") %>%
    condense_array_of_strings()
  return(y)
}

# Clean `mutation_types` column. (Also used with the `tissue_type` column...)
clean_mutation_types <- function(mt) {
  y <- str_split(mt, "[:punct:]") %>%
    condense_array_of_strings()
  return(y)
}


cgc_df <- read_csv(raw_cgc_file) %>%
  janitor::clean_names() %>%
  filter(str_to_lower(somatic) == "yes" && !is.na(somatic)) %>%
  select(
    hugo_symbol = gene_symbol, tier, hallmark,
    tumor_types_somatic = tumour_types_somatic,
    tissue_type, role_in_cancer, mutation_types
  ) %>%
  mutate(
    hallmark = str_to_lower(hallmark),
    hallmark = ifelse(is.na(hallmark), "no", hallmark),
    hallmark = (hallmark == "yes")
  ) %>%
  mutate(
    role_in_cancer = str_to_lower(role_in_cancer),
    role_in_cancer = ifelse(is.na(role_in_cancer), "unknown", role_in_cancer),
    is_oncogene = str_detect("oncogene", role_in_cancer),
    is_tsg = str_detect("tsg", role_in_cancer),
    is_fusion = str_detect("fusion", role_in_cancer),
  ) %>%
  mutate(
    tumor_types_somatic = purrr::map_chr(
      tumor_types_somatic, clean_tumor_types_somatic
    ),
    mutation_types = purrr::map_chr(mutation_types, clean_mutation_types),
    tissue_type = purrr::map_chr(tissue_type, clean_mutation_types)
  ) %>%
  select(-role_in_cancer) %>%
  arrange(hugo_symbol, tier, hallmark, tumor_types_somatic)


#### ---- Check for missing data ---- ####

if (any(is.na(cgc_df))) {
  any_na <- apply(cgc_df, 1, function(r) {
    any(is.na(r))
  })
  glimpse(cgc_df[any_na, ])
  stop("Missing data in CGC data frame.")
} else {
  print("No missing data points in CGC data frame.")
}


#### ---- Info ---- ####

print(
  knitr::kable(head(cgc_df), format = "markdown")
)

print(glue("dimensions: {nrow(cgc_df)}, {ncol(cgc_df)}"))

#### ---- Write data ---- ####

print(glue("writing data frame to file: '{cgc_out_file}'"))
write_csv(cgc_df, cgc_out_file)
