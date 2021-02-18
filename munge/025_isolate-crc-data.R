# Isolate the DepMap data for CRC cell lines.

library(nakedpipe)
library(magrittr)
library(tidyverse)

data_dir <- here::here("modeling_data")
data <- data.table::fread(file.path(data_dir, "depmap_modeling_dataframe.csv"), showProgress = FALSE)
data <- as_tibble(data)

cell_lines <- data %>%
  distinct(
    depmap_id, lineage, lineage_subtype, primary_or_metastasis, kras_mutation
  )

sort(table(cell_lines$lineage, useNA = "ifany"))

CRC_LINEAGES <- "colorectal"

crc_data <- data %>%
  filter(lineage %in% !!CRC_LINEAGES)

write_csv(crc_data, file.path(data_dir, "depmap_CRC_data.csv"))


#### ---- Create a sub-sample for testing ---- ####

set.seed(0)
GENES <- sample(unique(crc_data$hugo_symbol), 100)

crc_data %>%
  filter(hugo_symbol %in% !!GENES) %>%
  write_csv(file.path(data_dir, "depmap_CRC_data_subsample.csv"))
