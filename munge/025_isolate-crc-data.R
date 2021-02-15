# Isolate the DepMap data for CRC cell lines.

library(nakedpipe)
library(magrittr)
library(tidyverse)

data_dir <- here::here("modeling_data")
data <- read_csv(file.path(data_dir, "depmap_modeling_dataframe.csv"), guess_max = 1e5)

cell_lines <- data %>%
  distinct(
    depmap_id, lineage, lineage_subtype, primary_or_metastasis, kras_mutation
  )

sort(table(cell_lines$lineage, useNA = "ifany"))

CRC_LINEAGES <- "colorectal"

data %>%
  filter(lineage %in% !!CRC_LINEAGES) %>%
  write_csv(file.path(data_dir, "depmap_CRC_data.csv"))
