# Prepare the guide map for Project Score (downloaded from Tzelepis , 2016).

if (basename(getwd()) == "munge") {
  setwd("..")
}
source(".Rprofile")

library(tidyverse)

source("munge/munge_functions.R")


read_tzelepis2016_xlsx <- function(path) {
  readxl::read_xlsx(path, sheet = "Human v1 CRISPR library") %>%
    janitor::clean_names()
}


prepare_tzelepis2016_guide_map <- function(guide_map_xlsx) {
  guide_map <- read_tzelepis2016_xlsx(guide_map_xlsx) %>%
    select(sgrna_id = g_rna_id, sgrna = guide_sequence) %>%
    arrange(sgrna_id)
  return(guide_map)
}


prepare_tzelepis2016_guide_map(
  guide_map_xlsx = snakemake@input[["sgrna_lib"]]
) %>%
  readr::write_csv(snakemake@output[["outfile"]])
