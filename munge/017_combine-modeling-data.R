#!/usr/bin/env Rscript

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(tidyverse)

input_files <- unlist(snakemake@input["input_files"])
output_file <- snakemake@output[["out_file"]]


map_dfr(input_files, qs::qread) %>% write_csv(output_file)
