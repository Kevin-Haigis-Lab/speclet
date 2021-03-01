#!/usr/bin/env Rscript

.libPaths("/home/jc604/R-4.0/library")

library(tidyverse)

if (basename(getwd()) == "munge") {
  setwd("..")
}


input_files <- unlist(snakemake@input["input_files"])
output_file <- snakemake@output[["out_file"]]


map_dfr(input_files, qs::qread) %>% write_csv(output_file)
