#!/bin/env Rscript

.libPaths("/home/jc604/R-4.0/library")

library(tidyverse)

if (basename(getwd()) == "munge") { setwd("..") }


data_file <- snakemake@input[["data_file"]]
output_files <- unlist(snakemake@output["out_files"])


nested_d <- data.table::fread(data_file) %>%
    as_tibble() %>%
    group_by(depmap_id) %>%
    nest()

empty_data_df <- nested_d$data[[1]] %>% slice(0)


a <- tibble(output_file = output_files) %>%
    mutate(
        depmap_id = basename(output_file),
        depmap_id = tools::file_path_sans_ext(depmap_id),
        depmap_id = str_split_fixed(depmap_id, "_", 2)[, 2]
    ) %>%
    left_join(nested_d, by = "depmap_id") %>%
    pwalk(function(output_file, depmap_id, data) {
        if (is.null(data)) { data <- empty_data_df }
        qs::qsave(data, output_file)
    })
