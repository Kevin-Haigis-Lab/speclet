
if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(tidyverse)

input_files <- unlist(snakemake@input["input_files"])
output_file <- snakemake@output[["out_file"]]

# map_dfr(input_files, qs::qread) %>% write_csv(output_file)

append <- FALSE
for (f in input_files) {
  d <- qs::qread(f)
  if (nrow(d) > 0) {
    write_csv(d, output_file, append = append)
    append <- TRUE
  }
}

read_csv(output_file, n_max = 100) %>%
  glimpse()
