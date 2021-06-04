#!/bin/env python3

source("renv/activate.R")

try(
  options(prompt = glue::glue("{clisymbols::symbol$pointer} ")),
  silent = TRUE
)

try(
  options(continue = glue::glue("{clisymbols::symbol$dot} ")),
  silent = TRUE
)

options(max.print = 100)

qq <- function(save = "no") {
  q(save = save)
}
