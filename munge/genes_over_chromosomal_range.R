
library(memoise)
library(biomaRt)

mart <- useMart("ensembl")
mart <- useDataset("hsapiens_gene_ensembl", mart)


get_genes_on_region <- function(chr, start, end, remove_empty = TRUE) {
  attributes <- c(
    "chromosome_name",
    "start_position",
    "end_position",
    "strand",
    "hgnc_symbol"
  )

  filters <- c("chromosome_name", "start", "end")
  values <- list(
    chromosome = as.character(chr),
    start = as.character(start),
    end = as.character(end)
  )

  genes_over_region <- getBM(
    attributes = attributes,
    filters = filters,
    values = values,
    mart = mart
  ) %>%
    tibble::as_tibble() %>%
    dplyr::rename(hugo_symbol = hgnc_symbol)

  if (remove_empty) {
    genes_over_region <- dplyr::filter(genes_over_region, hugo_symbol != "")
  }

  return(genes_over_region)
}

get_genes_on_region <- memoise(get_genes_on_region)
