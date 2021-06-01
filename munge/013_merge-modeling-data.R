#!/usr/bin/env Rscript

if (basename(getwd()) == "munge") {
  setwd("..")
}

source(".Rprofile")

library(tidyverse)

ccle_rna <- snakemake@input[["ccle_rna"]]
ccle_gene_cn <- snakemake@input[["ccle_gene_cn"]]
ccle_segment_cn <- snakemake@input[["ccle_segment_cn"]]
ccle_mut <- snakemake@input[["ccle_mut"]]
achilles_lfc <- snakemake@input[["achilles_lfc"]]
score_cn <- snakemake@input[["score_cn"]]
score_lfc <- snakemake@input[["score_lfc"]]
sample_info <- snakemake@input[["sample_info"]]

out_file <- snakemake@output[["out_file"]]

# ccle_rna <- qs::qread("temp/ccle-rna_ACH-000001.qs")
# ccle_gene_cn <- qs::qread("temp/ccle-genecn_ACH-000001.qs")
# ccle_segment_cn <- qs::qread("temp/ccle-segmentcn_ACH-000001.qs")
# ccle_mut <- qs::qread("temp/ccle-mut_ACH-000001.qs")
# achilles_lfc <- qs::qread("temp/achilles-lfc_ACH-000001.qs")
# score_cn <- qs::qread("temp/score-segmentcn_ACH-000001.qs")
# score_lfc <- qs::qread("temp/score-lfc_ACH-000001.qs")
# sample_info <- read_csv("modeling_data/ccle_sample_info.csv")


read_csv(sample_info, n_max = 1) %>%
  qs::qsave(out_file)
