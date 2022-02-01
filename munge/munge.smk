# A Snakemake pipeline for preparing raw data for analysis.

import os
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import init, Fore, Back, Style

from speclet import project_configuration as project_config

init(autoreset=True)

DATA_DIR = Path("data")
DEPMAP_DIR = DATA_DIR / "depmap_21q3"
SCORE_DIR = DATA_DIR / "score_21q3"
CCLE_DIR = DATA_DIR / "ccle_21q3"
SANGER_COSMIC_DIR = DATA_DIR / "sanger-cosmic"

MODELING_DATA_DIR = Path("modeling_data")
MUNGE_DIR = Path("munge")
TESTS_DIR = Path("tests")

MUNGE_CONFIG = project_config.read_project_configuration().munge
TEMP_DIR = MUNGE_CONFIG.temporary_directory

ENVIRONMENT_YAML = "pipeline-environment.yml"

all_depmap_ids = pd.read_csv(DATA_DIR / "all-depmap-ids.csv").depmap_id.to_list()

if MUNGE_CONFIG.test:
    print("---- TESTING WITH A FEW CELL LINES ----")
    all_depmap_ids = all_depmap_ids[:10]
    all_depmap_ids += ["ACH-002227", "ACH-001738", "ACH-000956"]
# all_depmap_ids = ["ACH-000956"]


#### ---- Inputs ---- ####


def tidy_ccle_input(*args: Any, **kwargs: Any) -> dict[str, Path]:
    return {
        "rna_expr": CCLE_DIR / "CCLE_expression.csv",
        "segment_cn": CCLE_DIR / "CCLE_segment_cn.csv",
        "gene_cn": CCLE_DIR / "CCLE_gene_cn.csv",
        "gene_mutations": CCLE_DIR / "CCLE_mutations.csv",
        "sample_info": CCLE_DIR / "CCLE_sample_info.csv",
    }


def tidy_depmap_input(*args: Any, **kwargs: Any) -> dict[str, Path]:
    return {
        "common_essentials": DEPMAP_DIR / "common_essentials.csv",
        "nonessentials": DEPMAP_DIR / "nonessentials.csv",
        "achilles_dropped_guides": DEPMAP_DIR / "Achilles_dropped_guides.csv",
        "achilles_guide_efficacy": DEPMAP_DIR / "Achilles_guide_efficacy.csv",
        "achilles_guide_map": DEPMAP_DIR / "Achilles_guide_map.csv",
        "achilles_logfold_change": DEPMAP_DIR / "Achilles_logfold_change.csv",
        "achilles_raw_readcounts": DEPMAP_DIR / "Achilles_raw_readcounts.csv",
        "achilles_replicate_map": DEPMAP_DIR / "Achilles_replicate_map.csv",
        "achilles_gene_effect_ceres_unscaled": DEPMAP_DIR
        / "Achilles_gene_effect_unscaled_CERES.csv",
        "crispr_data_sources": DEPMAP_DIR / "CRISPR_dataset_sources.csv",
        "crispr_gene_effect_chronos": DEPMAP_DIR / "CRISPR_gene_effect.csv",
        "crispr_gene_effect_ceres": DEPMAP_DIR / "CRISPR_gene_effect_CERES.csv",
        "crispr_common_essentials": DEPMAP_DIR / "CRISPR_common_essentials.csv",
    }


def tidy_score_input(*args: Any, **kwargs: Any) -> dict[str, Path]:
    return {
        "score_gene_effect_ceres_unscaled": SCORE_DIR
        / "Score_gene_effect_CERES_unscaled.csv",
        "score_log_fold_change": SCORE_DIR / "Score_log_fold_change.csv",
        "score_raw_readcounts": SCORE_DIR / "Score_raw_readcounts.csv",
        "score_guide_map": SCORE_DIR / "Score_guide_gene_map.csv",
        "score_replicate_map": SCORE_DIR / "Score_replicate_map.csv",
    }


def clean_sanger_cgc_input(*args: Any, **kwargs: Any) -> dict[str, Path]:
    return {"cgc_input": SANGER_COSMIC_DIR / "cancer_gene_census.csv"}


#### ---- CI ---- ####


def _touch_input_dict(input_dict: dict[str, Path]) -> None:
    for p in input_dict.values():
        if not p.parent.exists():
            print(Fore.YELLOW + f"  mkdir: '{p.parent.as_posix()}'")
            p.parent.mkdir(parents=True)
        if not p.exists():
            print(Style.DIM + f"  touch: '{p.as_posix()}'")
            p.touch()
    return None


if os.getenv("CI") is not None:
    print(Style.BRIGHT + Fore.BLUE + "CI: touch input files")
    input_dict_fxns = (
        tidy_ccle_input,
        tidy_depmap_input,
        tidy_score_input,
        clean_sanger_cgc_input,
    )
    for input_dict_fxn in input_dict_fxns:
        input_dict = input_dict_fxn()
        _touch_input_dict(input_dict)


#### ---- Rules ---- ####


localrules:
    all,


rule all:
    input:
        # rules.tidy_ccle
        MODELING_DATA_DIR / "ccle_expression.csv",
        MODELING_DATA_DIR / "ccle_segment_cn.csv",
        MODELING_DATA_DIR / "ccle_gene_cn.csv",
        MODELING_DATA_DIR / "ccle_mutations.csv",
        MODELING_DATA_DIR / "ccle_sample_info.csv",
        # rules.tidy_depmap
        MODELING_DATA_DIR / "known_essentials.csv",
        MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv",
        MODELING_DATA_DIR / "achilles_read_counts.csv",
        MODELING_DATA_DIR / "crispr_gene_effect.csv",
        # rules.tidy_score
        MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",
        MODELING_DATA_DIR / "score_read_counts.csv",
        # rules.combine_data
        MODELING_DATA_DIR / "depmap-modeling-data.csv",
        # rules.modeling_data_subsets.output
        MODELING_DATA_DIR / "depmap-modeling-data_crc.csv",
        MODELING_DATA_DIR / "depmap-modeling-data_crc-subsample.csv",
        TESTS_DIR / "depmap_test_data.csv",
        # rules.auxillary_data_subsets
        MODELING_DATA_DIR / "copy_number_data_samples.npy",
        # rules.clean_sanger_cgc.output
        MODELING_DATA_DIR / "sanger_cancer-gene-census.csv",
        # rules.screen_total_counts_tables
        final_counts_table_out=MODELING_DATA_DIR
        / "depmap_replicate_total_read_counts.csv",
        pdna_table_out=MODELING_DATA_DIR / "depmap_pdna_total_read_counts.csv",


# ---- Prepare raw CCLE data ----


rule tidy_ccle:
    input:
        **tidy_ccle_input(),
    output:
        rna_expr=MODELING_DATA_DIR / "ccle_expression.csv",
        segment_cn=MODELING_DATA_DIR / "ccle_segment_cn.csv",
        gene_cn=MODELING_DATA_DIR / "ccle_gene_cn.csv",
        gene_mutations=MODELING_DATA_DIR / "ccle_mutations.csv",
        sample_info=MODELING_DATA_DIR / "ccle_sample_info.csv",
    script:
        "010_prepare-ccle-raw-data.R"


# ---- Prepare raw DepMap / Achilles data ----


rule tidy_depmap:
    input:
        **tidy_depmap_input(),
    output:
        known_essentials=MODELING_DATA_DIR / "known_essentials.csv",
        achilles_log_fold_change=(
            MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv"
        ),
        achilles_read_counts=MODELING_DATA_DIR / "achilles_read_counts.csv",
        crispr_gene_effect=MODELING_DATA_DIR / "crispr_gene_effect.csv",
    script:
        "015_prepare-dempap-raw-data.R"


rule prep_achilles_pdna:
    input:
        guide_map=tidy_depmap_input()["achilles_guide_map"],
        dropped_guides=tidy_depmap_input()["achilles_dropped_guides"],
        replicate_map=tidy_depmap_input()["achilles_replicate_map"],
        achilles_read_counts=tidy_depmap_input()["achilles_raw_readcounts"],
    output:
        achilles_batch_pdna_counts=MODELING_DATA_DIR
        / "achilles_pdna_batch_read_counts.csv",
        achilles_replcate_pdna_counts=MODELING_DATA_DIR
        / "achilles_replicate_pdna_batch_read_counts.csv",
    script:
        "020_prepare-achilles-pdna-batch-read-counts.R"


# ---- Prepare raw Score data ----


rule prep_score_sgrna_library:
    input:
        sgrna_lib=(
            DATA_DIR
            / "Tzelepis_2016"
            / "TableS1_Lists-of-gRNAs-in-the-Mouse-v2-and-Human-v1-CRISPR-Libraries.xlsx"
        ),
    output:
        outfile=MODELING_DATA_DIR / "Tzelepis2016_score_guide_map.csv",
    script:
        "003_prepare-Tzelepis2016-score-sgrna-library.R"


rule collate_score_readcounts:
    input:
        replicate_map=tidy_score_input()["score_replicate_map"],
    params:
        raw_counts_dir=str(SCORE_DIR / "Score_raw_sgrna_counts"),
    output:
        score_raw_readcounts=tidy_score_input()["score_raw_readcounts"],
    script:
        "005_collate-score-readcounts.R"


rule extract_score_pdna:
    input:
        unzip_complete_touch=Path("temp") / "unzip_score_readcounts.done",
        replicate_map=tidy_score_input()["score_replicate_map"],
    params:
        raw_counts_dir=str(SCORE_DIR / "Score_raw_sgrna_counts"),
    output:
        score_pdna=MODELING_DATA_DIR / "score_pdna_batch_read_counts.csv",
    script:
        "007_extract-score-pdna-batch-read-counts.R"


rule tidy_score:
    input:
        **tidy_score_input(),
        tzelepis_sgnra_lib=rules.prep_score_sgrna_library.output.outfile,
    output:
        log_fold_change=MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",
        score_read_counts=MODELING_DATA_DIR / "score_read_counts.csv",
    script:
        "025_prepare-score-raw-data.R"


# ---- Split data by DepMap ID ----


rule split_ccle_rna_expression:
    input:
        data_file=rules.tidy_ccle.output.rna_expr,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-rna_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    script:
        "030_split-file-by-depmapid.R"


rule split_ccle_gene_cn:
    input:
        data_file=rules.tidy_ccle.output.gene_cn,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-genecn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "030_split-file-by-depmapid.R"


rule split_ccle_segment_cn:
    input:
        data_file=rules.tidy_ccle.output.segment_cn,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-segmentcn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "030_split-file-by-depmapid.R"


rule split_ccle_mutations:
    input:
        data_file=rules.tidy_ccle.output.gene_mutations,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-mut_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    script:
        "030_split-file-by-depmapid.R"


## Split DepMap data


rule split_achilles_lfc:
    input:
        data_file=rules.tidy_depmap.output.achilles_log_fold_change,
    output:
        out_files=expand(
            (TEMP_DIR / "achilles-lfc_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "030_split-file-by-depmapid.R"


rule split_achilles_rc:
    input:
        data_file=rules.tidy_depmap.output.achilles_read_counts,
    output:
        out_files=expand(
            (TEMP_DIR / "achilles-readcounts_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "030_split-file-by-depmapid.R"


## Split Score data


rule split_score_lfc:
    input:
        data_file=rules.tidy_score.output.log_fold_change,
    output:
        out_files=expand(
            (TEMP_DIR / "score-lfc_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    script:
        "030_split-file-by-depmapid.R"


rule split_score_rc:
    input:
        data_file=rules.tidy_score.output.score_read_counts,
    output:
        out_files=expand(
            (TEMP_DIR / "score-readcounts_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "030_split-file-by-depmapid.R"


## Combined Achilles and Score gene effect file


rule split_crispr_geneeffect:
    input:
        data_file=rules.tidy_depmap.output.crispr_gene_effect,
    output:
        out_files=expand(
            (TEMP_DIR / "crsipr-geneeffect_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "030_split-file-by-depmapid.R"


# ---- Merge all data for a DepMapID and combine into a single data set ----


rule merge_data:
    input:
        "munge/035_merge-modeling-data.R",
        ccle_rna=TEMP_DIR / "ccle-rna_{depmapid}.qs",
        ccle_gene_cn=TEMP_DIR / "ccle-genecn_{depmapid}.qs",
        ccle_segment_cn=TEMP_DIR / "ccle-segmentcn_{depmapid}.qs",
        ccle_mut=TEMP_DIR / "ccle-mut_{depmapid}.qs",
        achilles_lfc=TEMP_DIR / "achilles-lfc_{depmapid}.qs",
        achilles_readcounts=TEMP_DIR / "achilles-readcounts_{depmapid}.qs",
        achilles_pdna=rules.prep_achilles_pdna.output.achilles_batch_pdna_counts,
        score_lfc=TEMP_DIR / "score-lfc_{depmapid}.qs",
        score_readcounts=TEMP_DIR / "score-readcounts_{depmapid}.qs",
        score_pdna=rules.extract_score_pdna.output.score_pdna,
        crispr_geneeffect=TEMP_DIR / "crsipr-geneeffect_{depmapid}.qs",
        sample_info=MODELING_DATA_DIR / "ccle_sample_info.csv",
    output:
        out_file=TEMP_DIR / "merged_{depmapid}.qs",
    script:
        "035_merge-modeling-data.R"


rule combine_data:
    input:
        input_files=expand(
            (TEMP_DIR / "merged_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    output:
        out_file=MODELING_DATA_DIR / "depmap-modeling-data.csv",
    script:
        "040_combine-modeling-data.R"


rule screen_total_counts_tables:
    input:
        depmap_modeling_df=rules.combine_data.output.out_file,
        achillles_pdna_df=rules.prep_achilles_pdna.output.achilles_batch_pdna_counts,
        score_pdna_df=rules.extract_score_pdna.output.score_pdna,
    output:
        final_counts_table_out=MODELING_DATA_DIR
        / "depmap_replicate_total_read_counts.csv",
        pdna_table_out=MODELING_DATA_DIR / "depmap_pdna_total_read_counts.csv",
    conda:
        ENVIRONMENT_YAML
    shell:
        "munge/065_total-read-count-tables.py"
        " {input.depmap_modeling_df}"
        " {input.achillles_pdna_df}"
        " {input.score_pdna_df}"
        " {output.final_counts_table_out}"
        " {output.pdna_table_out}"


# ---- Check modeling data ----


rule papermill_check_depmap_modeling_data:
    input:
        modeling_df=rules.combine_data.output.out_file,
        template_nb=MUNGE_DIR / "045_check-depmap-modeling-data_original.ipynb",
    output:
        notebook=MUNGE_DIR / "045_check-depmap-modeling-data_exec.ipynb",
    run:
        papermill.execute_notebook(
            input.template_nb,
            output.notebook,
            parameters={
                "DEPMAP_MODELING_DF": input.modeling_df,
            },
            prepare_only=True,
        )


rule check_depmap_modeling_data:
    input:
        modeling_df=rules.combine_data.output.out_file,
        check_nb=rules.papermill_check_depmap_modeling_data.output.notebook,
    output:
        output_md=MUNGE_DIR / "045_check-depmap-modeling-data_exec.md",
    conda:
        ENVIRONMENT_YAML
    version:
        "1.1"
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.check_nb} && "
        "nbqa black {input.check_nb} --nbqa-mutate && "
        "nbqa isort {input.check_nb} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.check_nb}"


# ---- Generate additional useful files ----


rule modeling_data_subsets:
    input:
        check_output=rules.check_depmap_modeling_data.output.output_md,
        modeling_df=rules.combine_data.output.out_file,
    output:
        crc_subset=MODELING_DATA_DIR / "depmap-modeling-data_crc.csv",
        crc_subsample=(MODELING_DATA_DIR / "depmap-modeling-data_crc-subsample.csv"),
        test_data=TESTS_DIR / "depmap-modeling-data_test-data.csv",
        bone_subset=MODELING_DATA_DIR / "depmap-modeling-data_bone.csv",
        crc_bone_subset=MODELING_DATA_DIR / "depmap-modeling-data_crc-bone.csv",
        crc_bone_subsample=MODELING_DATA_DIR
        / "depmap-modeling-data_crc-bone-subsample.csv",
        crc_bone_large_subsample=MODELING_DATA_DIR
        / "depmap-modeling-data_crc-bone-large-subsample.csv",
    script:
        "050_depmap-subset-dataframes.R"


rule auxillary_data_subsets:
    input:
        check_output=rules.check_depmap_modeling_data.output.output_md,
        crc_subset=rules.modeling_data_subsets.output.crc_subset,
    output:
        cna_sample=MODELING_DATA_DIR / "copy_number_data_samples.npy",
    conda:
        ENVIRONMENT_YAML
    shell:
        "munge/055_auxiliary-data-files.py {input.crc_subset} {output.cna_sample}"


rule clean_sanger_cgc:
    input:
        **clean_sanger_cgc_input(),
    output:
        cgc_output=MODELING_DATA_DIR / "sanger_cancer-gene-census.csv",
    script:
        "060_prep-sanger-cgc.R"
