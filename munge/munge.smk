# A Snakemake pipeline for preparing raw data for analysis.

import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from colorama import init, Fore, Back, Style

init(autoreset=True)

DATA_DIR = Path("data")
DEPMAP_DIR = DATA_DIR / "depmap_21q2"
SCORE_DIR = DATA_DIR / "score_21q2"
CCLE_DIR = DATA_DIR / "ccle_21q2"
SANGER_COSMIC_DIR = DATA_DIR / "sanger-cosmic"

MODELING_DATA_DIR = Path("modeling_data")
MUNGE_DIR = Path("munge")
TESTS_DIR = Path("tests")

# TEMP_DIR = Path("/n/no_backup2/dbmi/park/jc604/speclet/munge-intermediates")
TEMP_DIR = Path("temp")

ENVIRONMENT_YAML = "pipeline-environment.yml"

all_depmap_ids = pd.read_csv(DATA_DIR / "all-depmap-ids.csv").depmap_id.to_list()

print("---- TESTING WITH A FEW CELL LINES ----")
all_depmap_ids = all_depmap_ids[:10]  ### TESTING ###
# all_depmap_ids += ["ACH-002227", "ACH-001738"]


#### ---- Inputs ---- ####


def tidy_ccle_input(*args: Any, **kwargs: Any) -> Dict[str, Path]:
    return {
        "rna_expr": CCLE_DIR / "CCLE_expression.csv",
        "segment_cn": CCLE_DIR / "CCLE_segment_cn.csv",
        "gene_cn": CCLE_DIR / "CCLE_gene_cn.csv",
        "gene_mutations": CCLE_DIR / "CCLE_mutations.csv",
        "sample_info": CCLE_DIR / "CCLE_sample_info.csv",
    }


def tidy_depmap_input(*args: Any, **kwargs: Any) -> Dict[str, Path]:
    return {
        "common_essentials": DEPMAP_DIR / "common_essentials.csv",
        "nonessentials": DEPMAP_DIR / "nonessentials.csv",
        "achilles_dropped_guides": DEPMAP_DIR / "Achilles_dropped_guides.csv",
        "achilles_guide_efficacy": DEPMAP_DIR / "Achilles_guide_efficacy.csv",
        "achilles_guide_map": DEPMAP_DIR / "Achilles_guide_map.csv",
        "achilles_gene_effect": DEPMAP_DIR / "Achilles_gene_effect.csv",
        "achilles_gene_effect_unscaled": DEPMAP_DIR
        / "Achilles_gene_effect_unscaled.csv",
        "achilles_logfold_change": DEPMAP_DIR / "Achilles_logfold_change.csv",
        "achilles_raw_readcounts": DEPMAP_DIR / "Achilles_raw_readcounts.csv",
        "achilles_replicate_map": DEPMAP_DIR / "Achilles_replicate_map.csv",
        "all_gene_effect_chronos": DEPMAP_DIR / "CRISPR_gene_effect_Chronos.csv",
    }


def tidy_score_input(*args: Any, **kwargs: Any) -> Dict[str, Path]:
    return {
        "copy_number": SCORE_DIR / "SCORE_copy_number.csv",
        "gene_effect": SCORE_DIR / "SCORE_gene_effect.csv",
        "gene_effect_unscaled": SCORE_DIR / "SCORE_gene_effect_unscaled.csv",
        "log_fold_change": SCORE_DIR / "SCORE_logfold_change.csv",
        "guide_map": SCORE_DIR / "SCORE_guide_gene_map.csv",
        "replicate_map": SCORE_DIR / "SCORE_replicate_map.csv",
    }


def clean_sanger_cgc_input(*args: Any, **kwargs: Any) -> Dict[str, Path]:
    return {"cgc_input": SANGER_COSMIC_DIR / "cancer_gene_census.csv"}


#### ---- CI ---- ####


def _touch_input_dict(input_dict: Dict[str, Path]) -> None:
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
        "005_prepare-ccle-raw-data.R"


rule tidy_depmap:
    input:
        **tidy_depmap_input(),
    output:
        known_essentials=MODELING_DATA_DIR / "known_essentials.csv",
        achilles_log_fold_change=(
            MODELING_DATA_DIR / "achilles_log_fold_change_filtered.csv"
        ),
        achilles_read_counts=MODELING_DATA_DIR / "achilles_read_counts.csv",
        achilles_gene_effect=MODELING_DATA_DIR / "achilles_gene_effect.csv",
        chronos_gene_effect=MODELING_DATA_DIR / "chronos_gene_effect.csv",
    script:
        "007_prepare-dempap-raw-data.R"


rule tidy_score:
    input:
        **tidy_score_input(),
    output:
        copy_number=MODELING_DATA_DIR / "score_segment_cn.csv",
        gene_effect=MODELING_DATA_DIR / "score_gene_effect.csv",
        log_fold_change=MODELING_DATA_DIR / "score_log_fold_change_filtered.csv",
    script:
        "009_prepare-score-raw-data.R"


rule split_ccle_rna_expression:
    input:
        data_file=rules.tidy_ccle.output.rna_expr,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-rna_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_ccle_gene_cn:
    input:
        data_file=rules.tidy_ccle.output.gene_cn,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-genecn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_ccle_segment_cn:
    input:
        data_file=rules.tidy_ccle.output.segment_cn,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-segmentcn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_ccle_mutations:
    input:
        data_file=rules.tidy_ccle.output.gene_mutations,
    output:
        out_files=expand(
            (TEMP_DIR / "ccle-mut_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_achilles_lfc:
    input:
        data_file=rules.tidy_depmap.output.achilles_log_fold_change,
    output:
        out_files=expand(
            (TEMP_DIR / "achilles-lfc_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_achilles_rc:
    input:
        data_file=rules.tidy_depmap.output.achilles_read_counts,
    output:
        out_files=expand(
            (TEMP_DIR / "achilles-readcounts_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_score_cn:
    input:
        data_file=rules.tidy_score.output.copy_number,
    output:
        out_files=expand(
            (TEMP_DIR / "score-segmentcn_{depmapid}.qs").as_posix(),
            depmapid=all_depmap_ids,
        ),
    script:
        "011_split-file-by-depmapid.R"


rule split_score_lfc:
    input:
        data_file=rules.tidy_score.output.log_fold_change,
    output:
        out_files=expand(
            (TEMP_DIR / "score-lfc_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    script:
        "011_split-file-by-depmapid.R"


# Merge all data for a DepMapID.
rule merge_data:
    input:
        ccle_rna=TEMP_DIR / "ccle-rna_{depmapid}.qs",
        ccle_gene_cn=TEMP_DIR / "ccle-genecn_{depmapid}.qs",
        ccle_segment_cn=TEMP_DIR / "ccle-segmentcn_{depmapid}.qs",
        ccle_mut=TEMP_DIR / "ccle-mut_{depmapid}.qs",
        achilles_lfc=TEMP_DIR / "achilles-lfc_{depmapid}.qs",
        achilles_readcounts=TEMP_DIR / "achilles-readcounts_{depmapid}.qs",
        score_cn=TEMP_DIR / "score-segmentcn_{depmapid}.qs",
        score_lfc=TEMP_DIR / "score-lfc_{depmapid}.qs",
        sample_info=MODELING_DATA_DIR / "ccle_sample_info.csv",
    output:
        out_file=TEMP_DIR / "merged_{depmapid}.qs",
    version:
        "1"
    script:
        "013_merge-modeling-data.R"


rule combine_data:
    input:
        input_files=expand(
            (TEMP_DIR / "merged_{depmapid}.qs").as_posix(), depmapid=all_depmap_ids
        ),
    output:
        out_file=MODELING_DATA_DIR / "depmap_modeling_dataframe.csv",
    script:
        "015_combine-modeling-data.R"


rule check_depmap_modeling_data:
    input:
        modeling_df=rules.combine_data.output.out_file,
        check_nb=MUNGE_DIR / "017_check-depmap-modeling-data.ipynb",
    output:
        output_md=MUNGE_DIR / "017_check-depmap-modeling-data.md",
    conda:
        ENVIRONMENT_YAML
    shell:
        "jupyter nbconvert --to notebook --inplace --execute {input.check_nb} && "
        "nbqa black {input.check_nb} --nbqa-mutate && "
        "nbqa isort {input.check_nb} --nbqa-mutate && "
        "jupyter nbconvert --to markdown {input.check_nb}"


rule modeling_data_subsets:
    input:
        check_output=rules.check_depmap_modeling_data.output.output_md,
        modeling_df=rules.combine_data.output.out_file,
    output:
        crc_subset=MODELING_DATA_DIR / "depmap_modeling_dataframe_crc.csv",
        crc_subsample=(
            MODELING_DATA_DIR / "depmap_modeling_dataframe_crc-subsample.csv"
        ),
        test_data=TESTS_DIR / "depmap_test_data.csv",
    script:
        "019_depmap-subset-dataframes.R"


rule auxillary_data_subsets:
    input:
        check_output=rules.check_depmap_modeling_data.output.output_md,
        crc_subset=rules.modeling_data_subsets.output.crc_subset,
    output:
        cna_sample=MODELING_DATA_DIR / "copy_number_data_samples.npy",
    conda:
        ENVIRONMENT_YAML
    script:
        "021_auxiliary-data-files.py"


rule clean_sanger_cgc:
    input:
        **clean_sanger_cgc_input(),
    output:
        cgc_output=MODELING_DATA_DIR / "sanger_cancer-gene-census.csv",
    script:
        "025_prep-sanger-cgc.R"


rule all:
    input:
        rules.tidy_ccle.output,
        rules.tidy_depmap.output,
        rules.tidy_score.output,
        rules.combine_data.output,
        rules.modeling_data_subsets.output,
        rules.auxillary_data_subsets.output,
        rules.clean_sanger_cgc.output,
