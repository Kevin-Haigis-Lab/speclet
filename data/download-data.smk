from pathlib import Path
from typing import Dict, List

data_dir = Path("data")
depmap_dir = data_dir / "depmap_21q2"
score_dir = data_dir / "score_21q2"
ccle_dir = data_dir / "ccle_21q2"

depmap_downloads: Dict[str, str] = {
    "Achilles_gene_effect_unscaled.csv": "https://ndownloader.figshare.com/files/27902055",
    "Achilles_gene_effect.csv": "https://ndownloader.figshare.com/files/27902046",
    "Achilles_guide_efficacy.csv": "https://ndownloader.figshare.com/files/27902052",
    "Achilles_common_essentials.csv": "https://ndownloader.figshare.com/files/27902028",
    "Achilles_cell_line_efficacy.csv": "https://ndownloader.figshare.com/files/27902034",
    "Achilles_cell_line_growth_rate.csv": "https://ndownloader.figshare.com/files/27902025",
    "common_essentials.csv": "https://ndownloader.figshare.com/files/27902160",
    "nonessentials.csv": "https://ndownloader.figshare.com/files/27902370",
    "Achilles_logfold_change.csv": "https://ndownloader.figshare.com/files/27902121",
    "Achilles_logfold_change_failures.csv": "https://ndownloader.figshare.com/files/27902067",
    "Achilles_guide_map.csv": "https://ndownloader.figshare.com/files/27902061",
    "Achilles_replicate_map.csv": "https://ndownloader.figshare.com/files/27902085",
    "Achilles_replicate_QC_report_failing.csv": "https://ndownloader.figshare.com/files/27902088",
    "Achilles_dropped_guides.csv": "https://ndownloader.figshare.com/files/27902037",
    "Achilles_high_variance_genes.csv": "https://ndownloader.figshare.com/files/27902064",
    "CRISPR_gene_effect_Chronos.csv": "https://ndownloader.figshare.com/files/27902229",
    "CRISPR_gene_dependency_Chronos.csv": "https://ndownloader.figshare.com/files/27902175",
    "CRISPR_common_essentials_Chronos.csv": "https://ndownloader.figshare.com/files/27902166",
}

score_downloads: Dict[str, str] = {
    "SCORE_gene_effect.csv": "https://ndownloader.figshare.com/files/16623881",
    "SCORE_gene_dependency.csv": "https://ndownloader.figshare.com/files/16623884",
    "SCORE_gene_effect_unscaled.csv": "https://ndownloader.figshare.com/files/16623851",
    "SCORE_logfold_change.csv": "https://ndownloader.figshare.com/files/16623878",
    "SCORE_essential_genes.txt": "https://ndownloader.figshare.com/files/16623887",
    "SCORE_nonessential_genes.csv": "https://ndownloader.figshare.com/files/16623890",
    "SCORE_replicate_map.csv": "https://ndownloader.figshare.com/files/16623896",
    "SCORE_guide_gene_map.csv": "https://ndownloader.figshare.com/files/16623902",
    "SCORE_guide_efficacy.csv": "https://ndownloader.figshare.com/files/16623905",
    "SCORE_copy_number.csv": "https://ndownloader.figshare.com/files/16623893",
}

ccle_downloads: Dict[str, str] = {
    "CCLE_expression.csv": "https://ndownloader.figshare.com/files/27902091",
    "CCLE_segment_cn.csv": "https://ndownloader.figshare.com/files/27902157",
    "CCLE_gene_cn.csv": "https://ndownloader.figshare.com/files/27902124",
    "CCLE_mutations.csv": "https://ndownloader.figshare.com/files/27902118",
    "CCLE_mutations_bool_hotspot.csv": "https://ndownloader.figshare.com/files/27902130",
    "CCLE_mutations_bool_damaging.csv": "https://ndownloader.figshare.com/files/27902127",
    "CCLE_mutations_bool_nonconserving.csv": "https://ndownloader.figshare.com/files/27902133",
    "CCLE_mutations_bool_otherconserving.csv": "https://ndownloader.figshare.com/files/27902136",
    "CCLE_sample_info.csv": "https://ndownloader.figshare.com/files/27902376",
}


rule all:
    input:
        depmap_files=expand(
            depmap_dir / "{filename}", filename=list(depmap_downloads.keys())
        ),
        score_files=expand(
            score_dir / "{filename}", filename=list(score_downloads.keys())
        ),
        ccle_files=expand(
            ccle_dir / "{filename}", filename=list(ccle_downloads.keys())
        ),


rule download_depmap:
    output:
        filename=depmap_dir / "{filename}",
    params:
        url=lambda w: depmap_downloads[w.filename],
    shell:
        "wget --output-document {output.filename} {params.url}"


rule download_score:
    output:
        filename=score_dir / "{filename}",
    params:
        url=lambda w: score_downloads[w.filename],
    shell:
        "wget --output-document {output.filename} {params.url}"


rule download_ccle:
    output:
        filename=ccle_dir / "{filename}",
    params:
        url=lambda w: ccle_downloads[w.filename],
    shell:
        "wget --output-document {output.filename} {params.url}"
