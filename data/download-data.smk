from pathlib import Path
import os

import requests
from requests.auth import HTTPBasicAuth
import colorama
from snakemake.io import directory, touch

colorama.init(autoreset=True)

data_dir = Path("data")
temp_dir = Path("temp")
depmap_dir = data_dir / "depmap_21q3"
score_dir = data_dir / "score_21q3"
ccle_dir = data_dir / "ccle_21q3"
sanger_cosmic_dir = data_dir / "sanger-cosmic"

depmap_downloads: dict[str, str] = {
    "Achilles_cell_line_efficacy.csv": "https://ndownloader.figshare.com/files/29124108",
    "Achilles_cell_line_growth_rate.csv": "https://ndownloader.figshare.com/files/29124111",
    "Achilles_common_essentials.csv": "https://ndownloader.figshare.com/files/29124102",
    "Achilles_dropped_guides.csv": "https://ndownloader.figshare.com/files/29124123",
    "Achilles_gene_effect.csv": "https://ndownloader.figshare.com/files/29124198",
    "Achilles_gene_effect_CERES.csv": "https://ndownloader.figshare.com/files/29124195",
    "Achilles_gene_effect_unscaled_CERES.csv": "https://ndownloader.figshare.com/files/29124486",
    "Achilles_guide_efficacy.csv": "https://ndownloader.figshare.com/files/29124483",
    "Achilles_guide_map.csv": "https://ndownloader.figshare.com/files/29124492",
    "Achilles_logfold_change.csv": "https://ndownloader.figshare.com/files/29125299",
    "Achilles_logfold_change_failures.csv": "https://ndownloader.figshare.com/files/29124513",
    "Achilles_raw_readcounts.csv": "https://ndownloader.figshare.com/files/29124612",
    "Achilles_raw_readcounts_failures.csv": "https://ndownloader.figshare.com/files/29124720",
    "Achilles_replicate_map.csv": "https://ndownloader.figshare.com/files/29124735",
    "Achilles_replicate_QC_report_failing.csv": "https://ndownloader.figshare.com/files/29124738",
    "CRISPR_common_essentials.csv": "https://ndownloader.figshare.com/files/29125284",
    "CRISPR_dataset_sources.csv": "https://ndownloader.figshare.com/files/29125290",
    "CRISPR_gene_effect.csv": "https://ndownloader.figshare.com/files/29125323",
    "CRISPR_gene_effect_CERES.csv": "https://ndownloader.figshare.com/files/29125326",
    "nonessentials.csv": "https://ndownloader.figshare.com/files/29125332",
    "common_essentials.csv": "https://ndownloader.figshare.com/files/29125281",
}

score_downloads: dict[str, str] = {
    "Score_log_fold_change.csv": "https://ndownloader.figshare.com/files/16623878",
    "Score_gene_effect_CERES.csv": "https://ndownloader.figshare.com/files/16623881",
    "Score_gene_effect_CERES_unscaled.csv": "https://ndownloader.figshare.com/files/16623851",
    "Score_gene_effect_CHRONOS.csv": "https://ndownloader.figshare.com/files/28340607",
    "Score_guide_gene_map.csv": "https://ndownloader.figshare.com/files/16623902",
    "Score_replicate_map.csv": "https://ndownloader.figshare.com/files/16623896",
    "Score_raw_sgrna_counts.zip": "https://cog.sanger.ac.uk/cmp/download/raw_sgrnas_counts.zip",
}

ccle_downloads: dict[str, str] = {
    "CCLE_expression.csv": "https://ndownloader.figshare.com/files/29124747",
    "CCLE_gene_cn.csv": "https://ndownloader.figshare.com/files/29125230",
    "CCLE_segment_cn.csv": "https://ndownloader.figshare.com/files/29125278",
    "CCLE_mutations.csv": "https://ndownloader.figshare.com/files/29125233",
    "CCLE_mutations_bool_hotspot.csv": "https://ndownloader.figshare.com/files/29125245",
    "CCLE_mutations_bool_damaging.csv": "https://ndownloader.figshare.com/files/29125239",
    "CCLE_mutations_bool_nonconserving.csv": "https://ndownloader.figshare.com/files/29125248",
    "CCLE_mutations_bool_otherconserving.csv": "https://ndownloader.figshare.com/files/29125251",
    "CCLE_sample_info.csv": "https://ndownloader.figshare.com/files/29162481",
}


#### ---- Get URL for Sanger's CGC ---- ####


def get_sanger_cgc_url() -> str:
    sanger_email = os.getenv("SANGER_EMAIL")
    sanger_pass = os.getenv("SANGER_PASS")
    sanger_cgc_response = requests.get(
        "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v94/cancer_gene_census.csv",
        auth=HTTPBasicAuth(sanger_email, sanger_pass),
    )
    if sanger_cgc_response.status_code == 200:
        print(
            colorama.Fore.BLUE
            + colorama.Style.BRIGHT
            + "Successfully recieved a unique Cancer Gene Census URL."
        )
        url = sanger_cgc_response.json()["url"]
        if url is None:
            print(sanger_cgc_response.json())
            BaseException("Successul request, but no URL was returned.")
        return url
    BaseException("Unable to retrieve a unique Cancer Gene Census URL.")


CI = os.getenv("CI")

if CI is None:
    sanger_cgc_url = get_sanger_cgc_url()
else:
    print(colorama.Fore.YELLOW + f"CI: using fake URL for Sanger CGC file.")
    sanger_cgc_url = "www.mock-url.com"


#### ---- Rules ---- ####


rule all:
    input:
        depmap_files=expand(
            depmap_dir / "{filename}", filename=list(depmap_downloads.keys())
        ),
        score_files=expand(
            score_dir / "{filename}", filename=list(score_downloads.keys())
        ),
        ccle_files=expand(ccle_dir / "{filename}", filename=list(ccle_downloads.keys())),
        sanger_cgc=sanger_cosmic_dir / "cancer_gene_census.csv",
        depmap_id_list=data_dir / "all-depmap-ids.csv",
        # download_score_sgrna_library
        download_score_sgrna_library_file=(
            data_dir
            / "Tzelepis_2016"
            / "TableS1_Lists-of-gRNAs-in-the-Mouse-v2-and-Human-v1-CRISPR-Libraries.xlsx"
        ),
        # download_bailey_mutations
        bailey_mutations=data_dir / "bailey-2018-cell" / "bailey-cancer-genes.xlsx",


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


rule download_sanger:
    output:
        cgc_filename=sanger_cosmic_dir / "cancer_gene_census.tsv",
    params:
        url=sanger_cgc_url,
    shell:
        "wget --output-document {output.cgc_filename} {params.url}"


rule download_score_sgrna_library:
    params:
        url="https://www.cell.com/cms/10.1016/j.celrep.2016.09.079/attachment/0cb71c0e-1d31-40c3-898b-3215a39a2bae/mmc2.xlsx",
    output:
        out_file=(
            data_dir
            / "Tzelepis_2016"
            / "TableS1_Lists-of-gRNAs-in-the-Mouse-v2-and-Human-v1-CRISPR-Libraries.xlsx"
        ),
    shell:
        "wget --output-document {output.out_file} {params.url}"


rule unzip_score_readcounts:
    input:
        zipped_read_counts=score_dir / "Score_raw_sgrna_counts.zip",
    params:
        destination_dir=score_dir,
        default_unzipped_dir=score_dir / "00_raw_counts",
    output:
        raw_counts_dir=directory(score_dir / "Score_raw_sgrna_counts"),
        unzip_complete_touch=touch(temp_dir / "unzip_score_readcounts.done"),
    shell:
        "unzip -q {input.zipped_read_counts} -d {params.destination_dir}"
        " && mv {params.default_unzipped_dir} {output.raw_counts_dir}"


rule make_depmap_id_list:
    input:
        achilles_replicate_map=depmap_dir / "Achilles_replicate_map.csv",
        score_replicate_map=score_dir / "Score_replicate_map.csv",
        unzip_complete_touch=temp_dir / "unzip_score_readcounts.done",
    params:
        raw_counts_dir=score_dir / "Score_raw_sgrna_counts",
    output:
        depmap_id_list=data_dir / "all-depmap-ids.csv",
    version:
        "1.0"
    shell:
        "./data/list_all_depmapids.py"
        "  {output.depmap_id_list}"
        "  --achilles={input.achilles_replicate_map}"
        "  --score={input.score_replicate_map}"
        "  --score-reads-dir={params.raw_counts_dir}"


rule download_bailey_mutations:
    params:
        url="https://www.cell.com/cms/10.1016/j.cell.2018.02.060/attachment/b0abfafa-a1ff-4f7d-8842-b49ba8d32e08/mmc1.xlsx",
    output:
        out_file=data_dir / "bailey-2018-cell" / "bailey-cancer-genes.xlsx",
    version:
        "1.0"
    shell:
        "wget --output-document {output.out_file} {params.url}"
