# SpatialMosaic

Analysis code for *"Dual-scale supervision turns H&E slides into calibrated virtual molecular maps and reveals an outcome-associated proliferation–interferon co-localization in breast cancer"*.

The code is a set of analysis scripts, not an installable package. Data are not included.

## Data
| Input | Source |
|---|---|
| Spatial transcriptomics (He2020, HER2ST A–D, Meylan ccRCC, 10x breast) | HEST-1k |
| Bulk RNA-seq, clinical data, diagnostic slides | TCGA-BRCA, TCGA-KIRC (GDC); CPTAC-BRCA, CPTAC-CCRCC (PDC / TCIA) |
| Patch features | Public UNI2-h features (MahmoodLab) |
| TIL maps | Saltz et al. 2018 (GDC publication page) |

Data paths are set at the top of each `*_common.py` and in `config/`.

## Inputs
`inputs/` holds the small frozen inputs derived from the public sources above, laid out at the paths the scripts read: the HEST section manifest, slide–patient inventories and patient table, the CPTAC RNA sample manifest, the 9,036-gene protein-coding scoring background and programme gene members, and the download manifests (source URLs and checksums). Copy them into the repository root before running:

```bash
cp -R inputs/. .
```

Two third-party files are not redistributed and should be placed under `references/`:

| Path | Source |
|---|---|
| `references/gene_sets/msigdb_2026.2.Hs/h.all.v2026.2.Hs.symbols.gmt` | MSigDB Hallmark, release 2026.2.Hs |
| `references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json` | `utils_data/symbol2ensembl.json` from github.com/Graph-and-Geometric-Learning/STPath, commit 3b10e6d |

## Code
Scripts are numbered by stage (t3–t10) and run in numerical order. Run them from the repository root with all script folders on the path:

```bash
export PYTHONPATH=$(ls -d "$PWD"/scripts/*/ | tr '\n' :)
```

| Folder | Scripts | Analysis |
|---|---|---|
| `scripts/common/` | `t5_common` … `t10_common` | Shared paths, UCell, spatial statistics |
| `scripts/model/` | `t3b_background`, `t4_*`, `t5_st_bulk`, `t6_mean_features`, `t6_fit`, `t7_fit`, `t7_wsi_predict`, `t6_features`, `t7_features` | Targets, UNI2-h spot features, ST-only and dual-scale ridge, whole-slide maps, spatial features |
| `scripts/validation/` | `t8_extract`, `t8_measure`, `t8_analyze`, `t10_st_data`, `t10_ceiling`, `t9_atlas_*`, `t9_backbone_*`, `t9_stnet_finetune` | Held-out and HEST external ST, noise ceiling, 58-programme atlas, backbone comparison |
| `scripts/clinical/` | `t6_clinical`, `t7_associations`, `t10_clinical_review`, `t10_coloc_x_e2f`, `t10_sampling_stripes`, `t8_1b_analyze_til`, `t9_pam50_*` | Cox and PAM50 associations, design review, sampling-stripe check, TIL, external PAM50 |
| `scripts/figures/` | `t10_fig1`–`t10_fig5`, `t10_supp` | Main and supplementary figures with source data |

## Environments
`env/hemorph.yml` covers all analyses. `env/smbackbone.yml` covers the backbone comparison. R 4.5.2 with genefu 2.42.0 runs the PAM50 calls (`env/R_sessionInfo.txt`). Developed on Apple silicon with PyTorch MPS.

The T4 feature and model scripts verify SHA-256 hashes of the authors' analysis plan and configuration (`common/t4_display_provenance.py`). The plan itself is not distributed; when `PLAN.md` is absent these checks are skipped.

## Licence
CC0 1.0 Universal.
