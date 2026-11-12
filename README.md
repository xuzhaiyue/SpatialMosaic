# SpatialMosaic

Analysis code for *"Dual-scale supervision turns H&E slides into calibrated virtual molecular maps and reveals a prognostic proliferation–interferon co-localization in breast cancer"*.

The code is a set of analysis scripts, not an installable package. Data are not included.

## Data
| Input | Source |
|---|---|
| Spatial transcriptomics (He2020, HER2ST A–D, Meylan ccRCC, 10x breast) | HEST-1k |
| Bulk RNA-seq, clinical data, diagnostic slides | TCGA-BRCA, TCGA-KIRC (GDC); CPTAC-BRCA, CPTAC-CCRCC (PDC / TCIA) |
| Patch features | Public UNI2-h features (MahmoodLab) |
| TIL maps | Saltz et al. 2018 (GDC publication page) |

Data paths are set at the top of each `*_common.py` and in `config/`.

## Code
Scripts are numbered by stage and run in order from `scripts/`.

| Stage | Scripts | Analysis |
|---|---|---|
| Targets | `t3b_background`, `t4_targets_discovery`, `t5_st_bulk` | Gene background; Hallmark UCell scores for ST spots and bulk RNA |
| ST features and ST-only model | `t4_features_discovery`, `t4_models_discovery` | UNI2-h features at 128 µm per spot; ridge baseline |
| Dual-scale model | `t6_mean_features`, `t6_fit`, `t7_fit` | Patient cross-fitted ridge over ST and bulk; λ selected on validation |
| Whole-slide maps and features | `t7_wsi_predict`, `t6_features`, `t7_features` | Patch predictions; Moran's I, high fraction, local entropy, co-localization |
| Clinical associations | `t6_clinical`, `t7_associations`, `t10_clinical_review`, `t10_coloc_x_e2f`, `t10_sampling_stripes` | Cox and PAM50 associations; design review; sampling-stripe check |
| External ST | `t8_extract`, `t8_measure`, `t8_analyze`, `t10_ceiling` | HEST 10x breast sections; noise ceiling |
| TIL | `t8_1b_analyze_til` | Co-localization vs Saltz TIL density and pattern |
| Programme atlas, backbones, PAM50 | `t9_*`, `t9_pam50_call.R` | 58-programme atlas; CONCH, DenseNet-121, ST-Net; external PAM50 |
| Figures | `t10_st_data`, `t10_fig1`–`t10_fig5`, `t10_supp` | Main and supplementary figures with source data |

## Environments
`env/hemorph.yml` covers all analyses. `env/smbackbone.yml` covers the backbone comparison. R 4.5.2 with genefu 2.42.0 runs the PAM50 calls (`env/R_sessionInfo.txt`). Developed on Apple silicon with PyTorch MPS.

## Licence
CC0 1.0 Universal.
