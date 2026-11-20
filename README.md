# Arabidopsis dry-seed 2x2x2 factorial reanalysis

Code and derived result tables for a reanalysis of the dry-seed RNA-seq factorial in
GSE270988: all eight combinations of `abi3-12`, `dog1-4` and `aba2-1`, three
replicates per genotype, modelled as a balanced 2x2x2 design. The analysis estimates
how much of the expressed dry-seed transcriptome departs from log-scale additivity,
whether each pairwise interaction depends on the genotype at the third locus, which
genetic interaction types (synthetic, synergistic, masking, suppression, ...) underlie
the deviations, and whether dormancy, chlorophyll retention and seed longevity of the
same genotypes follow the same pattern.

No wet-lab work was performed here. Everything is derived from public data.

## Data sources

| Source | Content | Used for |
|---|---|---|
| GEO GSE270988 | 123 seed-maturation RNA-seq libraries (SeedMatExplorer) | 24 dry-seed libraries for the factorial; 36 libraries at 20/22/24 DAP; `cyp707a2` combination, `abi3-6`, `DOG1-Cvi` lines |
| GEO GSE251921 | processed 3' UMI counts, Col-0 and `dog1-4` dry seeds (4 + 4) | external dataset V1 |
| GEO GSE5634, GPL198 | AtGenExpress seed stages 6-10, ATH1 annotation | external dataset V2 |
| Artur et al. 2026, Additional file 3 | seed trait measurements (Tables S10, S11) and ABI3 targets of Tian et al. 2020 (Table S8) | trait factorial; ABI3-target overlap |
| GO Consortium | `tair.gaf.gz`, `go-basic.obo` | GO enrichment |
| Ensembl Plants release 60 | TAIR10 genome FASTA and GFF3 | promoter elements |

The GEO files are included under `data/raw` exactly as downloaded
(`data/raw/external/*/download.log` records the retrieval of the external sets). The
count matrix `GSE270988_raw_counts.csv` is the deposited StringTie/prepDE table.

Two inputs are not redistributed and must be fetched before running scripts 17-21:

1. **Supplementary tables of the source study.** Download Additional file 3
   (`12870_2026_9313_MOESM3_ESM.xlsx`) from the article page
   (https://doi.org/10.1186/s12870-026-09313-z) and save it as
   `data/raw/supplemental_tables.xlsx`. The article is published under
   CC BY-NC-ND 4.0, so the file is not copied into this repository.
2. **Reference annotation and genome.** Run `python src/fetch_references.py`. It downloads
   the GO files and the Ensembl Plants TAIR10 genome and annotation with resumable range
   requests and writes their URLs, sizes and SHA-256 sums to
   `data/raw/external/reference_files.json`. The sums of the files used for the results
   in this repository are already recorded there and checked against study-pinned hashes.
   If an upstream GO release has changed, the script stops rather than silently rewriting
   the study baseline. The separate `reference_data_for_release.zip` snapshot can be
   uploaded as a GitHub Release asset and unpacked at the repository root.

## Environment

Python 3.10.9 with numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.9.4,
joblib 1.6.0 and openpyxl (see `requirements.txt`). The negative-binomial routines are
implemented in `src/nb_glm.py`; no Bioconductor package is needed. Random seeds are
fixed, so the result tables are reproducible on the same library versions.

## Layout

```
src/             analysis and figure scripts, numbered in run order
data/raw/        deposited GEO files and reference checksums
data/processed/  sample table, frozen 24-sample design, input hashes
results/         derived tables, JSON summaries and run logs
```

## Running

All paths are resolved relative to the repository root.

```
python src/01_build_sample_table.py     # sample table from the count-matrix header
python src/02_build_main_design.py      # freeze the 24-sample 2x2x2 design
python src/03_qc.py                     # library statistics, filter counts, PCA, correlations
python src/04_pilot.py                  # timing and convergence check on a gene subset
python src/05_calibration.py --mode null --runs 500 --genes 2000 --jobs 12 --tag full500
python src/05_calibration.py --mode power --runs 60 --genes 2000 --jobs 12 --effect 0.5 --tag ad0.5
python src/05_calibration.py --mode power --runs 60 --genes 2000 --jobs 12 --effect 1.0 --tag ad1.0
python src/05_calibration.py --mode power --runs 60 --genes 2000 --jobs 12 --effect 1.5 --tag ad1.5
python src/05_calibration.py --mode power --runs 60 --genes 2000 --jobs 12 --effect3 1.5 --tag adb1.5
python src/06_main_analysis.py          # original nominal statistics; retained unchanged
python src/22_empirical_calibration.py  # empirical calibration, held-out null FDR evaluation
python src/07_sensitivity.py --analysis filter
python src/07_sensitivity.py --analysis norm
python src/07_sensitivity.py --analysis dispersion
python src/07_sensitivity.py --analysis ql
python src/07_sensitivity.py --analysis loo
python src/08_stage_support.py          # within-stage abi3 x dog1 at 20/22/24 DAP
python src/09_internal_support.py       # cyp707a2 combination, abi3-6 and DOG1-Cvi
python src/10_freeze_external.py        # freeze gene sets and gates before opening V1/V2
python src/11_validate_v1.py            # GSE251921
python src/12_validate_v2.py            # GSE5634 / GPL198
python src/14_candidate_audit.py        # numerical-stability audit of candidates
python src/fetch_references.py          # GO and TAIR10 reference files (see above)
python src/17_trait_factorial.py        # dormancy, chlorophyll and EPPO longevity factorial
python src/18_epistasis.py              # epistasis coefficients, model simulations, interaction types
python src/19_functional_annotation.py  # GO enrichment, ABI3 targets, literature gene panel
python src/20_promoter_elements.py      # RY / ABRE / G-box / DRE promoter enrichment
python src/13_external_figures.py       # external datasets figure
python src/15_stability_figure.py       # candidate stability figure
python src/16_main_figures.py           # design, QC, background dependence, stage support
python src/21_interaction_figures.py    # traits, epistasis, annotation and model figures
```

`08_stage_support.py` freezes the dry-seed signature in `results/stages/frozen_signature.json`
the first time it runs and refuses to proceed if that file no longer matches its stored
SHA-256 hash. `10_freeze_external.py` is run before the external matrices are inspected;
the frozen rules are stored in `results/external/frozen_external.json` and scripts 11 and
12 implement them literally.

## Outputs

- `results/qc` - library-level metrics, filter sweep, PCA coordinates, replicate correlations
- `results/tables` - per-gene LRT and contrast tables (`lrt_all_genes.csv.gz`, `contrasts_all_genes.csv.gz`), candidate lists, the stability audit, group means
- `results/sensitivity` - one file per sensitivity variant, plus summaries
- `results/calibration`, `results/power` - null and power simulations
- `results/stages` - per-stage interaction estimates and the frozen dry-seed signature
- `results/internal` - the same-study contrasts
- `results/external` - frozen external gene lists and the V1/V2 results
- `results/traits` - heteroscedastic cell-mean contrasts, BH across 42 interactions, 168 unique cell comparisons, ageing-curve AUC, secondary isotonic P50 and probit lack-of-fit diagnostics
- `results/epistasis` - per-gene single, double and interaction effects for the six pair-background combinations, epistasis coefficients with simulated model expectations, interaction-type counts
- `results/function` - gene sets, GO enrichment, ABI3-target overlap and the literature gene panel
- `results/promoters` - per-gene promoter element counts and enrichment tests
- `results/empirical_calibration` - 300 calibration and 100 independent evaluation simulations, the recalibrated primary table and held-out error-rate estimates

Figure scripts write to `results/figures`, which is not included; every figure is
rebuilt from the tables above.

## Interpretation notes

"Interaction" always means a departure from additivity on the log scale in a
negative-binomial model; it is not evidence of physical binding or direct transcriptional
control. The stage panel, the additional lines and the trait data come from the same
experiment as the primary data, so they show consistency rather than independent
replication; trait replicates are not paired with RNA-seq libraries and are analysed at
genotype level. The epistasis, trait, GO and promoter analyses were added after the
pre-specified primary analysis and are exploratory. Nominal q values are retained for
traceability, not treated as perfectly calibrated. The original all-null rejected-gene
proportion was 0.17%, but the chance of any rejection was 93.4%: those quantities are
not interchangeable. Empirical calibration gave 3/100 all-null LRT datasets and 6/100
seven-contrast Wald datasets with any BH rejection at q < 0.05. These finite-simulation
estimates address pooled model behaviour, not exact gene-conditional or biological FDR.
Non-significant differences are not equivalence tests or proof of one pathway.

Ageing AUC is the primary trait summary. The original probit fit returned negative P50
for three triple-mutant curves despite initial viability above 50%, so those fits are
retained as diagnostics rather than clamped to zero. Empirical P50 is interpolated from
a monotone curve. Trait comparisons use separate genotype variances, not pooled variance.

## License

MIT for the code in this repository. See `LICENSE`. Third-party data keep their own
terms.
