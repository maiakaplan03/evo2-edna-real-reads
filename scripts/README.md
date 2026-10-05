# Scripts behind each figure

**Readable, not re-runnable from this repository alone.** These are the scripts that produced the
figures in the main README, copied from the analysis tree. They read Evo 2 outputs that are too
large for GitHub (per-base probabilities for 200,898 real sequences ≈ 625 MB, mean-pooled
embeddings ≈ 1.6 GB) and real reads that are not distributed here (see *Data* in the main README).
Paths inside the scripts are relative to the original analysis folders, so they will not resolve in
this layout without the data. Session codes in comments (`s29`, `s37`, …) date each step.

The story notebook at the top of the repository does not run these scripts: it reads the small
tables in `data/`, which `00_export_notebook_tables.py` derived from the outputs of the scripts below
(copies, column subsets or group summaries; nothing recomputed).

| Figure | Folder | Scripts, in run order | Main inputs |
|---|---|---|---|
| 1 · per-base profile, simulated contamination | `01_simulated_contamination_profile/` | `agg_perbase_phase1.py` → `p1_profile_contamination_redraw.py` | Evo 2 per-base NLL of the simulated cohorts (H100 run) |
| 2 · perplexity by real class | `02_perplexity_real_classes/` | `s29_figs.py` | per-sequence perplexity of the real scoring set (H200 run) |
| 3–4 · embedding map (PCA, UMAP) | `03_embedding_map/` | `map1_build.py` → `w7_figures.py 3` | mean-pooled embeddings, real + simulated, joined through the 1,560 shared anchors |
| 5 · novelty ladder | `04_novelty_ladder/` | `map1_build.py` (shared PCA, in `03_embedding_map/`) → `w9_dims_and_ci.py` (scaled embedding, bootstrap CIs, nulls) → `w11_ladder_A_vs_D.py` (figure) | embeddings, vsearch identities to MIDORI |
| 6–7 · per-base profile, real off-targets | `05_offtargets/` | `e1_build_table.py` → `e3_features.py` → `p2_profile_offtargets_real.py` | per-base NLL, off-target labels |
| 8 · off-target AUROC vs baselines | `05_offtargets/` | `e1_build_table.py` → `e2_cluster_and_midori.sh` → `e3_features.py` → `e4_compare.py` → `e4_check.py` (integrity check) → `e5_quicklook_figures.py` (run with `DROP=onehot SUF=_noOH`) | all Evo 2 features + GC, 4-mers, MIDORI identity, COI profile score |
| 9 · chimeras | `06_chimeras/` | `run_chain.sh` (runs `j1` → `j5` → `j2` → `j3` → `j4` → `j7`) → `j3_run_benchmark.sh` (uchime_ref) → `k6_figures.py` | per-base A/C/G/T probabilities of the simulated C2 cohorts |

Notes:

- `w7_figures.py` and `w11_ladder_A_vs_D.py` also draw variants that are not shown in the README;
  the README figures are `fig3_map_pca`, `fig3_map_umap` and `ladder_D_standardised`.
- The last line of `run_chain.sh` also calls `k7_rc_figures.py`, which draws figures that are not
  part of this page and is not included.
- `e4_compare.py` can also score a raw one-hot copy of the read when that feature file exists. It is
  not part of the figures here.

Software: Python 3.10 with numpy, pandas, matplotlib, scikit-learn, scipy and umap-learn;
vsearch for clustering, MIDORI searches and uchime. Evo 2 (`evo2_7b`) was run on rented GPUs
(H100 for the simulated cohorts, H200 for the real reads).
