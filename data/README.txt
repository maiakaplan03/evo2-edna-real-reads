Tables read by evo2_real_reads_story.ipynb. Built by scripts/00_export_notebook_tables.py from the outputs
of the scripts in scripts/ (copies, column subsets or group summaries; nothing recomputed).
No read sequences and no read identifiers of real reads are included.

01_simulated_profiles.npz          per-position NLL (mean, 25th/75th percentile, n) of 7 simulated cohorts
02_real_perplexity.csv.gz          class + mean perplexity of each of the 200,898 real sequences
03_embedding_map.csv.gz            group + 2-D PCA and UMAP coordinates of 101,897 embedded reads
04_novelty_ladder.csv              embedding AUROC, 95 % bootstrap CI and null per identity band and per genus x band
05_offtarget_profiles.csv.gz       per-position NLL (mean, 25th/75th percentile) per library x off-target group
05_offtarget_auroc.csv             AUROC, CI and null of every score for the three off-target comparisons
06_chimera_detection.csv           uchime_ref and Evo 2 entropy scores of the simulated chimera benchmark
06_chimera_localisation_T6.csv     estimated vs true junction of 1,500 C2-T6 chimeras
06_chimera_confidence_T6.csv       detection score and localisation error of the same reads
06_chimera_localisation_by_tier.csv  median localisation error and chance, per tier, score and seed
export_log.json                    row counts written by the export
