#!/usr/bin/env bash
# e2_cluster_and_midori.sh -- s37, 2026-09-29.  RUN ON THE MAC:  caffeinate bash e2_cluster_and_midori.sh
# Two independent vsearch jobs (needs vsearch, already installed in s26). Expected ~5-10 min in total.
#  (a) CLUSTER all reads at 97 % identity. WHY: reads that differ by one or two bases (often one sequence
#      plus sequencing errors) must sit in the SAME half of the train/test split, or the test half contains
#      near-copies of training reads and every trained score (embedding, profile, k-mer) looks better
#      than it is. e4 reports BOTH splits (random and by cluster) so the size of any leakage is visible.
#  (b) SEARCH full MIDORI for the reads never searched in s26 (STOP, evident off-targets, PROFNEG), with
#      the EXACT s26 parameters, so framing B (matched identity to MIDORI) can include them too.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; EVO="$(cd "$HERE/../../.." && pwd)"
DB="$EVO/Secuencias_COI_tax_IDs.fasta"
command -v vsearch >/dev/null || { echo "vsearch not found (brew install vsearch)"; exit 1; }
[ -f "$HERE/all_reads.fasta" ] || { echo "run e1_build_table.py first"; exit 1; }
LOG="$HERE/e2.log"; : > "$LOG"
echo "vsearch: $(vsearch --version 2>&1 | grep -m1 vsearch)" | tee -a "$LOG"
T0=$(date +%s)
echo "== (a) clustering $(grep -c '>' "$HERE/all_reads.fasta") reads at 97 %" | tee -a "$LOG"
vsearch --cluster_size "$HERE/all_reads.fasta" --id 0.97 --strand plus --threads 0 \
        --uc "$HERE/clusters_97.uc" 2>&1 | tail -4 | tee -a "$LOG"
echo "   clusters: $(grep -c '^C' "$HERE/clusters_97.uc")   ($(( $(date +%s) - T0 )) s)" | tee -a "$LOG"
T1=$(date +%s)
echo "== (b) full MIDORI for $(grep -c '>' "$HERE/midori_missing.fasta") reads (s26 parameters)" | tee -a "$LOG"
vsearch --usearch_global "$HERE/midori_missing.fasta" --db "$DB" --id 0.60 --strand plus \
        --maxaccepts 20 --maxrejects 200 --threads 0 --output_no_hits \
        --userout "$HERE/hits_midori_missing.tsv" --userfields query+target+id+alnlen+mism+opens+ql+tl+qcov+tcov \
        2>&1 | tail -3 | tee -a "$LOG"
echo "   hit lines: $(wc -l < "$HERE/hits_midori_missing.tsv")   ($(( $(date +%s) - T1 )) s)" | tee -a "$LOG"
echo "== e2 DONE (log in e2.log)." | tee -a "$LOG"
