#!/usr/bin/env bash
# ONE command. Runs uchime_ref against the real reference set, then scores both
# methods on the identical reads. ~1 minute.
set -uo pipefail
cd "$(dirname "$0")"
REF=../reference_build/references_N1565.fasta
[ -f "$REF" ] || { echo "reference set not found at $REF"; exit 1; }
echo ">>> uchime_ref  (the baseline gets the actual reference database)"
vsearch --uchime_ref j3_benchmark.fasta --db "$REF" \
        --uchimeout j3_uchime_ref.tsv --nonchimeras /dev/null --chimeras /dev/null \
        --threads 4 2>&1 | tail -6
echo
echo ">>> uchime_denovo  (reported as a second baseline, handicapped - see j3_build_benchmark.py)"
vsearch --uchime_denovo j3_benchmark.fasta \
        --uchimeout j3_uchime_denovo.tsv --nonchimeras /dev/null --chimeras /dev/null 2>&1 | tail -6
echo
python3 j4_compare.py
