#!/usr/bin/env bash
# s37: the s32 figure pipeline re-run with the RANK standardisation (only the standardisation line differs, see
# ADDENDUM_rc_2026-09-29.txt section 8), then every figure drawn standalone + next to its s32 original.
set -e
cd "$(dirname "$0")"
echo "[$(date +%T)] j1 (detection + localisation, all cohorts, 5 channels incl. embedding)"; python3 -u j1_matched_and_localisation.py --with-emb > j1.log 2>&1
echo "[$(date +%T)] j5 (per-read errors T5/T6)";      python3 -u j5_figure_data.py > j5.log 2>&1
echo "[$(date +%T)] j2 (score vs accuracy)";          python3 -u j2_worked_vs_missed.py > j2.log 2>&1
echo "[$(date +%T)] j3 (benchmark reads)";            python3 -u j3_build_benchmark.py > j3.log 2>&1
echo "[$(date +%T)] j4 (join uchime)";                python3 -u j4_compare.py > j4.log 2>&1
echo "[$(date +%T)] j7 (entropy on benchmark)";       python3 -u j7_add_entropy_to_benchmark.py > j7.log 2>&1
python3 - <<'PY'
import pandas as pd
a = pd.read_csv("j3_labels.csv"); b = pd.read_csv("../../../analysis_s32/j3_labels.csv")
same = list(a.id) == list(b.id); print("benchmark reads identical to s32:", same)
assert same, "benchmark reads differ from s32 -> the uchime columns would not belong to these reads. STOP."
PY
echo "[$(date +%T)] figures"; cd .. && python3 -u k6_figures.py && python3 -u k7_rc_figures.py
echo "[$(date +%T)] ALL DONE -> nn_project/analysis_s37/figures/"
