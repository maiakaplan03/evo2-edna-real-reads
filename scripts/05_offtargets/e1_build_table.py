#!/usr/bin/env python3
"""
e1_build_table.py -- s37, 2026-09-29.  RUN ON THE MAC:  caffeinate python3 e1_build_table.py   (~1-2 min)

Builds ONE row per read for the off-target analysis and checks every join against the sequence itself.
WHY key by sequence: the s28 manifest has duplicate source uids, so an id-only join can silently attach the
wrong numbers to a read (feedback_data_hygiene #3). Every id join below is re-checked on the sequence.

Output (this folder):
  reads_table.tsv       one row per labelled read: group (T5 and T10), library, labels, MIDORI identity,
                        profile score, GC, s29 perplexity, and the row of that read in each s29 file
  all_reads.fasta       every read in the table, for the 97 % clustering (e2)
  midori_missing.fasta  reads never searched against full MIDORI (STOP, evident off-targets, PROFNEG) (e2)
  e1_checks.json        every count and check printed below
GROUPS (what each read is compared as):
  EUK         eukaryotic COI that passed every COI check (the reference class)
  HID_prok / HID_grey / HID_notCOI   passed every check, labeller says not eukaryotic COI (HIDDEN)
  STOP        pseudogene-like: failed ONLY the stop-codon check (Brazil)
  EV_off401 / EV_off404 / EV_off368  EVIDENT off-targets: flagged by the length filter
  PROFNEG     flagged by the profile filter (scored in run 1; enters only after B1-B4)
  other       none of the above (reported, never compared)
"""
import json, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; NN = HERE.parents[1]; EVO = NN.parent
LAB  = NN/"analysis_s35/labels/reads_labels_2026-09-27.tsv"
FA28 = NN/"analysis_s28/scoring_set/real_scoring_set_2026-09-17.fasta"
FANR = NN/"analysis_s36/launch/UPLOAD/new_run_2026-09-27.fasta"
PPL  = NN/"analysis_s28/scoring_set/perplexity_final_2026-09-19.csv"
EIDX = NN/"analysis_s29/real_run/Tier 3/emb_mean_index.csv"
PIDX = EVO/"tesis_run/evo2_results_real/real_scoring_set_2026-09-17/probs_acgt_index.csv"
UQ   = NN/"analysis_s26/vsearch_midori/queries_profile_ge0_UNIQUE.fasta"
ASG  = NN/"analysis_s26/vsearch_midori/fullmidori_assignments.tsv"
chk = {}
def say(k, v): chk[k] = v; print(f"  {k}: {v}", flush=True)
def fasta(p):
    ids, seqs = [], []
    for l in open(p):
        l = l.strip()
        if l.startswith(">"): ids.append(l[1:].split(";")[0].split()[0]); seqs.append([])
        elif l: seqs[-1].append(l)
    return dict(zip(ids, ("".join(s) for s in seqs)))

d = pd.read_csv(LAB, sep="\t")
say("labelled_reads", len(d)); assert d.read_id.is_unique and d.seq.is_unique
# ---- 1. every read's id must point at the SAME sequence in the fasta that was scored
f28, fnr = fasta(FA28), fasta(FANR)
s28 = d.scored_in == "s28_H200_run"; nr = d.scored_in == "new_run_2026-09-27"
say("scored_s28", int(s28.sum())); say("scored_new_run", int(nr.sum()))
ok28 = d[s28].apply(lambda r: f28.get(r.read_id) == r.seq, axis=1)
oknr = d[nr].apply(lambda r: fnr.get(r.read_id) == r.seq, axis=1)
say("id_to_sequence_match_s28", f"{int(ok28.sum())}/{int(s28.sum())}")
say("id_to_sequence_match_new_run", f"{int(oknr.sum())}/{int(nr.sum())}")
assert ok28.all() and oknr.all(), "an id points at a different sequence -- STOP"
# ---- 2. rows in the s29 files (id-keyed; the ids were just verified against the scored sequence)
eidx = pd.read_csv(EIDX); pidx = pd.read_csv(PIDX); ppl = pd.read_csv(PPL)
assert eidx.read_id.is_unique and pidx.read_id.is_unique and ppl.Query_ID.is_unique
d["emb_row"]   = d.read_id.map(dict(zip(eidx.read_id, eidx.row))).where(s28)
d["probs_row"] = d.read_id.map(dict(zip(pidx.read_id, pidx.row))).where(s28)
d["ppl_s29"]   = d.read_id.map(dict(zip(ppl.Query_ID, ppl.Perplexity))).where(s28)
say("s28_reads_missing_emb_row", int(d[s28].emb_row.isna().sum()))
say("s28_reads_missing_probs_row", int(d[s28].probs_row.isna().sum()))
say("s28_reads_missing_perplexity", int(d[s28].ppl_s29.isna().sum()))
assert d[s28][["emb_row", "probs_row", "ppl_s29"]].notna().all().all()
# ---- 3. identity to full MIDORI (s26 vsearch), joined by SEQUENCE through the UNIQUE query fasta
uq = fasta(UQ); seq2uid = {s: u for u, s in uq.items()}
asg = pd.read_csv(ASG, sep="\t", usecols=["uid", "best_id", "p", "status"])
d["midori_uid"] = d.seq.map(seq2uid)
d = d.merge(asg.rename(columns={"best_id": "midori_id", "p": "midori_phylum", "status": "midori_status"}),
            left_on="midori_uid", right_on="uid", how="left").drop(columns="uid")
d["midori_searched"] = d.midori_uid.notna()
say("reads_with_midori_search_s26", int(d.midori_searched.sum()))
# ---- 4. library, GC, groups
d["lib"] = np.select([d.classes.str.contains("brazil") & ~d.classes.str.contains("abyssal"),
                      d.classes.str.contains("abyssal") & ~d.classes.str.contains("brazil")],
                     ["brazil", "abyssal"], "both")
d["gc"] = d.seq.str.count("[GC]") / d.seq.str.len()
EDNA = d.classes.isin(["edna_brazil", "edna_abyssal"])
def groups(label, hidden):
    g = pd.Series("other", index=d.index)
    g[EDNA & d.passed_all_coi_checks & (label == "euk")] = "EUK"
    for L in ("prok", "grey", "notCOI"):
        g[hidden & (label == L)] = "HID_" + L
    g[d.stop] = "STOP"
    for c in ("abyssal_off401", "abyssal_off404", "abyssal_off368"):
        g[d.classes == c] = "EV_" + c.split("_")[1]
    g[d.profneg] = "PROFNEG"
    return g
d["group_T5"] = groups(d.label, d.hidden); d["group_T10"] = groups(d.label_T10, d.hidden_T10)
tab = pd.crosstab([d.lib], d.group_T5); print(tab.to_string())
chk["groups_T5_by_lib"] = tab.to_dict(); chk["groups_T10_by_lib"] = pd.crosstab(d.lib, d.group_T10).to_dict()
say("hidden_that_are_not_edna_passed", int((d.hidden & ~(EDNA & d.passed_all_coi_checks)).sum()))
# ---- 5. write
cols = ["read_id", "scored_in", "classes", "lib", "group_T5", "group_T10", "label", "label_T10", "hidden",
        "hidden_T10", "passed_all_coi_checks", "stop", "profneg", "orphan_hit", "margin", "prof_score0",
        "shift_best_id", "shift_gain", "n_TGA", "total_count", "midori_searched", "midori_id", "midori_phylum",
        "midori_status", "gc", "ppl_s29", "emb_row", "probs_row", "seq"]
d[cols].to_csv(HERE/"reads_table.tsv", sep="\t", index=False)
with open(HERE/"all_reads.fasta", "w") as fh:
    for r in d.itertuples(): fh.write(f">{r.read_id}\n{r.seq}\n")
miss = d[~d.midori_searched]
with open(HERE/"midori_missing.fasta", "w") as fh:
    for r in miss.itertuples(): fh.write(f">{r.read_id}\n{r.seq}\n")
say("midori_missing_reads", len(miss)); say("midori_missing_by_group", miss.group_T5.value_counts().to_dict())
json.dump(chk, open(HERE/"e1_checks.json", "w"), indent=1, default=str)
print("\n== e1 DONE. Next: caffeinate bash e2_cluster_and_midori.sh")
