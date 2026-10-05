#!/usr/bin/env python3
"""
00_export_notebook_tables.py -- builds the small tables in data/ that the story notebook reads.

Runs only inside the full analysis tree (it needs the Evo 2 outputs, which are too large for GitHub):
    python3 00_export_notebook_tables.py /path/to/nn_project /path/to/repo/data

Nothing is recomputed here. Every table is a copy, a column subset, or a group summary of a file that
an earlier script in scripts/ wrote. No read sequences and no read identifiers of real reads are exported.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path

NN, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
log = {}

# 1 -- simulated cohorts: per-position NLL summaries (from agg_perbase_phase1.py)
Z = np.load(NN / "analysis_s29/perbase_phase1_profiles.npz")
COH = ["C0_REF", "C1_illumina", "C4_briggs_low", "C4_briggs_high", "C2_chimera_T2", "C2_chimera_T6", "C3_insert"]
keep = {f"{c}__{s}": Z[f"{c}__{s}"] for c in COH for s in ("mean", "q25", "q75", "n")}
np.savez_compressed(OUT / "01_simulated_profiles.npz", **keep); log["01"] = len(keep)

# 2 -- real classes: mean perplexity per sequence + class label only (from the H200 run + manifest)
B = NN / "analysis_s28/scoring_set"
p = pd.read_csv(B / "perplexity_final_2026-09-19.csv")
m = pd.read_csv(B / "real_scoring_set_2026-09-17_manifest.tsv", sep="\t", usecols=["read_id", "classes"])
d = p.merge(m, left_on="Query_ID", right_on="read_id")
d = d.assign(cls=d["classes"].fillna("UNLABELLED"))[["cls", "Perplexity"]].rename(columns={"Perplexity": "perplexity"})
d.to_csv(OUT / "02_real_perplexity.csv.gz", index=False, float_format="%.6f"); log["02"] = len(d)

# 3 -- embedding map: 2-D coordinates + group label only (from map1_build.py / w7_figures.py)
M = np.load(NN / "analysis_s29/mapsuite_sample.npz", allow_pickle=True)
U = np.load(NN / "analysis_s29/w8_umap_coords.npy")
e = pd.DataFrame(dict(group=M["labels"], pc1=M["X128"][:, 0], pc2=M["X128"][:, 1], umap1=U[:, 0], umap2=U[:, 1]))
e.to_csv(OUT / "03_embedding_map.csv.gz", index=False, float_format="%.5f")
log["03"] = len(e); log["03_pc_variance"] = [float(v) for v in M["explained_variance_ratio"][:2]]

# 4 -- novelty ladder: AUROC + bootstrap CI + null per comparison, embedding version D (from w9_dims_and_ci.py)
w = pd.read_csv(NN / "analysis_s29/w9_results_D_pca128_wht.csv")
w.to_csv(OUT / "04_novelty_ladder.csv", index=False); log["04"] = len(w)

# 5a -- real off-targets: per-position NLL summaries per library x group (from e1/e3)
E = NN / "analysis_s37/offtarget_eval"
t = pd.read_csv(E / "reads_table.tsv", sep="\t", usecols=["read_id", "lib", "group_T5"], low_memory=False)
rows = pd.read_csv(E / "features/rows.tsv", sep="\t"); frow = dict(zip(rows.read_id, range(len(rows))))
t["frow"] = t.read_id.map(frow); t = t[t.frow.notna()]; t["frow"] = t.frow.astype(int)
NLL = np.load(E / "features/nll.npy", mmap_mode="r")
GROUPS = {"brazil": ["EUK", "HID_prok", "HID_grey", "HID_notCOI", "STOP"],
          "abyssal": ["EUK", "HID_prok", "HID_grey", "HID_notCOI"]}
out = []
for lib, gs in GROUPS.items():
    for g in gs:
        idx = np.sort(t[(t.lib == lib) & (t.group_T5 == g)].frow.values)
        X = np.asarray(NLL[idx], dtype=np.float64)
        out.append(pd.DataFrame(dict(lib=lib, group=g, n=len(idx), pos=np.arange(1, X.shape[1] + 1),
                                     mean_nll=X.mean(0), q25_nll=np.percentile(X, 25, axis=0),
                                     q75_nll=np.percentile(X, 75, axis=0),
                                     mean_ppl_51_311=float(np.exp(X[:, 50:].mean())))))
pd.concat(out).to_csv(OUT / "05_offtarget_profiles.csv.gz", index=False, float_format="%.6f"); log["05a"] = len(out)

# 5b -- off-target AUROCs (from e4_compare.py, de-duplicated by e4_check.py)
r = pd.concat([pd.read_csv(E / f"results_{f}_clean.tsv", sep="\t") for f in ("hidden", "stop")])
r = r[(r.framing == "A") & (r["T"] == 5) & (r.classifier == "ridge") & (r.split == "cluster")
      & (r.comparison.isin(["HID_prok_vs_EUK", "STOP_vs_EUK"])) & (r.feature != "onehot")]
r = r[["comparison", "lib", "n_pos", "n_neg", "feature", "auroc", "ci_lo", "ci_hi", "null_lo", "null_hi"]]
r.to_csv(OUT / "05_offtarget_auroc.csv", index=False); log["05b"] = len(r)

# 6 -- chimeras, rank standardisation (from run_chain.sh: j1, j2, j4/j7, j5)
C = NN / "analysis_s37/c2_rc/s32_rank"
bm = pd.read_csv(C / "j4_benchmark_scored.csv")[["cohort", "is_chimera", "uchime_ref", "evo2_entropy"]]
bm.to_csv(OUT / "06_chimera_detection.csv", index=False)
per = pd.read_csv(C / "j5_per_read_errors.csv")
per = per[(per.cohort == "C2_chimera_T6") & (per.channel == "logodds")][["err", "est", "bp"]]
per.to_csv(OUT / "06_chimera_localisation_T6.csv", index=False)
wm = pd.read_csv(C / "j2_worked_vs_missed.csv"); wm = wm[wm.cohort == "C2_chimera_T6"][["score", "err"]]
wm.to_csv(OUT / "06_chimera_confidence_T6.csv", index=False)
loc = pd.read_csv(C / "j1_localisation_per_seed.csv")
loc.to_csv(OUT / "06_chimera_localisation_by_tier.csv", index=False)
log["06"] = dict(detection=len(bm), localisation_T6=len(per), confidence_T6=len(wm), by_tier=len(loc))

json.dump(log, open(OUT / "export_log.json", "w"), indent=1); print(json.dumps(log, indent=1))
