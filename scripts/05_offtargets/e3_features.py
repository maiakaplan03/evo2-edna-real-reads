#!/usr/bin/env python3
"""
e3_features.py -- s37, 2026-09-29.  RUN ON THE MAC:  caffeinate python3 e3_features.py   (~5-10 min)

Turns the s29 forward outputs into one feature matrix per Evo 2 quantity, in the row order of reads_table.tsv
(s28-scored reads only; PROFNEG comes from run 1 and is added after B1-B4). Every matrix is checked against
an INDEPENDENT reconstruction before it is written (feedback_data_hygiene: outputs are verified against
something computed another way, never by the code that produced them).

EVO 2 QUANTITIES (what each one is)
  embA    the mean-pooled layer-28 embedding (4096 numbers per read), centred, projected on the 128 fixed PCs,
          length-normalised                           = geometry A of the novelty ladder (s29/s34)
  embD    the same 128 PCs, each divided by its spread over 20,000 random reads (whitening) = geometry D
  nll     the per-position profile: at each of 311 positions, -log p(the base that is actually there);
          high = Evo 2 did not expect that base
  acgt    the full per-position A/C/G/T distribution (311 x 4): WHICH base Evo 2 expected, not only how
          surprised it was
  codon3  mean NLL at window positions 1,2,3 (mod 3). For reads that passed the checks (register 0) these
          are codon positions 1,2,3, so the third column is where codon usage lives
  (perplexity is already in reads_table.tsv as ppl_s29: one number per read = exp(mean NLL))
COMPETITOR (not Evo 2)
  kmer4   counts of the 256 possible 4-letter words in the read (composition; a k-mer classifier's input)
CHECKS (the script stops if any fails)
  C1 nll from the probs file == per_base_nll stored by the scorer in Tier 1, on 2,000 random reads, within one
     bfloat16 rounding step of the stored value (the scorer rounds per_base_nll to bf16)          C2 exp(mean nll) == s29 perplexity on every read (rel. diff < 1e-2)
  C3 embA of 300 reads == the w9 formula recomputed from scratch in float64 (max |diff| < 1e-3)
  C4 no NaN / Inf anywhere      C5 the 4-mer counts of every read sum to 309
"""
import json, time, numpy as np, pandas as pd
from pathlib import Path
T0 = time.time()
import os
SMOKE = int(os.environ.get("SMOKE", "0"))      # SMOKE=3000 -> test on 3,000 reads into features_smoke/ (test only)
HERE = Path(__file__).resolve().parent; NN = HERE.parents[1]; EVO = NN.parent
OUT = HERE/("features_smoke" if SMOKE else "features"); OUT.mkdir(exist_ok=True)
EMB  = NN/"analysis_s29/real_run/Tier 3/emb_mean.f16"
PRB  = EVO/"tesis_run/evo2_results_real/real_scoring_set_2026-09-17/probs_acgt.f16"
PB1  = NN/"analysis_s29/real_run/Tier 1/per_base.jsonl"
MAP  = NN/"analysis_s29/mapsuite_sample.npz"
NTOT, D, L = 200898, 4096, 311
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
chk = {}
t = pd.read_csv(HERE/"reads_table.tsv", sep="\t", usecols=["read_id", "scored_in", "emb_row", "probs_row", "ppl_s29", "seq"])
t = t[t.scored_in == "s28_H200_run"].reset_index(drop=True)
if SMOKE: t = t.sample(SMOKE, random_state=0).reset_index(drop=True)
N = len(t)
t[["read_id"]].to_csv(OUT/"rows.tsv", sep="\t", index=False); log("reads:", N)
er, pr = t.emb_row.astype(int).values, t.probs_row.astype(int).values
# ---------------- embeddings: geometry A and D exactly as w9_dims_and_ci.py
Z = np.load(MAP, allow_pickle=True); comp = Z["pca_components"].astype(np.float32); mu = Z["pca_mean"].astype(np.float32)
E = np.memmap(EMB, dtype=np.float16, mode="r", shape=(NTOT, D))
rs = np.random.default_rng(20260925).choice(NTOT, 20000, replace=False)       # same 20k reads and seed as w9
S = np.asarray(E[np.sort(rs)], dtype=np.float32); sd_pc = ((S - mu) @ comp.T).std(0); del S
def L2(M): return M / np.linalg.norm(M, axis=1, keepdims=True)
embA = np.empty((N, 128), np.float32); embD = np.empty((N, 128), np.float32)
o = np.argsort(er); CH = 8192
for s in range(0, N, CH):
    idx = o[s:s+CH]; X = np.asarray(E[er[idx]], dtype=np.float32); P = (X - mu) @ comp.T
    embA[idx] = L2(P); embD[idx] = L2(P / sd_pc)
log("embeddings done")
r3 = np.random.default_rng(1).choice(N, 300, replace=False)
X64 = np.asarray(E[er[r3]], dtype=np.float64); P64 = (X64 - mu.astype(np.float64)) @ comp.T.astype(np.float64)
ref = P64 / np.linalg.norm(P64, axis=1, keepdims=True)
chk["C3_embA_max_absdiff"] = float(np.abs(ref - embA[r3]).max()); assert chk["C3_embA_max_absdiff"] < 1e-3
# ---------------- probs -> acgt, nll, codon3
Pm = np.memmap(PRB, dtype=np.float16, mode="r", shape=(NTOT, L, 5))
code = {"A": 0, "C": 1, "G": 2, "T": 3}
B = np.array([[code[c] for c in s] for s in t.seq.values], dtype=np.uint8)          # N x 312, all ACGT (checked)
acgt = np.empty((N, L, 4), np.float16); nll = np.empty((N, L), np.float32)
o = np.argsort(pr)
for s in range(0, N, CH):
    idx = o[s:s+CH]; X = np.asarray(Pm[pr[idx]])                                 # chunk x 311 x 5
    acgt[idx] = X[:, :, :4]
    pobs = np.take_along_axis(X[:, :, :4].astype(np.float32), B[idx, 1:, None].astype(np.int64), axis=2)[:, :, 0]
    nll[idx] = -np.log(np.clip(pobs, 1e-8, None))                                  # row i predicts base i+1
log("probs done")
# C1: against the scorer's own per_base_nll (Tier 1), which was written by a different code path
want = set(t.read_id.values[np.random.default_rng(2).choice(N, 2000, replace=False)]); pos = dict(zip(t.read_id, range(N)))
diffs = []
with open(PB1) as fh:
    for line in fh:
        rid = line[8:line.index('"', 8)]
        if rid in want:
            j = json.loads(line); x = np.array(j["per_base_nll"], np.float64); dd = np.abs(x - nll[pos[rid]])
            # the scorer stored per_base_nll rounded to bfloat16 (8-bit mantissa: 1.3125, 1.273438, ...), so the
            # allowed gap is one bf16 step at that magnitude, 2^(floor(log2|x|) - 7), plus float16 rounding of p
            step = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(x), 1e-6))) - 7)
            diffs.append(float((dd / (step + 2e-3)).max()))
chk["C1_n_compared"] = len(diffs); chk["C1_worst_diff_in_bf16_steps"] = float(max(diffs))
assert len(diffs) == 2000 and max(diffs) <= 1.0, chk
rel = np.abs(np.exp(nll.mean(1)) / t.ppl_s29.values - 1)
chk["C2_ppl_reldiff_max"] = float(rel.max()); chk["C2_ppl_reldiff_median"] = float(np.median(rel)); assert rel.max() < 1e-2, chk
posn = np.arange(1, L + 1)                                                          # the base each row predicts
codon3 = np.stack([nll[:, posn % 3 == k].mean(1) for k in (0, 1, 2)], 1).astype(np.float32)
# ---------------- k-mer competitor
k4 = (B[:, :-3].astype(np.int64) * 64 + B[:, 1:-2] * 16 + B[:, 2:-1] * 4 + B[:, 3:])
kmer = np.bincount((np.arange(N)[:, None] * 256 + k4).ravel(), minlength=N * 256).reshape(N, 256).astype(np.uint16)
chk["C5_kmer_rowsum_all_309"] = bool((kmer.sum(1) == 309).all()); assert chk["C5_kmer_rowsum_all_309"]
for name, M in [("embA", embA), ("embD", embD), ("nll", nll), ("acgt", acgt), ("codon3", codon3)]:
    ok = bool(np.isfinite(M).all()); chk[f"C4_finite_{name}"] = ok; assert ok, name
    np.save(OUT/f"{name}.npy", M)
np.save(OUT/"kmer4.npy", kmer)
json.dump(chk, open(OUT/"e3_checks.json", "w"), indent=1); log("checks:", chk)
log("== e3 DONE -- all checks passed. Next: e4_compare.py")
