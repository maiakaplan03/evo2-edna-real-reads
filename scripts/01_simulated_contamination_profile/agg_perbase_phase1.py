#!/usr/bin/env python3
"""Aggregate phase1 per-position NLL into a small npz: per-cohort mean + quartile bands.
Deterministic hash-based subsample (target ~2500 reads/cohort) so the figure is reproducible."""
import json, numpy as np, pandas as pd, hashlib, sys, time
T0=time.time()
FRAME='../analysis_s26/ANALYSIS_FRAME_2026-09-16.csv'
PB='../../tesis_run/evo2_results_h100/phase1_all/Tier 1/per_base.jsonl'
TARGET=2500
fr=pd.read_csv(FRAME, usecols=['seq_id','cohort'])
coh=dict(zip(fr.seq_id, fr.cohort))
sizes=fr.cohort.value_counts().to_dict()
keep_p={c: min(1.0, TARGET/n) for c,n in sizes.items()}
def keep(rid, c):
    p=keep_p[c]
    if p>=1.0: return True
    h=int.from_bytes(hashlib.blake2b(rid.encode(),digest_size=4).digest(),'big')
    return (h % 1000000) < p*1000000
buf={}
n_lines=0; n_kept=0
with open(PB) as f:
    for line in f:
        n_lines+=1
        i=line.find('"',7); j=line.find('"',i+1)   # id value between quotes after {"id": "
        rid=line[i+1:j]
        c=coh.get(rid)
        if c is None or not keep(rid,c): continue
        d=json.loads(line)
        v=d.get('per_base_nll')
        if not v: continue
        buf.setdefault(c,[]).append(np.asarray(v,dtype=np.float32))
        n_kept+=1
print(f"lines={n_lines}  kept={n_kept}  t={time.time()-T0:.1f}s")
out={}
for c,rows in buf.items():
    A=np.vstack(rows)                      # reads x 311, NLL in nats
    # store in NLL space; the figure exponentiates -> exp(mean NLL) = geometric-mean perplexity,
    # which is what a per-base perplexity is. Averaging exp(NLL) instead lets a few huge-NLL
    # reads dominate every position (that bug produced an unreadable oscillating curve).
    out[f"{c}__mean"]=A.mean(0)
    out[f"{c}__q25"]=np.percentile(A,25,axis=0)
    out[f"{c}__q75"]=np.percentile(A,75,axis=0)
    out[f"{c}__median"]=np.median(A,axis=0)
    out[f"{c}__n"]=np.array([A.shape[0]])
np.savez_compressed('perbase_phase1_profiles.npz', **out)
print("cohorts:",len(buf)," wrote perbase_phase1_profiles.npz  total t=%.1fs"%(time.time()-T0))
