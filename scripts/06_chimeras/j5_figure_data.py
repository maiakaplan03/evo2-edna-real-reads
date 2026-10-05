#!/usr/bin/env python3
"""j5 - per-read localisation errors for ALL channels at C2-T5/T6, for Figure 3."""
import numpy as np, pandas as pd, time
T0=time.time()
exec(open('j1_matched_and_localisation.py').read().split(
     "# ---------------------------------------------------------------- run")[0])
rng = np.random.default_rng(20260921)
eidx = pd.read_csv(EIDX); ec=[c for c in eidx.columns if 'id' in c.lower()][0]
er=[c for c in eidx.columns if c!=ec][0]; erow=dict(zip(eidx[ec],eidx[er]))

negcl = rng.choice(HC['C2_chimera_T6'], NNEG, replace=False)
free = np.array([x for x in ALL_CL if x not in set(negcl.tolist())])
base_ids = [cl2id[x] for x in rng.choice(free, NBASE, replace=False)]
MU,SD = {},{}
Bp = probs_channels(_read(PROBS,[prow[i] for i in base_ids],LP,CP), base_ids)
for k,v in Bp.items(): MU[k]=np.sort(np.asarray(v,np.float64),0); SD[k]=None
Be = emb_channel(_read(PROJ,[erow[i] for i in base_ids],LE,DE))
for k,v in Be.items(): MU[k]=np.sort(np.asarray(v,np.float64),0); SD[k]=None
del Bp,Be

out=[]
for c in ['C2_chimera_T5','C2_chimera_T6']:
    ids=[s for s in rng.choice(POOL[c],1500,replace=False) if s in BP]
    ch = probs_channels(_read(PROBS,[prow[i] for i in ids],LP,CP), ids)
    ch.update(emb_channel(_read(PROJ,[erow[i] for i in ids],LE,DE)))
    bp=np.array([BP[s] for s in ids],float)
    for k in ['logodds','pmax_pobs','entropy','emb64','probs4']:
        sc,est=cusum(ch[k],MU[k],SD[k],OFFSET[k])
        for j,s in enumerate(ids):
            out.append({'cohort':c,'channel':k,'id':s,
                        'err':abs(float(est[j])-bp[j]),'est':float(est[j]),'bp':float(bp[j]),'score':float(sc[j])})
    print(f'  {c} done ({time.time()-T0:.0f}s)',flush=True)
pd.DataFrame(out).to_csv('j5_per_read_errors.csv',index=False)
print(f'wrote j5_per_read_errors.csv  ({len(out)} rows)  total {time.time()-T0:.0f}s')
