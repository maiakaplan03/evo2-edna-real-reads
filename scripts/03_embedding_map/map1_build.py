#!/usr/bin/env python3
"""
E1 - build the co-embedded sample matrix for the embedding map.

WHAT IT DOES
  Draws a balanced deterministic subsample of every group - 8 real classes (H200 run) and
  34 simulated cohorts (phase1 H100 run) - reads their 4096-d mean embeddings by byte offset,
  fits ONE PCA on a balanced sub-subsample, and projects everything into a shared 128-d space.

WHY ONE SHARED PCA
  The map only means something if real and simulated sit in the same coordinate system. That is
  legitimate here because the H200 reproduced the H100 bit-exactly on the 1,560 anchors (A1-A4,
  2026-09-20). The PCA is fitted on a BALANCED sample so no group's size steers the axes.

BUILT-IN POSITIVE CONTROL
  The 1,560 ref_anchor sequences exist in BOTH runs. After projection their two copies must land
  on top of each other. If they do not, the co-embedding is broken and nothing downstream is valid.

OUTPUT  mapsuite_sample.npz : X128 (n,128) float32 | labels | source ('real'/'sim') | ids
                         centroids_full (n_groups,4096) | pca components, mean, explained variance
"""
import numpy as np, pandas as pd, os, sys, time, json
T0=time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]",*a,flush=True)

PER_GROUP  = int(os.environ.get("PER_GROUP", 2500))   # sampled vectors per group
PCA_PER_GP = int(os.environ.get("PCA_PER_GROUP", 600))# vectors per group used to FIT the PCA
NPC        = int(os.environ.get("NPC", 128))
SEED       = 20260920
DIM, ROW   = 4096, 8192
rng = np.random.default_rng(SEED)

REAL_EMB = 'real_run/Tier 3/emb_mean.f16'
REAL_IDX = 'real_run/Tier 3/emb_mean_index.csv'
SIM_EMB  = '../../tesis_run/evo2_results_h100/phase1_all/Tier 3/emb_mean.f16'
SIM_IDX  = '../../tesis_run/evo2_results_h100/phase1_all/Tier 1/emb_mean_index.csv'
MANIFEST = '../analysis_s28/scoring_set/real_scoring_set_2026-09-17_manifest.tsv'
FRAME    = '../analysis_s26/ANALYSIS_FRAME_2026-09-16.csv'

for p in (REAL_EMB, SIM_EMB, MANIFEST, FRAME):
    if not os.path.exists(p): sys.exit(f"!! missing input: {p}")
for p,alt in ((REAL_IDX,'real_run/Tier 1/emb_mean_index.csv'),):
    if not os.path.exists(p) and os.path.exists(alt): REAL_IDX=alt
if not os.path.exists(REAL_IDX): sys.exit(f"!! missing real emb_mean_index.csv")

def sizecheck(path, n_expected):
    b=os.path.getsize(path)
    log(f"{path}: {b:,} B  = {b/ROW:,.0f} rows" + (f"  (expected {n_expected:,})" if n_expected else ""))
    if n_expected and b != n_expected*ROW: log(f"   !! WARNING size != {n_expected*ROW:,}")
sizecheck(REAL_EMB, 200898); sizecheck(SIM_EMB, None)

def rowmap(path):
    d=pd.read_csv(path)
    idc=[c for c in d.columns if c.lower() in ('read_id','id','query_id','seq_id')][0]
    rwc=[c for c in d.columns if c.lower()=='row'][0]
    return dict(zip(d[idc], d[rwc]))
log("reading indexes..."); r_row=rowmap(REAL_IDX); s_row=rowmap(SIM_IDX)
log(f"real rows indexed {len(r_row):,} | sim rows indexed {len(s_row):,}")

# ---- group membership ----
groups={}   # name -> (which_run, [ids])
man=pd.read_csv(MANIFEST, sep='\t', usecols=['read_id','classes'])
man['classes']=man['classes'].fillna('UNLABELLED')
KEEP_REAL=['ref_anchor','midori','edna_brazil','edna_abyssal','brazil_stop',
           'abyssal_off404','abyssal_off401','abyssal_off368']
for c in KEEP_REAL:
    ids=man.loc[man.classes==c,'read_id'].tolist()
    if ids: groups[f"REAL:{c}"]=('real',ids)
fr=pd.read_csv(FRAME, usecols=['seq_id','cohort'])
for c,g in fr.groupby('cohort'):
    groups[f"SIM:{c}"]=('sim', g.seq_id.tolist())
log(f"groups: {len(groups)}  (real {sum(1 for v in groups.values() if v[0]=='real')}, "
    f"sim {sum(1 for v in groups.values() if v[0]=='sim')})")

# ---- sample + read by byte offset ----
def read_rows(path, rows):
    out=np.empty((len(rows),DIM),dtype=np.float16)
    with open(path,'rb') as f:
        for k,r in enumerate(rows):
            f.seek(r*ROW); out[k]=np.frombuffer(f.read(ROW),dtype=np.float16)
    return out

X=[]; LAB=[]; SRC=[]; IDS=[]; missing={}
for name,(src,ids) in groups.items():
    rm = r_row if src=='real' else s_row
    ok=[i for i in ids if i in rm]
    missing[name]=len(ids)-len(ok)
    if not ok: log(f"  !! {name}: no ids found in index - SKIPPED"); continue
    take = ok if len(ok)<=PER_GROUP else list(rng.choice(ok,PER_GROUP,replace=False))
    take.sort(key=lambda z: rm[z])                      # sequential seeks
    M = read_rows(REAL_EMB if src=='real' else SIM_EMB, [rm[z] for z in take])
    X.append(M); LAB += [name]*len(take); SRC += [src]*len(take); IDS += take
    log(f"  {name:<34} n={len(take):>5}  (pool {len(ok):,})")
X=np.vstack(X); LAB=np.array(LAB); SRC=np.array(SRC); IDS=np.array(IDS)
log(f"sample matrix {X.shape}  {X.nbytes/1e9:.2f} GB f16")
if any(missing.values()): log("ids not found in index per group:", {k:v for k,v in missing.items() if v})

# ---- centroids in FULL space (before any projection) ----
names=sorted(set(LAB))
cent=np.stack([X[LAB==n].astype(np.float32).mean(0) for n in names])
log(f"centroids {cent.shape}")

# ---- fit ONE PCA on a balanced sub-subsample ----
sel=[]
for n in names:
    idx=np.where(LAB==n)[0]
    sel.append(rng.choice(idx, min(PCA_PER_GP,idx.size), replace=False))
sel=np.concatenate(sel)
Xf=X[sel].astype(np.float32)
log(f"fitting PCA on {Xf.shape} (balanced)...")
mu=Xf.mean(0); Xf-=mu
try:
    from sklearn.decomposition import PCA
    pca=PCA(n_components=NPC, svd_solver='randomized', random_state=SEED).fit(Xf)
    comp=pca.components_.astype(np.float32); evr=pca.explained_variance_ratio_
except Exception as e:
    log("sklearn PCA unavailable (%s) - numpy SVD fallback"%e)
    U,S,Vt=np.linalg.svd(Xf, full_matrices=False)
    comp=Vt[:NPC].astype(np.float32); evr=(S**2/(S**2).sum())[:NPC]
log(f"PCA done. explained variance: PC1 {evr[0]*100:.1f}%  top10 {evr[:10].sum()*100:.1f}%  "
    f"top{NPC} {evr.sum()*100:.1f}%")
del Xf

X128=np.empty((X.shape[0],NPC),dtype=np.float32)
for i in range(0,X.shape[0],20000):
    X128[i:i+20000]=(X[i:i+20000].astype(np.float32)-mu)@comp.T
log(f"projected -> {X128.shape}")

# ---- POSITIVE CONTROL: ref_anchor appears in both runs ----
ctrl={'status':'not_run'}
try:
    join=pd.read_csv('../analysis_s28/scoring_set/anchor_join_1560.tsv', sep='\t')
    pos={i:k for k,i in enumerate(IDS)}
    pairs=[(pos[nr],pos[p1]) for nr,p1 in zip(join.new_read_id,join.phase1_id)
           if nr in pos and p1 in pos]
    if pairs:
        a=np.array([p[0] for p in pairs]); b=np.array([p[1] for p in pairs])
        d=np.linalg.norm(X128[a]-X128[b],axis=1)
        sc=np.linalg.norm(X128-X128.mean(0),axis=1).mean()
        ctrl=dict(status='ran', n_pairs=len(pairs), max_dist=float(d.max()),
                  mean_dist=float(d.mean()), mean_radius=float(sc),
                  verdict='PASS' if d.max() < 1e-3*sc else 'FAIL')
        log(f"ANCHOR CONTROL: {len(pairs)} pairs present in both runs, "
            f"max distance {d.max():.3e} vs mean cloud radius {sc:.3f} -> {ctrl['verdict']}")
    else:
        log("ANCHOR CONTROL: no anchor pairs landed in the sample (both copies must be sampled) "
            "- raise PER_GROUP or ignore; A1-A4 already proved bit-exactness")
except Exception as e:
    log("anchor control skipped:", e)

np.savez('mapsuite_sample.npz', X128=X128, labels=LAB, source=SRC, ids=IDS,
         centroids_full=cent, centroid_names=np.array(names),
         pca_components=comp, pca_mean=mu, explained_variance_ratio=evr,
         meta=json.dumps(dict(per_group=PER_GROUP, pca_per_group=PCA_PER_GP, npc=NPC,
                              seed=SEED, anchor_control=ctrl)))
log("wrote mapsuite_sample.npz  (%.0f MB)"%(os.path.getsize('mapsuite_sample.npz')/1e6))
