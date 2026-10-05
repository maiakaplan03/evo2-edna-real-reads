#!/usr/bin/env python3
"""
w9 - two questions Maia asked on 2026-09-25 (s33), about the embedding scores behind Fig 1/Fig 2.

Q1  Does the 128-d PCA (or the choice to centre, or NOT to standardise) change the result?
    Same comparisons, same caps, same seeds as w6_proper_nulls.py; only the geometry changes:
      A  pca128      : u = L2( W(x-mu) )              <- what the figures used (must reproduce w6)
      B  full_ctr    : u = L2( x-mu )                 all 4096 dims, centred, no PCA
      C  full_std    : u = L2( (x-m)/sd )             all 4096 dims, each dim standardised
      D  pca128_wht  : u = L2( W(x-mu) / sd_pc )      128 PCs, each PC standardised (whitening)
      E  raw         : u = L2( x )                    all 4096 dims, NOT centred
    mu, W = the fixed mapsuite PCA (label-free). m, sd, sd_pc = from 20,000 random reads of the whole
    scoring set (all classes, no labels). The 'x-mu' sample and the sd sample differ; that is fine
    because neither sees a label.
Q2  Error margins. The old plotted point = median over 15 random fold splits; that spread only shows
    fold-assignment noise. Here: 200 bootstrap resamples of READS within each group (folds assigned
    to the ORIGINAL reads first, so a duplicated read never sits in both folds -> no leakage).
    CI = 2.5-97.5 percentiles. Also the null for Fig 1 (>=99 split against itself), which w6 lacked.
Usage: python3 w9_dims_and_ci.py NBOOT [VARIANT,...]  (one variant per call fits the 3-min shell limit)
Writes w9_results[_VARIANT].csv (one row per variant x comparison) and w9_cosines.csv.
"""
import numpy as np, pandas as pd, time, sys
T0=time.time()
DIM,ROW=4096,8192; EMB='real_run/Tier 3/emb_mean.f16'
Z=np.load('mapsuite_sample.npz',allow_pickle=True); comp=Z['pca_components'].astype(np.float32); mu=Z['pca_mean'].astype(np.float32)
NBOOT=int(sys.argv[1]) if len(sys.argv)>1 else 200
ONLY=sys.argv[2].split(',') if len(sys.argv)>2 else None
TAG='_'+'_'.join(ONLY) if ONLY else ''
def log(*a): print(f"[{time.time()-T0:6.0f}s]",*a,flush=True)
def rank(a):
    n=a.size; o=np.argsort(a,kind='mergesort'); r=np.empty(n); r[o]=np.arange(1,n+1)
    i=0
    while i<n:
        j=i
        while j+1<n and a[o[j+1]]==a[o[i]]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    return r
def auroc(p,n):
    p=np.asarray(p,float); n=np.asarray(n,float); p=p[np.isfinite(p)]; n=n[np.isfinite(n)]
    if p.size<5 or n.size<5: return float('nan')
    a=np.concatenate([p,n]); r=rank(a); return (r[:p.size].sum()-p.size*(p.size+1)/2)/(p.size*n.size)
# ---- identical data assembly to w6
man=pd.read_csv('../analysis_s28/scoring_set/real_scoring_set_2026-09-17_manifest.tsv',sep='\t',usecols=['read_id','classes','source_ids'])
asg=pd.read_csv('../analysis_s26/vsearch_midori/fullmidori_assignments.tsv',sep='\t',usecols=['uid','best_id','p','g','s'])
man['uid']=man.source_ids.astype(str).str.split(';').str[0]
d=man.merge(asg,on='uid',how='left')
ridx=pd.read_csv('real_run/Tier 3/emb_mean_index.csv'); rowof=dict(zip(ridx.read_id,ridx.row))
d['row']=d.read_id.map(rowof); d=d[d.row.notna()].copy(); d['row']=d.row.astype(int)
real=d[d.classes.isin(['edna_brazil','edna_abyssal'])&d.best_id.notna()].copy()
FINE=[(99,101,'>=99'),(98,99,'98-99'),(97,98,'97-98'),(95,97,'95-97'),(92,95,'92-95'),
      (88,92,'88-92'),(84,88,'84-88'),(80,84,'80-84'),(70,80,'70-80'),(0,70,'<70')]
def band(x):
    for lo,hi,l in FINE:
        if lo<=x<hi: return l
real['band']=real.best_id.map(band)
def raw(rows):
    rows=np.asarray(rows); o=np.argsort(rows); out=np.empty((len(rows),DIM),np.float32)
    with open(EMB,'rb') as f:
        for k in o: f.seek(int(rows[k])*ROW); out[k]=np.frombuffer(f.read(ROW),dtype=np.float16)
    return out
def take(s,cap=2000): return s if len(s)<=cap else s.sample(cap,random_state=20260920)
# ---- label-free normalisation statistics from 20,000 random reads of the whole scoring set
rs=np.random.default_rng(20260925).choice(len(ridx),20000,replace=False)
S=raw(ridx.row.values[rs]); m_all=S.mean(0); sd_all=S.std(0); sd_all[sd_all<1e-6]=1e-6
Ppc=(S-mu)@comp.T; sd_pc=Ppc.std(0); del S,Ppc
log("stats sample done; dims with sd<1e-3:",int((sd_all<1e-3).sum()))
def L2(M): return M/np.linalg.norm(M,axis=1,keepdims=True)
VAR={'A_pca128':   lambda X: L2((X-mu)@comp.T),
     'B_full_ctr': lambda X: L2(X-mu),
     'C_full_std': lambda X: L2((X-m_all)/sd_all),
     'D_pca128_wht':lambda X: L2(((X-mu)@comp.T)/sd_pc),
     'E_raw':      lambda X: L2(X)}
# ---- scoring (identical to w6.ho, plus a version with fixed folds for the bootstrap)
def ho_f(A,B,fa,fb):
    sa=np.empty(len(A)); sb=np.empty(len(B))
    for f in (0,1):
        ca=A[fa!=f].mean(0); ca/=np.linalg.norm(ca); cb=B[fb!=f].mean(0); cb/=np.linalg.norm(cb)
        sa[fa==f]=A[fa==f]@cb-A[fa==f]@ca; sb[fb==f]=B[fb==f]@cb-B[fb==f]@ca
    return auroc(sb,sa)
def ho(A,B,r): return ho_f(A,B,r.integers(0,2,len(A)),r.integers(0,2,len(B)))
def boot(A,B,seed0):
    out=[]
    for b in range(NBOOT):
        r=np.random.default_rng(seed0+b)
        fa0=r.integers(0,2,len(A)); fb0=r.integers(0,2,len(B))
        ia=r.integers(0,len(A),len(A)); ib=r.integers(0,len(B),len(B))
        out.append(ho_f(A[ia],B[ib],fa0[ia],fb0[ib]))
    return np.percentile(out,[2.5,97.5])
def null(A,seed0,n=15):
    nl=[]
    for i in range(n):
        r=np.random.default_rng(seed0+i); pm=r.permutation(len(A)); h=len(A)//2
        nl.append(ho(A[pm[:h]],A[pm[h:2*h]],r))
    return np.percentile(nl,[2.5,97.5]), float(np.median(nl))
# ---- assemble the raw matrices once
br=real[(real.classes=='edna_brazil')&(real.p=='Ascomycota')]
TASKS=[]  # (fig, label, rawA, rawB, point_seed0, argument order as in w6)
R=br[br.band=='>=99']; XR=raw(take(R).row.values)
for lo_,hi_,lab in FINE:
    if lab=='>=99': continue
    s=br[br.band==lab]
    if len(s)<50: continue
    TASKS.append(('fig1',lab,raw(take(s).row.values),XR,600))
GEN={}
for TEST in ['98-99','97-98','95-97']:
    for g,sub in br.groupby('g'):
        a=sub[sub.band=='>=99']; b=sub[sub.band==TEST]
        if min(len(a),len(b))<40: continue
        if g not in GEN: GEN[g]=raw(take(a,600).row.values)
        TASKS.append(('fig2',f'{g}|{TEST}',GEN[g],raw(take(b,600).row.values),700))
log("raw matrices loaded:",len(TASKS),"comparisons")
rows=[]; cos=[]
for vn,fn in VAR.items():
    if ONLY and vn not in ONLY: continue
    UR=fn(XR); UG={g:fn(X) for g,X in GEN.items()}
    # mean cosine between two random >=99 reads (the 'how spread are the arrows' number)
    r=np.random.default_rng(1); i=r.integers(0,len(UR),5000); j=r.integers(0,len(UR),5000); k=i!=j
    cos.append(dict(variant=vn,mean_cos_random_pairs=float(np.mean(np.sum(UR[i[k]]*UR[j[k]],1)))))
    (nlo,nhi),nmed=null(UR,900)
    rows.append(dict(variant=vn,fig='fig1',comparison='null >=99 vs itself',auroc=nmed,ci_lo=nlo,ci_hi=nhi,null_lo=nlo,null_hi=nhi))
    gnull={g:null(U,800) for g,U in UG.items()}
    for fig,lab,XA,XB,s0 in TASKS:
        if fig=='fig1':  A=fn(XA); B=UR; g=None          # w6 order: ho(band, ref)
        else:            g=lab.split('|')[0]; A=UG[g]; B=fn(XB)   # w6 order: ho(ref, band)
        pt=float(np.median([ho(A,B,np.random.default_rng(s0+i)) for i in range(15)]))
        lo,hi=boot(A,B,50000)
        nl=gnull[g][0] if g else (nlo,nhi)
        rows.append(dict(variant=vn,fig=fig,comparison=lab,auroc=pt,ci_lo=lo,ci_hi=hi,null_lo=nl[0],null_hi=nl[1],
                         n_a=len(A),n_b=len(B)))
    log(vn,"done")
    pd.DataFrame(rows).to_csv(f'w9_results{TAG}.csv',index=False); pd.DataFrame(cos).to_csv(f'w9_cosines{TAG}.csv',index=False)
log("finished")
