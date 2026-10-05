#!/usr/bin/env python3
"""
j1 - two questions in one pass. CPU only, SIMULATED phase1 data, no GPU, no rescore.
Run from nn_project/analysis_s32/.

=============================================================================
Q1. IS THE C2 NEGATIVE CLASS COMPOSITION-MATCHED?  (it is not)
=============================================================================
C0_REF is 1,550 reads, exactly one per reference cluster. C2 and C5 were built
from only 224 of those clusters. e4/e5/i1 all drew the negative class from ALL
1,550. So every C2/C5 AUROC so far compared 224 hosts' worth of sequence against
a 1,550-host background, while C1 and C3 (which use all 1,550) were matched by
construction.

That is exactly data-hygiene trap 1. And the cohorts with the mismatch are
exactly the cohorts that score BELOW chance:
    C1 0.518 (matched, correct)   C3 0.862 (matched, strong)
    C2-shallow 0.41   C5 0.499    (both UNmatched, both anomalous)

Mechanism: Evo2's surprise is dominated by how familiar a taxon is. A narrower,
more familiar 224-cluster slice gives flatter profiles -> smaller CUSUM -> AUROC
below 0.5, with no junction involved.

PREDICTION, stated before running: with matched negatives the C2-shallow numbers
move UP toward 0.50 and C2-T6 holds or rises. If T6 does not move, the
explanation is wrong and we have learned that instead.

The matched negatives are the cohort's OWN parent reads, so the matched arm also
asks the sharper question: can you tell host A + partner B from host A alone?

⛔ DESIGN NOTE - a first version of this script got this wrong and the built-in
control caught it. It used C2/C5's 224 clusters as the matched negative class for
EVERY cohort, which lifted C1 and C3 by +0.10 too - cohorts that do not use those
clusters at all and therefore must not move. All that measured was that those 224
references happen to give lower change-point scores than an average C0_REF read.
True, and it is the composition effect itself, but it is not a controlled test.

So the arms are now built to differ in EXACTLY ONE thing:
  both arms     224 negatives, one read per cluster, and a 700-read baseline drawn
                from clusters used as negatives by no cohort in that arm
  unmatched     those 224 clusters sampled from all 1,550
  matched       those 224 clusters sampled from the COHORT'S OWN clusters
C1 and C3 use all 1,550 clusters, so for them the two arms are the same draw by
construction and their delta MUST be ~0. That is the control. If C1 or C3 moves
by more than noise, stop - the arms are not clean and no other row can be read.

COST: n_neg is 224 in both arms, so the Hanley-McNeil SE is about 0.025
throughout - the same in both columns, which is what makes the delta readable.

=============================================================================
Q2. LOCALISATION, WITH A METRIC THAT CAN ACTUALLY BE READ
=============================================================================
e4 scored localisation as "fraction of argmax inside the ambiguity interval".
Measured interval widths (median): T2 19 bp, T3 12, T4 9, T5 6, T6 4.
So at T6 that metric demands the estimate land within +/- 2 bases. For scale,
the 2024 MBE benchmark of breakpoint localisation reports mean errors of 37 bp
(3SEQ), 74 bp (MaxChi) and 143 bp (GARD) for alignment methods that are HANDED
the parent sequences. Nobody localises to +/- 2 bases.

So this script reports, per tier and per channel:
  median |error| in bp   <- PRIMARY. Comparable across tiers and to the literature.
  signed median error    <- the coordinate check. Should be ~0. A constant
                            non-zero value is an off-by-one, not biology.
  inside-interval %      <- secondary, strict.
  permutation chance     <- for BOTH, by reassigning each read's argmax to
                            another read's ground truth, 200 times. This is
                            exact: it preserves the real argmax distribution and
                            the real interval-width distribution. e4's
                            width/222 chance line did not, and flattered us --
                            truth only ever lives in bases 80-230 while the
                            argmax may land anywhere in 70-291, so an estimator
                            that always guessed 155 would have beaten it.

CHANNELS
--------
probs4     the 4-vector of predicted base probabilities, d=4
entropy    -SUM p ln p over A/C/G/T, d=1
pmax_pobs  p(most likely base) - p(observed base), d=1.   ** Maia, 2026-09-21 **
           Zero wherever the model's favourite base IS the one in the read, and
           it grows with how badly the model wanted something else. Two reasons
           it is the right shape for a C2 junction, not just another channel:
             1. It is DIRECTIONAL in the way the physics is. Past the junction
                the model is still conditioned on parent A while reading parent
                B, so wherever B differs from A the model should prefer A's
                base - which is exactly what this quantity measures. The NLL
                collapses that to "surprised", which is symmetric and carries
                no direction.
             2. It is SELF-NORMALISING per position. In a genuinely hard region
                p(top1) is low and p(observed) is low together, so the
                difference stays small. -log p(observed) has no such reference
                level and reads "hard position" and "wrong lineage" alike.
           Needs the read sequence, which the i1 docstring called unavailable.
           It is available: phase1_all.fasta, all 336,895 reads, pure ACGT, all
           312 bp (checked).
logodds    log p(top1) - log p(observed), d=1. The scale-free version of the
           same idea, and identical to the NLL minus the smallest NLL achievable
           at that position. EXPLORATORY, not pre-registered - it is listed so
           the two can be compared once, not so the better one can be picked and
           reported as if it had been chosen in advance.
emb64      the projected embedding truncated to 64 dims (--with-emb), d=64

COORDINATES, the one real trap here
-----------------------------------
  embeddings: 312 tokens, token t is base t                  -> est_base = b + WARM
  probs:      311 rows,  row i is the distribution over base i+1
              a split at row b means base b+1 is the first disturbed one
                                                              -> est_base = b + WARM + 1
Both offsets are applied below and BOTH are checked by the signed median, which
is printed for every channel. Do not trust the localisation numbers until the
signed medians are near zero.
"""
import numpy as np, pandas as pd, time, sys

T0 = time.time()
WITH_EMB = '--with-emb' in sys.argv

SEEDS = [20260921, 11, 202, 3003, 40004]
NBASE, NC0, NPER = 700, 850, 1500
NPERM = 200

ROOT  = '../../../../tesis_run/evo2_results_h100/phase1_all/'
PROBS = ROOT + 'Tier 2/probs_acgt.f16'
PIDX  = ROOT + 'Tier 1/probs_acgt_index.csv'
PROJ  = ROOT + 'Tier 4/emb_proj.f16'
EIDX  = ROOT + 'Tier 1/emb_proj_index.csv'
FRAME = '../../../analysis_s26/ANALYSIS_FRAME_2026-09-16.csv'
CACHE = '../../../analysis_s25/c2_cache.npz'
SEQS  = ('../../../data_generation/2_build_contamination_cohorts/'
         'phase1_output_stage2_2026-09-02/phase1_all.fasta')

LP, CP = 311, 5        # probs:      311 rows x 5 stored values (A,C,G,T,residual)
LE, DE = 312, 128      # embeddings: 312 tokens x 128 projected dims
KEMB   = 64            # e4/e5 peak
WARM, MINSEG = 50, 20
EPS = 1e-7

COH = ['C1_illumina', 'C2_chimera_T2', 'C2_chimera_T3', 'C2_chimera_T4',
       'C2_chimera_T5', 'C2_chimera_T6', 'C5_stacked_T2', 'C3_insert']
C2COH = [c for c in COH if c.startswith('C2_')]

# ---------------------------------------------------------------- statistic
# IDENTICAL to e4/e5/i1. Nothing here is new; only what we feed it and what we
# keep from it changes.
def rank(a):
    n = a.size; o = np.argsort(a, kind='mergesort'); r = np.empty(n); r[o] = np.arange(1, n + 1)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[o[j + 1]] == a[o[i]]: j += 1
        if j > i: r[o[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return r

def auroc(p, n):
    p = np.asarray(p, float); n = np.asarray(n, float)
    return (rank(np.concatenate([p, n]))[:p.size].sum() - p.size * (p.size + 1) / 2) / (p.size * n.size)


# ---- s37 PATCH (addendum 8): rank standardisation against the baseline reads, per position and dimension
def _norm_ppf(u):
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00]
    u=np.asarray(u,float); z=np.empty_like(u); lo,hi=u<0.02425,u>1-0.02425; mid=~(lo|hi)
    q=np.sqrt(-2*np.log(u[lo])); z[lo]=(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=np.sqrt(-2*np.log(1-u[hi])); z[hi]=-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q=u[mid]-.5; r=q*q; z[mid]=(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return z
def _rankz(V, SORTED):
    V=np.asarray(V,np.float64); Z=np.empty(V.shape); N=SORTED.shape[0]
    for j in range(V.shape[1]):
        for k in range(V.shape[2]):
            s=SORTED[:,j,k]; lt=np.searchsorted(s,V[:,j,k],'left'); le=np.searchsorted(s,V[:,j,k],'right')
            Z[:,j,k]=_norm_ppf((lt+.5*(le-lt)+.5)/(N+1))
    return Z
def cusum(V, MU, SD, offset):
    """V (n, Lw, d) already warm-trimmed. Returns (score, est_base)."""
    V = _rankz(V, MU) if SD is None else (V - MU[None]) / SD[None]
    n, Lw, d = V.shape
    cs = np.cumsum(V, axis=1); tot = cs[:, -1, :]
    b = np.arange(MINSEG, Lw - MINSEG)
    left  = cs[:, b - 1, :] / b[None, :, None]
    right = (tot[:, None, :] - cs[:, b - 1, :]) / (Lw - b)[None, :, None]
    S = (b * (Lw - b) / Lw)[None, :] * ((left - right) ** 2).sum(2)
    tr = np.maximum(np.diff(V, axis=1).var(1).sum(1) / 2.0, 1e-9)
    return S.max(1) / tr, b[S.argmax(1)] + WARM + offset

# ---------------------------------------------------------------- readers
def _read(path, rows, L, C):
    rowb = L * C * 2
    out = np.empty((len(rows), L, C), np.float16)
    with open(path, 'rb') as f:
        for r in sorted(range(len(rows)), key=lambda z: rows[z]):
            f.seek(int(rows[r]) * rowb)
            out[r] = np.frombuffer(f.read(rowb), dtype=np.float16).reshape(L, C)
    return out

B2I = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
LUT = np.full(256, -1, np.int8)
for _b, _i in B2I.items(): LUT[ord(_b)] = _i

def load_seqs(path):
    d = {}; sid = None; buf = []
    for line in open(path):
        if line[0] == '>':
            if sid: d[sid] = ''.join(buf)
            sid = line[1:].split()[0]; buf = []
        else: buf.append(line.strip())
    if sid: d[sid] = ''.join(buf)
    return d

def observed_index(ids):
    """(n, 311) int index of the base each probs row is predicting.
    probs row i is the distribution over base i+1, so the observed base for
    row i is seq[i+1]: rows 0..310 <-> bases 1..311 of a 312 bp read."""
    out = np.empty((len(ids), LP), np.int8)
    for r, sid in enumerate(ids):
        out[r] = LUT[np.frombuffer(SEQ[sid][1:].encode(), np.uint8)]
    assert out.min() >= 0, 'non-ACGT base in a read - observed-base channels invalid'
    return out

def probs_channels(X, ids):
    """X (n,311,5) -> {name: (n, Lw, d)} already warm-trimmed."""
    p = X[:, :, :4].astype(np.float32)
    p = np.clip(p, EPS, None); p /= p.sum(2, keepdims=True)
    H = -(p * np.log(p)).sum(2)                # Shannon entropy, natural log
    obs = observed_index(ids)
    pobs = np.take_along_axis(p, obs[:, :, None].astype(np.intp), axis=2)[:, :, 0]
    pmax = p.max(2)
    return {'probs4':    p[:, WARM:, :],
            'entropy':   H[:, WARM:, None],
            'pmax_pobs': (pmax - pobs)[:, WARM:, None],
            'logodds':   (np.log(pmax) - np.log(pobs))[:, WARM:, None]}

def emb_channel(X):
    return {'emb64': X[:, WARM:, :KEMB].astype(np.float32)}

# offsets from the coordinate derivation in the docstring
OFFSET = {'probs4': 1, 'entropy': 1, 'pmax_pobs': 1, 'logodds': 1, 'emb64': 0}

# ---------------------------------------------------------------- load
pidx = pd.read_csv(PIDX)
pc = [c for c in pidx.columns if 'id' in c.lower()][0]
pr = [c for c in pidx.columns if c != pc][0]
prow = dict(zip(pidx[pc], pidx[pr]))
if WITH_EMB:
    eidx = pd.read_csv(EIDX)
    ec = [c for c in eidx.columns if 'id' in c.lower()][0]
    er = [c for c in eidx.columns if c != ec][0]
    erow = dict(zip(eidx[ec], eidx[er]))

fr = pd.read_csv(FRAME, usecols=['seq_id', 'cohort', 'host_cluster'])
POOL = {c: [s for s in fr.loc[fr.cohort == c, 'seq_id'] if s in prow] for c in ['C0_REF'] + COH}

# --- cluster bookkeeping ---------------------------------------------------
c0 = fr[fr.cohort == 'C0_REF']
cl2id = {cl: sid for cl, sid in zip(c0.host_cluster, c0.seq_id) if sid in prow}
ALL_CL = np.array(sorted(cl2id))
HC = {c: np.array(sorted(set(fr.loc[fr.cohort == c, 'host_cluster'].dropna()) & set(cl2id)))
      for c in COH}
NNEG = int(min(len(HC[c]) for c in COH))       # 224, set by C2/C5
print(f'C0_REF clusters {len(ALL_CL)} | negatives per arm {NNEG} | clusters per cohort: '
      + '  '.join(f'{c.split("_")[0]}{c[-2:] if c[-2] == "T" else ""}:{len(HC[c])}' for c in COH))
# at most 2 distinct negative pools per arm (see the draw below), so the worst
# case leaves len(ALL_CL) - 2*NNEG clusters free for the baseline
assert len(ALL_CL) - 2 * NNEG >= NBASE, 'not enough clusters left for a disjoint baseline'

print('loading read sequences for the observed-base channels ...', flush=True)
SEQ = load_seqs(SEQS)
print(f'  {len(SEQ)} sequences loaded')

z = np.load(CACHE, allow_pickle=True)
BP = dict(zip(z['ids'], z['bp'])); LO = dict(zip(z['ids'], z['amb_lo'])); HI = dict(zip(z['ids'], z['amb_hi']))

# ---------------------------------------------------------------- run
CH = ['probs4', 'entropy', 'pmax_pobs', 'logodds'] + (['emb64'] if WITH_EMB else [])
PROBS_CH = ['probs4', 'entropy', 'pmax_pobs', 'logodds']
det_rows, loc_rows = [], []

for si, seed in enumerate(SEEDS, 1):
    rng = np.random.default_rng(seed)

    # Build both arms: per-cohort negative clusters, then ONE baseline per arm
    # drawn from clusters this arm never uses as a negative.
    ARMS = {}
    for arm in ('unmatched', 'matched'):
        # ONE draw per distinct cluster pool, shared by every cohort that has
        # that pool. In the unmatched arm there is a single pool (all 1,550), so
        # all eight cohorts face the same negatives - which is what e5/i1 did.
        # In the matched arm there are two: the 224-cluster pool that C2 and C5
        # were built from, and the all-1,550 pool for C1 and C3.
        # ⛔ Drawing independently per cohort (the first version) put up to
        # 8 x 224 = ~1,100 distinct clusters into the "already used as a
        # negative" set and left too few for a disjoint 700-read baseline.
        # Sharing is also the more correct comparison: cohorts with identical
        # cluster pools SHOULD face identical negatives.
        pools, negcl = {}, {}
        for c in COH:
            key = 'ALL' if arm == 'unmatched' else HC[c].tobytes()
            if key not in pools:
                src = ALL_CL if arm == 'unmatched' else HC[c]
                pools[key] = rng.choice(src, NNEG, replace=False)
            negcl[c] = pools[key]
        used = set().union(*[set(v.tolist()) for v in pools.values()])
        free = np.array([x for x in ALL_CL if x not in used])
        assert len(free) >= NBASE, (f'only {len(free)} clusters left for the baseline in the '
                                    f'{arm} arm, need {NBASE}')
        base_ids = [cl2id[x] for x in rng.choice(free, NBASE, replace=False)]
        ARMS[arm] = (base_ids, {c: [cl2id[x] for x in negcl[c]] for c in COH})

    # read each cohort's raw arrays ONCE, reuse for both arms
    sel = {c: list(rng.choice(POOL[c], min(NPER, len(POOL[c])), replace=False)) for c in COH}
    raw = {}
    for c in COH:
        raw[c] = {'probs': _read(PROBS, [prow[i] for i in sel[c]], LP, CP)}
        if WITH_EMB: raw[c]['emb'] = _read(PROJ, [erow[i] for i in sel[c]], LE, DE)

    for arm, (base_ids, neg_by_cohort) in ARMS.items():
        # baseline mu/sd per position
        MU, SD = {}, {}
        Bp = probs_channels(_read(PROBS, [prow[i] for i in base_ids], LP, CP), base_ids)
        for k, v in Bp.items(): MU[k] = np.sort(np.asarray(v, np.float64), 0); SD[k] = None
        if WITH_EMB:
            Be = emb_channel(_read(PROJ, [erow[i] for i in base_ids], LE, DE))
            for k, v in Be.items(): MU[k] = np.sort(np.asarray(v, np.float64), 0); SD[k] = None
            del Be
        del Bp

        # negatives are per cohort now (that IS the variable under test), so they
        # are scored inside the cohort loop against the same MU/SD.
        for c in COH:
            neg_ids = neg_by_cohort[c]
            neg = {}
            np_ = probs_channels(_read(PROBS, [prow[i] for i in neg_ids], LP, CP), neg_ids)
            for k in PROBS_CH: neg[k] = cusum(np_[k], MU[k], SD[k], OFFSET[k])[0]
            del np_
            if WITH_EMB:
                ne = emb_channel(_read(PROJ, [erow[i] for i in neg_ids], LE, DE))
                neg['emb64'] = cusum(ne['emb64'], MU['emb64'], SD['emb64'], OFFSET['emb64'])[0]
                del ne

            chs = probs_channels(raw[c]['probs'], sel[c])
            if WITH_EMB: chs.update(emb_channel(raw[c]['emb']))
            for k in CH:
                sc, est = cusum(chs[k], MU[k], SD[k], OFFSET[k])
                det_rows.append({'seed': seed, 'arm': arm, 'cohort': c, 'channel': k,
                                 'auroc': round(auroc(sc, neg[k]), 4),
                                 'n_neg': len(neg_ids)})
                # localisation: only C2 has a breakpoint ground truth
                if c in C2COH and arm == 'matched':
                    ids = sel[c]
                    ok = [j for j, s in enumerate(ids) if s in BP]
                    if len(ok) < 20: continue
                    e  = est[ok].astype(float)
                    bp = np.array([BP[ids[j]] for j in ok], float)
                    lo = np.array([LO[ids[j]] for j in ok], float)
                    hi = np.array([HI[ids[j]] for j in ok], float)
                    err = e - bp
                    ins = ((e >= lo) & (e <= hi)).mean()
                    # permutation chance: keep the argmaxes, shuffle WHICH truth they face
                    pr_abs, pr_ins = [], []
                    for _ in range(NPERM):
                        q = rng.permutation(len(ok))
                        pr_abs.append(np.median(np.abs(e - bp[q])))
                        pr_ins.append(((e >= lo[q]) & (e <= hi[q])).mean())
                    loc_rows.append({'seed': seed, 'cohort': c, 'channel': k, 'n': len(ok),
                                     'med_abs_err': round(float(np.median(np.abs(err))), 2),
                                     'signed_med_err': round(float(np.median(err)), 2),
                                     'chance_med_abs_err': round(float(np.mean(pr_abs)), 2),
                                     'inside_pct': round(100 * ins, 2),
                                     'chance_inside_pct': round(100 * float(np.mean(pr_ins)), 2),
                                     'med_interval_w': float(np.median(hi - lo + 1))})
            del chs
    del raw
    print(f'[seed {si}/{len(SEEDS)}] done ({time.time() - T0:.0f}s)', flush=True)

det = pd.DataFrame(det_rows); det.to_csv('j1_detection_per_seed.csv', index=False)
loc = pd.DataFrame(loc_rows); loc.to_csv('j1_localisation_per_seed.csv', index=False)

# ---------------------------------------------------------------- report
print('\n' + '=' * 96)
print('Q1  DETECTION AUROC  -  unmatched vs composition-matched negative class')
print('=' * 96)
print('n_neg is 224 in BOTH arms, so the SE is about 0.025 in both columns.')
print('Treat a delta smaller than ~0.035 as noise.\n')
for k in CH:
    print(f'--- channel {k}')
    print(f'{"cohort":<20}{"unmatched":>14}{"matched":>14}{"delta":>10}')
    for c in COH:
        u = det[(det.channel == k) & (det.arm == 'unmatched') & (det.cohort == c)]['auroc']
        m = det[(det.channel == k) & (det.arm == 'matched')   & (det.cohort == c)]['auroc']
        tag = '  <- CONTROL, must be ~0' if c in ('C1_illumina', 'C3_insert') else ''
        print(f'{c:<20}{u.mean():>8.3f}±{u.std(ddof=1):<5.3f}{m.mean():>8.3f}±{m.std(ddof=1):<5.3f}'
              f'{m.mean() - u.mean():>+10.3f}{tag}')
    print()
print('READ IT LIKE THIS')
print('  C1 and C3 draw negatives from all 1,550 clusters in BOTH arms, so their delta must')
print('  be ~0. If either moves by more than ~0.035, the arms are not clean - stop and fix.')
print('  C2-shallow rising toward 0.50 and C2-T6 holding or rising = the composition')
print('  explanation is right. C2-T6 unchanged = it is wrong, and the 0.41 stays unexplained.')

print('\n' + '=' * 96)
print('Q2  LOCALISATION  (C2 only - the only cohorts with a breakpoint ground truth)')
print('=' * 96)
print(f'{"cohort":<18}{"chan":<9}{"med|err| bp":>12}{"chance":>9}{"signed med":>12}'
      f'{"inside%":>9}{"chance%":>9}{"interval w":>11}')
for c in C2COH:
    for k in CH:
        g = loc[(loc.cohort == c) & (loc.channel == k)]
        if not len(g): continue
        print(f'{c:<18}{k:<9}{g.med_abs_err.mean():>12.1f}{g.chance_med_abs_err.mean():>9.1f}'
              f'{g.signed_med_err.mean():>12.1f}{g.inside_pct.mean():>9.1f}'
              f'{g.chance_inside_pct.mean():>9.1f}{g.med_interval_w.mean():>11.1f}')
print('\nREAD IT LIKE THIS')
print('  signed med     must be near 0. A constant offset of +/-1 or +/-2 that is the SAME')
print('                 in every tier is a coordinate bug, not biology. Fix OFFSET and re-run.')
print('  med|err| bp    the number to quote. Literature reference points, for methods that are')
print('                 GIVEN the parents: 3SEQ 37 bp, MaxChi 74 bp, GARD 143 bp.')
print('  chance         from permuting which ground truth each argmax faces, so it already')
print('                 contains any positional bias in the argmax. Beating it is the claim.')
print(f'\nfiles: j1_detection_per_seed.csv  j1_localisation_per_seed.csv   total {time.time() - T0:.0f}s')
