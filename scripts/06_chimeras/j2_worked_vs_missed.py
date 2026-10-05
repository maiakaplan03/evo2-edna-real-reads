#!/usr/bin/env python3
"""
j2 - WHY does localisation work on about half the deep chimeras and not the rest?
CPU only. Run from nn_project/analysis_s32/.

j1 showed the C2-T6 error distribution is a MIXTURE, not a uniformly accurate
estimator: p25 = 4 bp, median 18 bp, p90 = 118 bp, with 51.9 % of reads inside
20 bp. A median alone would hide that. This script asks what separates the reads
it pins from the reads it misses, using only quantities that exist independently
of the estimator.

THE CANDIDATE EXPLANATION, stated before looking: a junction is only findable if
the two parents actually DIFFER near it. Where parent B happens to match parent A
for a long stretch after the breakpoint, there is no evidence in the read at all
and no method - ours, uchime's or a human's - can place it. If that is right, the
misses are a property of the DATA, not a failure of the channel, and the honest
claim becomes "localises to a few bases wherever the evidence exists".

PREDICTORS (all computed from the read and its parent A, never from the estimate)
  gap_to_eddence   bases from the breakpoint to the FIRST position where the read
                   differs from parent A. This is the direct measure of "how soon
                   does evidence start", and it is the one the explanation names.
  local_evidence   mismatches vs parent A within [bp, bp+40)
  total_evidence   mismatches vs parent A over the whole downstream half
  amb_width        the stored ambiguity interval hi - lo + 1
  parent_distance  from the FASTA header
  bp               breakpoint position - a check that it is not just an edge effect
  prefix_noise     mismatches vs parent A UPSTREAM of bp (the error layer). A
                   control: it should NOT predict success.
"""
import numpy as np, pandas as pd, time
T0 = time.time()
exec(open('j1_matched_and_localisation.py').read().split(
     "# ---------------------------------------------------------------- run")[0])

CHAN   = 'logodds'
HITBP  = 20                      # "localised" = within 20 bp, as in j1's table
COHS   = ['C2_chimera_T5', 'C2_chimera_T6']
NREAD  = 1500
rng = np.random.default_rng(20260921)

# baseline, matched arm, exactly as j1 builds it
negcl = rng.choice(HC['C2_chimera_T6'], NNEG, replace=False)
free = np.array([x for x in ALL_CL if x not in set(negcl.tolist())])
base_ids = [cl2id[x] for x in rng.choice(free, NBASE, replace=False)]
Bp = probs_channels(_read(PROBS, [prow[i] for i in base_ids], LP, CP), base_ids)
MU = {k: np.sort(np.asarray(v, np.float64), 0) for k, v in Bp.items()}
SD = {k: None for k, v in Bp.items()}
neg = cusum(probs_channels(_read(PROBS, [prow[i] for i in [cl2id[x] for x in negcl]], LP, CP),
                           [cl2id[x] for x in negcl])[CHAN],
            MU[CHAN], SD[CHAN], OFFSET[CHAN])[0]

# headers carry parent_a / breakpoint / parent_distance
META = {}
with open(SEQS) as f:
    for line in f:
        if line[0] == '>':
            q = line[1:].strip().split()
            META[q[0]] = {k: v for k, v in (t.split('=', 1) for t in q[1:] if '=' in t)}

rows = []
for c in COHS:
    ids = [s for s in rng.choice(POOL[c], NREAD, replace=False) if s in BP]
    ch  = probs_channels(_read(PROBS, [prow[i] for i in ids], LP, CP), ids)
    sc, est = cusum(ch[CHAN], MU[CHAN], SD[CHAN], OFFSET[CHAN])
    for j, s in enumerate(ids):
        m = META[s]; pa = 'C0_REF_' + m['parent_a']
        if pa not in SEQ: continue
        bp = int(BP[s]); r = SEQ[s]; a = SEQ[pa]
        dn = [i for i in range(bp, len(r)) if r[i] != a[i]]
        up = sum(1 for i in range(0, bp) if r[i] != a[i])
        rows.append({'cohort': c, 'id': s, 'bp': bp,
                     'err': abs(float(est[j]) - bp), 'score': float(sc[j]),
                     'gap_to_evidence': (dn[0] - bp) if dn else 999,
                     'local_evidence': sum(1 for i in dn if i < bp + 40),
                     'total_evidence': len(dn),
                     'amb_width': int(HI[s] - LO[s] + 1),
                     'parent_distance': float(m.get('parent_distance', 'nan')),
                     'prefix_noise': up})
d = pd.DataFrame(rows)
d['hit'] = d.err <= HITBP
d.to_csv('j2_worked_vs_missed.csv', index=False)

P = ['gap_to_evidence', 'local_evidence', 'total_evidence', 'amb_width',
     'parent_distance', 'bp', 'prefix_noise']
print(f'\nchannel {CHAN}   "localised" = |error| <= {HITBP} bp   n per cohort ~{NREAD}\n')
for c in COHS:
    g = d[d.cohort == c]; h, m_ = g[g.hit], g[~g.hit]
    print(f'=== {c}   localised {len(h)}/{len(g)} ({100*len(h)/len(g):.1f} %)')
    print(f'{"predictor":<18}{"localised (med)":>17}{"missed (med)":>15}{"AUROC":>9}{"":>3}')
    for p_ in P:
        A = auroc(h[p_].values, m_[p_].values)
        star = ' **' if abs(A - .5) > .10 else (' *' if abs(A - .5) > .05 else '')
        print(f'{p_:<18}{h[p_].median():>17.1f}{m_[p_].median():>15.1f}{A:>9.3f}{star}')
    print(f'{"detection score":<18}{h.score.median():>17.2f}{m_.score.median():>15.2f}'
          f'{auroc(h.score.values, m_.score.values):>9.3f}')
    print(f'  (for reference, the negatives score a median of {np.median(neg):.2f})')
    # the headline restatement
    ev = g[g.gap_to_evidence <= 10]
    print(f'  reads whose evidence starts within 10 bp of the breakpoint: n={len(ev)}, '
          f'median |error| {ev.err.median():.0f} bp, localised {100*ev.hit.mean():.1f} %')
    ev2 = g[g.gap_to_evidence > 10]
    print(f'  reads whose evidence starts later than that:                n={len(ev2)}, '
          f'median |error| {ev2.err.median():.0f} bp, localised {100*ev2.hit.mean():.1f} %\n')
print(f'AUROC here = P(predictor is higher on a localised read). 0.5 = no relation.')
print(f'wrote j2_worked_vs_missed.csv   total {time.time()-T0:.0f}s')
