#!/usr/bin/env python3
"""
j3 - build the head-to-head benchmark input: OUR channel vs uchime, same reads.

WHY uchime_ref AND NOT uchime_denovo
uchime_denovo finds a chimera only when BOTH parents are present in the same
file at higher abundance than the read. Our C2 reads' parent B is usually a
MIDORI donor that is not in any cohort, so denovo would be handicapped by our
file layout rather than by its own ability - an unfair win for us.
uchime_ref instead gets references_N1565.fasta, which CONTAINS parent A and most
parent Bs. That is the strongest possible version of the baseline: the
alignment-based method is handed the actual reference database, while Evo2 sees
nothing but the read. If we match it under those conditions, the comparison
means something.

⛔ IDENTICAL READS. The read ids come from j2_worked_vs_missed.csv, so uchime is
scored on exactly the reads our number was computed on, plus the same 224
composition-matched negatives. No re-sampling.
"""
import numpy as np, pandas as pd
exec(open('j1_matched_and_localisation.py').read().split(
     "# ---------------------------------------------------------------- run")[0])

d = pd.read_csv('j2_worked_vs_missed.csv')
rng = np.random.default_rng(20260921)
negcl = rng.choice(HC['C2_chimera_T6'], NNEG, replace=False)     # same draw as j2
neg_ids = [cl2id[x] for x in negcl]

rows = []
with open('j3_benchmark.fasta', 'w') as f:
    for _, r in d.iterrows():
        f.write(f'>{r.id};size=1\n{SEQ[r.id]}\n')
        rows.append({'id': r.id, 'cohort': r.cohort, 'is_chimera': 1,
                     'evo2_score': r.score, 'evo2_err_bp': r.err})
    for s in neg_ids:
        f.write(f'>{s};size=1\n{SEQ[s]}\n')
        rows.append({'id': s, 'cohort': 'C0_REF', 'is_chimera': 0,
                     'evo2_score': np.nan, 'evo2_err_bp': np.nan})

# the negatives need OUR score too, on the same baseline j2 used
free = np.array([x for x in ALL_CL if x not in set(negcl.tolist())])
base_ids = [cl2id[x] for x in rng.choice(free, NBASE, replace=False)]
Bp = probs_channels(_read(PROBS, [prow[i] for i in base_ids], LP, CP), base_ids)
MU = {k: np.sort(np.asarray(v, np.float64), 0) for k, v in Bp.items()}
SD = {k: None for k, v in Bp.items()}
ns = cusum(probs_channels(_read(PROBS, [prow[i] for i in neg_ids], LP, CP), neg_ids)['logodds'],
           MU['logodds'], SD['logodds'], OFFSET['logodds'])[0]
t = pd.DataFrame(rows)
t.loc[t.cohort == 'C0_REF', 'evo2_score'] = ns
t.to_csv('j3_labels.csv', index=False)
print(f'j3_benchmark.fasta  {len(t)} reads  '
      f'({(t.is_chimera==1).sum()} chimeras, {(t.is_chimera==0).sum()} negatives)')
print('j3_labels.csv  with our logodds score per read')
