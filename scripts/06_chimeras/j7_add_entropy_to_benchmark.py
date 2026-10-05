#!/usr/bin/env python3
"""j7 - add Evo2's BEST DETECTION channel (entropy) to the benchmark table.
Figure 5 was comparing uchime against logodds, which is our best LOCALISER but not
our best detector. Showing only logodds understates Evo2's detection by 0.16 AUROC.
Both are now shown: logodds (best localiser) and entropy (best detector)."""
import numpy as np, pandas as pd
exec(open('j1_matched_and_localisation.py').read().split(
     "# ---------------------------------------------------------------- run")[0])
rng = np.random.default_rng(20260921)
negcl = rng.choice(HC['C2_chimera_T6'], NNEG, replace=False)
free = np.array([x for x in ALL_CL if x not in set(negcl.tolist())])
base_ids = [cl2id[x] for x in rng.choice(free, NBASE, replace=False)]
Bp = probs_channels(_read(PROBS,[prow[i] for i in base_ids],LP,CP), base_ids)
MU={k:np.sort(np.asarray(v,np.float64),0) for k,v in Bp.items()}; SD={k:None for k,v in Bp.items()}
bm = pd.read_csv('j4_benchmark_scored.csv')
ids = list(bm.id)
ch = probs_channels(_read(PROBS,[prow[i] for i in ids],LP,CP), ids)
bm['evo2_entropy'] = cusum(ch['entropy'], MU['entropy'], SD['entropy'], OFFSET['entropy'])[0]
bm.to_csv('j4_benchmark_scored.csv', index=False)
print('added evo2_entropy to j4_benchmark_scored.csv')
