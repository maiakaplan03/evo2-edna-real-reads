#!/usr/bin/env python3
"""
e4_check.py -- integrity check of the e4 results files.   python3 e4_check.py   (seconds)
For every family that has a results file:
  1 every comparison has every (score x classifier x split) row it should   -> lists anything MISSING
  2 rows present more than once are EXACT copies (same numbers)             -> any disagreement is flagged
  3 every null (shuffled labels) is centred near 0.5                        -> flags null medians outside 0.44-0.56
  4 rows written empty on purpose (a score that does not exist for some reads) are listed with their note
Writes results_<family>_clean.tsv (duplicates removed) for the figures step. Never modifies the original files.
"""
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
K = ["comparison", "T", "lib", "band", "feature", "classifier", "split"]
FEATS = [("ppl", "ridge"), ("gc", "ridge"), ("midori_id", "ridge"), ("prof_score0", "ridge"), ("codon3", "ridge"),
         ("embA", "ridge"), ("embA", "centroid"), ("embD", "ridge"), ("embD", "centroid"), ("nll", "ridge"),
         ("nll", "centroid"), ("acgt", "ridge"), ("acgt", "centroid"), ("kmer4", "ridge"), ("kmer4", "centroid"),
         ("onehot", "ridge"), ("onehot", "centroid")]
TOTAL = {"hidden": 14, "stop": 22, "evident": 47, "caught_missed": 4}
for fam, tot in TOTAL.items():
    f = HERE/f"results_{fam}.tsv"
    if not f.exists(): print(f"== {fam}: not run yet"); continue
    r = pd.read_csv(f, sep="\t")
    num = [c for c in ["auroc", "ci_lo", "ci_hi", "null_med"] if c in r]
    dup = r[r.duplicated(K, keep=False)]
    disagree = dup.groupby(K)[num].nunique(dropna=False).gt(1).any(axis=1).sum() if len(dup) else 0
    c = r.drop_duplicates(K)
    comps = c.groupby(["comparison", "T", "lib", "band"])
    missing = []
    for key, g in comps:
        have = set(zip(g.feature, g.classifier))
        for fc in FEATS:
            for sp in ("random", "cluster"):
                if fc not in have or sp not in set(g[(g.feature == fc[0]) & (g.classifier == fc[1])].split):
                    missing.append((key, fc, sp))
    miss_by_feat = pd.Series([m[1][0] + "/" + m[1][1] for m in missing]).value_counts().to_dict()
    empty = c[c.auroc.isna()] if "auroc" in c else c.iloc[0:0]
    nulls_off = c[(c.null_med < 0.44) | (c.null_med > 0.56)] if "null_med" in c else c.iloc[0:0]
    print(f"== {fam}: {comps.ngroups}/{tot} comparisons | {len(r)} rows written, {len(r) - len(c)} exact duplicates "
          f"({disagree} duplicated keys with DIFFERENT numbers)")
    print(f"   missing rows: {len(missing)}" + (f"  by score: {miss_by_feat}" if missing else ""))
    print(f"   rows with no AUROC: {len(empty)}" + (f"  notes: {empty.note.dropna().unique().tolist() if 'note' in empty else ''}" if len(empty) else ""))
    print(f"   nulls outside 0.44-0.56: {len(nulls_off)}" + (f"  -> {nulls_off[K + ['null_med']].to_string(index=False)}" if len(nulls_off) else ""))
    c.to_csv(HERE/f"results_{fam}_clean.tsv", sep="\t", index=False)
